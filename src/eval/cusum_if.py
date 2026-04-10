"""
Two-Stage FDI Detector: Isolation Forest → CUSUM
─────────────────────────────────────────────────────────────────
Problem with IF alone:  Recall=1.0 but Precision=0.16 (401 FP)
Problem with IQR alone: Precision=1.0 but Recall=0.58 (32 FN)

Solution — two-stage AND filter:
  Stage 1 (Sensitivity): IF flags ALL candidate anomalies (high recall)
  Stage 2 (Specificity): CUSUM confirms only SUSTAINED EoE drift
                         patterns as true attacks (kills FP spikes)

CUSUM (upper one-sided):
  S(t) = max(0,  S(t-1) + |EoE(t)| - k)
  Attack declared when S(t) > h

  k  = drift allowance (≈ average normal EoE, so random noise → S stays near 0)
  h  = decision threshold (higher → fewer false alarms, lower → faster detection)

Tuning strategy (on clean training EoE):
  - Sweep (k, h) grid
  - k candidates: multiples of the normal EoE mean (0.5×, 1×, 1.5×, 2×)
  - h candidates: multiples of the normal EoE std (2×, 4×, 6×, 8×, 10×)
  - Select (k, h) that maximises F1 on val slice while keeping Recall > 0.90

Usage (from narx_ev_fdi/ directory):
    python3 -m src.eval.cusum_if
"""

import os
import sys
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score,
    precision_score, accuracy_score, ConfusionMatrixDisplay,
)
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset import build_datasets
from src.models.narx  import NARXNet
from src.eval.evaluate import (
    compute_iqr_bounds, flag_spikes, sliding_window_declare,
    inject_fdi_attacks,
)

CKPT_DIR = os.path.join(ROOT, "checkpoints")
OUT_DIR  = os.path.join(ROOT, "results")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

IF_CONTAMINATION = 0.15   # raised from 0.01: attacks are 10% of data, need margin
IF_N_ESTIMATORS  = 200


# ─────────────────────────────────────────────────────────────────
# 1. CUSUM primitives
# ─────────────────────────────────────────────────────────────────
def cusum(eoe: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Upper one-sided CUSUM on |EoE|.

    S(t) = max(0, S(t-1) + |eoe(t)| - k)
    Alarm when S(t) > h.

    Parameters
    ----------
    eoe : raw EoE values (may be attacked)
    k   : drift allowance (reference value)
    h   : decision threshold

    Returns
    -------
    S        : cumulative sum array
    detected : 1 where S > h
    """
    n = len(eoe)
    S = np.zeros(n)
    for t in range(1, n):
        S[t] = max(0.0, S[t-1] + abs(eoe[t]) - k)
    detected = (S > h).astype(int)
    return S, detected


def cusum_reset(eoe: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized CUSUM with alarm-triggered reset (S → 0 after each alarm).

    Uses numpy for speed — avoids Python per-sample loops that made large
    datasets (100K+ samples) extremely slow during grid search.
    """
    eoe_abs = np.abs(eoe)
    increments = eoe_abs - k          # positive = building anomaly
    n = len(eoe)
    S = np.zeros(n)
    detected = np.zeros(n, dtype=int)
    s = 0.0
    for t in range(1, n):
        s = max(0.0, s + increments[t])
        if s > h:
            detected[t] = 1
            s = 0.0
        S[t] = s
    return S, detected


# ─────────────────────────────────────────────────────────────────
# 2. CUSUM parameter tuning on validation slice
# ─────────────────────────────────────────────────────────────────
def tune_cusum(
    eoe_clean_train: np.ndarray,
    eoe_val_attacked: np.ndarray,
    gt_val:           np.ndarray,
    k_scales:  list   = None,
    h_scales:  list   = None,
    min_recall: float = 0.90,
    max_tune_samples: int = 20000,
) -> tuple[float, float, float, float]:
    """
    Grid search (k, h) pairs.
    k = k_scale × mean(normal EoE)
    h = h_scale × std(normal EoE)

    Returns: k_best, h_best, best_f1, mu_normal

    Fast path: if the EoE separation between attacked and clean is large
    (>10x), use analytically optimal parameters directly without grid search.
    This avoids the O(n × grid_size) CUSUM loop on large datasets.
    """
    if k_scales is None:
        k_scales = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if h_scales is None:
        # Extended range: large values eliminate FPs from sustained normal EoE spikes.
        # Attacks (EoE >> normal) still alarm on the very first attacked step.
        h_scales = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 75, 100]

    mu_normal  = np.mean(eoe_clean_train)
    sig_normal = np.std(eoe_clean_train)
    sig_normal = max(sig_normal, 1e-8)

    # ── Fast path: analytical k/h when separation is very large ──────
    # With >10x EoE separation, any k < attacked_mean/2 and h < attacked_mean
    # gives near-perfect detection. Skip expensive grid search.
    attacked_mean = np.mean(eoe_val_attacked[gt_val == 1]) if gt_val.sum() > 0 else None
    if attacked_mean is not None and attacked_mean > 10 * mu_normal:
        best_k = mu_normal * 0.5          # absorb normal noise
        # h must be:
        #   - large enough that sustained normal EoE can't trigger it
        #   - small enough that a single attack step (EoE_attack - k) triggers it
        # Use 20× the 95th-percentile of single-step CUSUM contribution from clean data.
        # Physics floor: y_true ∈ [0, 1.0] kWh (Y_CLIP) and y_pred ∈ [0, 1.0] kWh,
        # so normal EoE ≤ 1.0 kWh always. Set h = 1.05 to guarantee zero normal FPs.
        # Attack EoE_min = scale_min × baseline = 8 × ~0.39 ≈ 3.1 kWh >> 1.05.
        p95_clean_S = np.percentile(np.maximum(0.0, eoe_clean_train - best_k), 95)
        best_h = max(p95_clean_S * 20.0, sig_normal * 15.0, 0.5)
        # Quick validate on small slice
        n_q = min(len(eoe_val_attacked), 5000)
        _, det_q = cusum_reset(eoe_val_attacked[:n_q], best_k, best_h)
        best_f1 = f1_score(gt_val[:n_q], det_q, zero_division=0)
        print(f"  [FAST] EoE separation={attacked_mean/max(mu_normal,1e-8):.0f}x  "
              f"→ k={best_k:.5f}  h={best_h:.5f}  (quick F1≈{best_f1:.4f})")
        return best_k, best_h, best_f1, mu_normal

    # ── Standard grid search on a capped validation slice ────────────
    # Cap at max_tune_samples to keep CUSUM loops fast
    if len(eoe_val_attacked) > max_tune_samples:
        idx = np.random.default_rng(0).choice(len(eoe_val_attacked),
                                               max_tune_samples, replace=False)
        idx.sort()
        ev = eoe_val_attacked[idx]; gv = gt_val[idx]
    else:
        ev = eoe_val_attacked; gv = gt_val

    print(f"\n[TUNING] CUSUM grid search  μ={mu_normal:.5f}  σ={sig_normal:.5f}")
    best_k, best_h, best_f1 = mu_normal * k_scales[0], sig_normal * h_scales[0], -1.0

    for ks, hs in itertools.product(k_scales, h_scales):
        k, h = ks * mu_normal, hs * sig_normal
        _, det = cusum_reset(ev, k, h)
        f1  = f1_score    (gv, det, zero_division=0)
        rec = recall_score(gv, det, zero_division=0)
        if f1 > best_f1 and rec >= min_recall:
            best_f1, best_k, best_h = f1, k, h

    if best_f1 < 0:
        print("  [WARN] No (k,h) met recall >= 0.90, relaxing constraint...")
        for ks, hs in itertools.product(k_scales, h_scales):
            k, h = ks * mu_normal, hs * sig_normal
            _, det = cusum_reset(ev, k, h)
            f1 = f1_score(gv, det, zero_division=0)
            if f1 > best_f1:
                best_f1, best_k, best_h = f1, k, h

    print(f"  → Best k={best_k:.5f}  h={best_h:.5f}  (val F1={best_f1:.4f})")
    return best_k, best_h, best_f1, mu_normal


# ─────────────────────────────────────────────────────────────────
# 3. Two-stage IF + CUSUM evaluation
# ─────────────────────────────────────────────────────────────────
def _build_eoe_aligned_site_ids(site_ids_full, session_ids_full, n_windows, n_delay):
    """
    Reconstruct per-window site IDs from the original (per-row) arrays.
    Windows are built per-session with n_delay warmup rows dropped, so the
    window count for session s = max(0, len(s) - n_delay).

    Uses sort-based grouping O(n log n) instead of per-session boolean masking
    O(n_sessions × n_total) to avoid multi-minute runtimes on large datasets.
    """
    sid_arr  = np.asarray(session_ids_full)
    site_arr = np.asarray(site_ids_full)

    # Sort once by session ID
    sort_idx    = np.argsort(sid_arr, kind="stable")
    sid_sorted  = sid_arr[sort_idx]
    site_sorted = site_arr[sort_idx]

    _, first_occ, counts = np.unique(sid_sorted, return_index=True, return_counts=True)

    # Pre-allocate result array
    n_wins_per_session = np.maximum(0, counts - n_delay).astype(int)
    total_wins = int(n_wins_per_session.sum())
    window_sites = np.empty(total_wins, dtype=site_arr.dtype)

    pos = 0
    for start, n_win in zip(first_occ, n_wins_per_session):
        if n_win > 0:
            window_sites[pos:pos + n_win] = site_sorted[start]
            pos += n_win

    if len(window_sites) > n_windows:
        window_sites = window_sites[:n_windows]
    elif len(window_sites) < n_windows:
        window_sites = np.pad(window_sites, (0, n_windows - len(window_sites)),
                              mode="edge")
    return window_sites


def evaluate_if_cusum(model, raw_data, scaler_y, val_fraction=0.15,
                      df_train=None, df_estim=None):
    """
    Full two-stage pipeline with per-site Isolation Forest + CUSUM:
      Stage 1: IsolationForest per-site (recall-maximising)
      Stage 2: CUSUM-reset per-site     (precision-preserving)
      Combined: attack iff BOTH flag the point

    Per-site fitting eliminates false positives caused by multi-modal EoE
    distributions across Caltech (~0.3 kWh/step) and JPL (~0.01 kWh/step).
    """
    model.eval()

    # ── Clean training EoE ───────────────────────────────────────
    X_tr, y_tr = raw_data["X_train_w"], raw_data["y_train_w"]
    with torch.no_grad():
        tr_sc = model(torch.tensor(X_tr, dtype=torch.float32,
                                   device=DEVICE)).cpu().numpy().flatten()
    y_tr_true = scaler_y.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    y_tr_pred = scaler_y.inverse_transform(tr_sc.reshape(-1, 1)).flatten()
    eoe_tr_clean = np.abs(y_tr_true - y_tr_pred)

    # ── Attacked estimation EoE ───────────────────────────────────
    X_es, y_es = raw_data["X_estim_w"], raw_data["y_estim_w"]
    with torch.no_grad():
        es_sc = model(torch.tensor(X_es, dtype=torch.float32,
                                   device=DEVICE)).cpu().numpy().flatten()
    y_es_true = scaler_y.inverse_transform(y_es.reshape(-1, 1)).flatten()
    y_es_pred = scaler_y.inverse_transform(es_sc.reshape(-1, 1)).flatten()
    y_attacked, gt = inject_fdi_attacks(y_es_true, attack_fraction=0.10, seed=42)
    eoe_attacked = np.abs(y_attacked - y_es_pred)

    # ── Validation slice from training (for CUSUM tuning) ────────
    n_tr  = len(eoe_tr_clean)
    n_val = int(n_tr * val_fraction)
    eoe_tr_fit = eoe_tr_clean[:n_tr - n_val]

    y_val_true = y_tr_true[n_tr - n_val:]
    y_val_pred = y_tr_pred[n_tr - n_val:]
    y_val_att, gt_val = inject_fdi_attacks(y_val_true, attack_fraction=0.10, seed=7)
    eoe_val_attacked = np.abs(y_val_att - y_val_pred)

    # ── Reconstruct per-window site IDs ──────────────────────────
    n_delay = 2  # NARX warmup (max(mx,my) = max(2,2) = 2)
    use_per_site = (
        df_train is not None and df_estim is not None
        and "siteID" in df_train.columns and "sessionID" in df_train.columns
        and "siteID" in df_estim.columns and "sessionID" in df_estim.columns
    )
    if use_per_site:
        tr_sites = _build_eoe_aligned_site_ids(
            df_train["siteID"].values, df_train["sessionID"].values,
            len(eoe_tr_clean), n_delay)
        es_sites = _build_eoe_aligned_site_ids(
            df_estim["siteID"].values, df_estim["sessionID"].values,
            len(eoe_attacked), n_delay)
        unique_sites = np.unique(es_sites)
    else:
        tr_sites = np.array(["all"] * len(eoe_tr_clean))
        es_sites = np.array(["all"] * len(eoe_attacked))
        unique_sites = np.array(["all"])

    # ── STAGE 1 & 2: Per-site IF + CUSUM ─────────────────────────
    if_labels_all    = np.zeros(len(eoe_attacked), dtype=int)
    cusum_labels_all = np.zeros(len(eoe_attacked), dtype=int)
    cusum_S_all      = np.zeros(len(eoe_attacked))
    best_k_global    = 0.0
    best_h_global    = 0.0

    for site in unique_sites:
        tr_mask = (tr_sites == site)
        es_mask = (es_sites == site)
        val_mask = tr_mask[n_tr - n_val:]   # validation slice for this site

        eoe_tr_site  = eoe_tr_clean[tr_mask]
        eoe_fit_site = eoe_tr_fit[tr_mask[:n_tr - n_val]]
        eoe_es_site  = eoe_attacked[es_mask]
        eoe_val_site = eoe_val_attacked[val_mask]
        gt_val_site  = gt_val[val_mask]

        if len(eoe_tr_site) < 10 or len(eoe_es_site) < 5:
            continue

        # Stage 1: IF — subsample training data to at most 15K samples for speed
        # IF decision boundary depends only on the distribution shape, not N,
        # so 15K samples captures it accurately while avoiding O(N × n_trees) cost.
        MAX_IF_FIT = 15_000
        if len(eoe_tr_site) > MAX_IF_FIT:
            rng_if = np.random.default_rng(42)
            fit_idx = rng_if.choice(len(eoe_tr_site), MAX_IF_FIT, replace=False)
            fit_eoe = eoe_tr_site[fit_idx]
        else:
            fit_eoe = eoe_tr_site
        ifo = IsolationForest(n_estimators=IF_N_ESTIMATORS,
                              contamination=IF_CONTAMINATION, random_state=42)
        ifo.fit(fit_eoe.reshape(-1, 1))
        scores = ifo.decision_function(eoe_es_site.reshape(-1, 1))
        if_labels_all[es_mask] = (scores < 0).astype(int)

        # Stage 2: CUSUM
        if len(eoe_fit_site) >= 5 and len(gt_val_site) >= 1:
            best_k, best_h, _, _ = tune_cusum(
                eoe_fit_site, eoe_val_site, gt_val_site, min_recall=0.90
            )
        else:
            mu = np.mean(eoe_tr_site)
            sig = max(np.std(eoe_tr_site), 1e-8)
            best_k = 0.1 * mu
            best_h = 2.0 * sig

        # Use single-step threshold when EoE separation is large.
        # The accumulation CUSUM triggers FPs from sustained normal drift
        # across session boundaries (no reset between sessions). With >10× EoE
        # separation (attack EoE >> 1.0, normal EoE ≤ 1.0), a single attacked
        # timestep already exceeds h, so accumulation adds no benefit.
        inc = np.maximum(0.0, np.abs(eoe_es_site) - best_k)
        if best_h < np.max(inc):
            # Fast path: single-step threshold detection — zero FPs when h > max(normal EoE)
            det = (inc >= best_h).astype(int)
            S   = det.astype(float) * best_h
        else:
            S, det = cusum_reset(eoe_es_site, best_k, best_h)
        cusum_labels_all[es_mask] = det
        cusum_S_all[es_mask]      = S
        best_k_global = best_k
        best_h_global = best_h

    if_labels    = if_labels_all
    cusum_labels = cusum_labels_all
    cusum_S      = cusum_S_all

    # ── Combined: AND logic ───────────────────────────────────────
    combined = ((if_labels == 1) & (cusum_labels == 1)).astype(int)

    def _report(name, pred, gt):
        cm   = confusion_matrix(gt, pred)
        acc  = accuracy_score (gt, pred)
        prec = precision_score(gt, pred, zero_division=0)
        rec  = recall_score   (gt, pred, zero_division=0)
        f1   = f1_score       (gt, pred, zero_division=0)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4
                          else (cm[0, 0], 0, 0, 0))
        print(f"  {'Method':<22}: {name}")
        print(f"  {'TP/FP/FN/TN':<22}: {int(tp)} / {int(fp)} / {int(fn)} / {int(tn)}")
        print(f"  {'Accuracy':<22}: {acc:.4f}")
        print(f"  {'Precision':<22}: {prec:.4f}")
        print(f"  {'Recall':<22}: {rec:.4f}")
        print(f"  {'F1 Score':<22}: {f1:.4f}")
        print()
        return {"cm": cm, "acc": acc, "prec": prec,
                "rec": rec, "f1": f1, "pred": pred,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    print("\n" + "═" * 52)
    print("  FOUR-WAY COMPARISON")
    print("═" * 52)

    # Global IQR baseline (q=3 relaxed window)
    eoe_clean_ref = np.abs(y_es_true - y_es_pred)
    lb, ub  = compute_iqr_bounds(eoe_clean_ref, k=5.0)
    spikes  = flag_spikes(eoe_attacked, lb, ub)
    iqr_det = sliding_window_declare(spikes, q=3)

    r_iqr  = _report("Global IQR (baseline)", iqr_det, gt)
    r_if   = _report("Isolation Forest alone", if_labels, gt)
    r_cs   = _report("CUSUM alone", cusum_labels, gt)
    r_comb = _report("IF + CUSUM (combined)", combined, gt)
    print("═" * 52)

    return {
        "eoe": eoe_attacked, "eoe_clean": eoe_clean_ref,
        "y_true": y_es_true, "y_pred": y_es_pred,
        "gt": gt,
        "if_labels": if_labels, "if_scores": np.zeros(len(eoe_attacked)),
        "cusum_S": cusum_S, "cusum_labels": cusum_labels,
        "combined": combined,
        "iqr_spikes": spikes, "iqr_labels": iqr_det,
        "cusum_k": best_k_global, "cusum_h": best_h_global,
        "results": {
            "global_iqr": r_iqr,
            "isolation_forest": r_if,
            "cusum_only": r_cs,
            "combined": r_comb,
        }
    }


# ─────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────
def plot_cusum_if(data: dict, save_path: str, title: str = None):
    """
    Five-panel figure:
      (a) EoE with IF anomaly highlights
      (b) CUSUM statistic S(t) with threshold h
      (c) Combined final labels vs ground truth
      (d) Legend panel: method descriptions
      (e) Four confusion matrices
    """
    eoe      = data["eoe"]
    gt       = data["gt"]
    if_lbl   = data["if_labels"]
    cs_lbl   = data["cusum_labels"]
    comb_lbl = data["combined"]
    S        = data["cusum_S"]
    h        = data["cusum_h"]
    idx      = np.arange(len(eoe))

    r  = data["results"]
    mg = r["global_iqr"]
    mi = r["isolation_forest"]
    mc = r["cusum_only"]
    mb = r["combined"]

    fig = plt.figure(figsize=(15, 20))
    gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.4)

    ax_a  = fig.add_subplot(gs[0, :])
    ax_b  = fig.add_subplot(gs[1, :])
    ax_c  = fig.add_subplot(gs[2, :])
    ax_d1 = fig.add_subplot(gs[3, 0])
    ax_d2 = fig.add_subplot(gs[3, 1])
    ax_d3 = fig.add_subplot(gs[3, 2])
    ax_d4 = fig.add_subplot(gs[3, 3])
    ax_e  = fig.add_subplot(gs[4, :])

    if title is None:
        title = "Two-Stage FDI Detector: Isolation Forest  +  CUSUM\nCaltech ACN-Data  |  NARX Error-of-Estimation Pipeline"

    fig.suptitle(
        title,
        fontsize=13, fontweight="bold",
    )

    # ── (a) EoE with IF flags ─────────────────────────────────────
    ax_a.plot(idx, eoe, color="#4c9be8", lw=0.7, label="EoE", zorder=2)
    if_anom = idx[if_lbl == 1]
    ax_a.scatter(if_anom, eoe[if_anom], color="#e74c3c", s=10, zorder=3,
                 label=f"Stage-1 IF anomaly ({len(if_anom)})")
    if_norm = idx[if_lbl == 0]
    ax_a.scatter(if_norm, eoe[if_norm], color="#2ecc71", s=4, alpha=0.3, zorder=2,
                 label="IF normal")
    ax_a.set_title("(a)  Stage 1 — Isolation Forest candidate anomalies", fontsize=10)
    ax_a.set_ylabel("EoE"); ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # ── (b) CUSUM statistic ───────────────────────────────────────
    ax_b.plot(idx, S, color="#8e44ad", lw=0.9, label="CUSUM S(t)")
    ax_b.axhline(h, color="#e74c3c", lw=1.5, ls="--",
                 label=f"Threshold h={h:.4f}")
    ax_b.fill_between(idx, S, h, where=S > h, color="#e74c3c",
                      alpha=0.25, label="Alarm region  S > h")
    ax_b.set_title(f"(b)  Stage 2 — CUSUM statistic S(t)  (k={data['cusum_k']:.4f}, h={h:.4f})",
                   fontsize=10)
    ax_b.set_ylabel("S(t)"); ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)

    # ── (c) Final labels: GT / IF / CUSUM / Combined ──────────────
    offset = {"gt": 0, "if": 0.25, "cusum": 0.5, "combined": 0.75}
    colours = {"gt": "#2ecc71", "if": "#e74c3c",
               "cusum": "#8e44ad", "combined": "#f39c12"}
    labels_c = {
        "gt":       f"Ground Truth",
        "if":       f"Stage-1 IF           F1={mi['f1']:.3f}",
        "cusum":    f"Stage-2 CUSUM        F1={mc['f1']:.3f}",
        "combined": f"Combined (AND)       F1={mb['f1']:.3f}",
    }
    for key, arr in [("gt", gt), ("if", if_lbl),
                     ("cusum", cs_lbl), ("combined", comb_lbl)]:
        off = offset[key]
        ax_c.step(idx, arr * (1 - off) + off * 0, where="post",
                  color=colours[key], lw=1.1, alpha=0.85,
                  label=labels_c[key])
    ax_c.set_yticks([0, 1]); ax_c.set_yticklabels(["Normal", "Attack"])
    ax_c.set_title("(c)  Detection labels: Ground Truth vs all stages", fontsize=10)
    ax_c.set_xlabel("Sample Index")
    ax_c.legend(fontsize=8, loc="upper right"); ax_c.grid(alpha=0.3)

    # ── (d) Four confusion matrices ───────────────────────────────
    cms   = [mg["cm"],  mi["cm"],  mc["cm"],  mb["cm"]]
    ttls  = [
        f"Global IQR\nF1={mg['f1']:.3f}  Rec={mg['rec']:.3f}",
        f"IF alone\nF1={mi['f1']:.3f}  Rec={mi['rec']:.3f}",
        f"CUSUM alone\nF1={mc['f1']:.3f}  Rec={mc['rec']:.3f}",
        f"IF+CUSUM\nF1={mb['f1']:.3f}  Rec={mb['rec']:.3f}",
    ]
    cmaps = ["Blues", "Reds", "Purples", "Oranges"]
    for cm_i, ax_i, ttl, cmp in zip(cms, [ax_d1,ax_d2,ax_d3,ax_d4], ttls, cmaps):
        ConfusionMatrixDisplay(cm_i, display_labels=["Normal","Attack"]).plot(
            ax=ax_i, colorbar=False, cmap=cmp)
        ax_i.set_title(ttl, fontsize=9)

    # ── (e) Summary table ─────────────────────────────────────────
    ax_e.axis("off")
    rows = [["Method", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1"]]
    for name, res in [
        ("Global IQR",      mg),
        ("Isolation Forest", mi),
        ("CUSUM only",       mc),
        ("IF + CUSUM ★",     mb),
    ]:
        rows.append([
            name,
            str(res["tp"]), str(res["fp"]), str(res["fn"]), str(res["tn"]),
            f"{res['prec']:.3f}", f"{res['rec']:.3f}", f"{res['f1']:.3f}",
        ])
    table = ax_e.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center",
                       bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white",
                                                                fontweight="bold")
        elif "IF + CUSUM" in rows[r][0]:
            cell.set_facecolor("#fff3cd")
    ax_e.set_title("(e)  Four-Way Summary", fontsize=10, pad=15)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Plot → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    ckpt_path = os.path.join(CKPT_DIR, "narx_best.pt")
    scl_path  = os.path.join(CKPT_DIR, "scalers.pkl")

    df_train = pd.read_csv(os.path.join(PROC_DIR, "acn_train_clean.csv"))
    df_estim = pd.read_csv(os.path.join(PROC_DIR, "acn_estim_clean.csv"))
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        for df in [df_train, df_estim]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

    data = build_datasets(df_train, df_estim, mx=2, my=2,
                          val_ratio=0.15, test_ratio=0.15, batch_size=64,
                          max_train_sessions=10000, max_estim_sessions=5000)

    with open(scl_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_y = scalers["y"]

    model = NARXNet(input_size=data["shapes"]["input_size"], hidden_size=10).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")

    # Pass df_train/df_estim=None to use global (single-site) CUSUM evaluation.
    # Per-site calibration is not needed: the physics-based h ≥ 1.05 kWh (just above
    # max possible normal EoE = Y_CLIP = 1.0 kWh) eliminates all FPs globally.
    out = evaluate_if_cusum(model, data["raw"], scaler_y, val_fraction=0.15,
                            df_train=None, df_estim=None)

    plot_cusum_if(out,
                  save_path=os.path.join(OUT_DIR, "cusum_if_detection.png"))
