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

IF_CONTAMINATION = 0.01
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
    CUSUM with alarm-triggered reset (S → 0 after each declared alarm).
    Prevents a single large burst from permanently keeping S elevated.
    """
    n = len(eoe)
    S = np.zeros(n)
    detected = np.zeros(n, dtype=int)
    for t in range(1, n):
        S[t] = max(0.0, S[t-1] + abs(eoe[t]) - k)
        if S[t] > h:
            detected[t] = 1
            S[t] = 0.0   # reset on alarm
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
) -> tuple[float, float, float, float]:
    """
    Grid search (k, h) pairs.
    k = k_scale × mean(normal EoE)
    h = h_scale × std(normal EoE)

    Returns: k_best, h_best, best_f1
    """
    if k_scales is None:
        k_scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if h_scales is None:
        h_scales = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

    mu_normal  = np.mean(eoe_clean_train)
    sig_normal = np.std(eoe_clean_train)

    print("\n[TUNING] CUSUM grid search on validation slice:")
    print(f"  Normal EoE μ={mu_normal:.5f}  σ={sig_normal:.5f}")
    print(f"  {'k':>8}  {'h':>8}  {'F1':>8}  {'Recall':>8}  {'Prec':>8}")

    best_k, best_h, best_f1 = mu_normal * k_scales[0], sig_normal * h_scales[0], -1.0

    for ks, hs in itertools.product(k_scales, h_scales):
        k, h = ks * mu_normal, hs * sig_normal
        _, det = cusum_reset(eoe_val_attacked, k, h)
        f1   = f1_score       (gt_val, det, zero_division=0)
        rec  = recall_score   (gt_val, det, zero_division=0)
        prec = precision_score(gt_val, det, zero_division=0)

        # Require recall > min_recall to prefer high-sensitivity configurations
        if f1 > best_f1 and rec >= min_recall:
            best_f1, best_k, best_h = f1, k, h
            print(f"  {k:>8.5f}  {h:>8.5f}  {f1:>8.4f}  {rec:>8.4f}  {prec:>8.4f}  ← best so far")

    if best_f1 < 0:
        # Relax recall constraint if nothing found
        print("  [WARN] No (k,h) met recall >= 0.90, relaxing constraint...")
        for ks, hs in itertools.product(k_scales, h_scales):
            k, h = ks * mu_normal, hs * sig_normal
            _, det = cusum_reset(eoe_val_attacked, k, h)
            f1 = f1_score(gt_val, det, zero_division=0)
            if f1 > best_f1:
                best_f1, best_k, best_h = f1, k, h

    print(f"  → Best k={best_k:.5f}  h={best_h:.5f}  (val F1={best_f1:.4f})\n")
    return best_k, best_h, best_f1, mu_normal


# ─────────────────────────────────────────────────────────────────
# 3. Two-stage IF + CUSUM evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate_if_cusum(model, raw_data, scaler_y, val_fraction=0.15):
    """
    Full two-stage pipeline:
      Stage 1: IsolationForest (recall-maximising)
      Stage 2: CUSUM-reset     (precision-preserving)
      Combined: attack iff BOTH flag the point
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

    # ── STAGE 1: Isolation Forest ─────────────────────────────────
    ifo = IsolationForest(n_estimators=IF_N_ESTIMATORS,
                          contamination=IF_CONTAMINATION, random_state=42)
    ifo.fit(eoe_tr_clean.reshape(-1, 1))
    if_scores  = ifo.decision_function(eoe_attacked.reshape(-1, 1))
    if_labels  = (if_scores < 0).astype(int)

    # ── STAGE 2: CUSUM tuning then detection ──────────────────────
    best_k, best_h, _, mu_normal = tune_cusum(
        eoe_tr_fit, eoe_val_attacked, gt_val, min_recall=0.90
    )
    cusum_S, cusum_labels = cusum_reset(eoe_attacked, best_k, best_h)

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

    # Global IQR baseline
    eoe_clean_ref = np.abs(y_es_true - y_es_pred)
    lb, ub  = compute_iqr_bounds(eoe_clean_ref, k=5.0)
    spikes  = flag_spikes(eoe_attacked, lb, ub)
    iqr_det = sliding_window_declare(spikes, q=5)

    r_iqr  = _report("Global IQR (baseline)", iqr_det, gt)
    r_if   = _report("Isolation Forest alone", if_labels, gt)
    r_cs   = _report("CUSUM alone", cusum_labels, gt)
    r_comb = _report("IF + CUSUM (combined)", combined, gt)
    print("═" * 52)

    return {
        "eoe": eoe_attacked, "eoe_clean": eoe_clean_ref,
        "y_true": y_es_true, "y_pred": y_es_pred,
        "gt": gt,
        "if_labels": if_labels, "if_scores": if_scores,
        "cusum_S": cusum_S, "cusum_labels": cusum_labels,
        "combined": combined,
        "iqr_spikes": spikes, "iqr_labels": iqr_det,
        "cusum_k": best_k, "cusum_h": best_h,
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
def plot_cusum_if(data: dict, save_path: str):
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

    fig.suptitle(
        "Two-Stage FDI Detector: Isolation Forest  +  CUSUM\n"
        "Caltech ACN-Data  |  NARX Error-of-Estimation Pipeline",
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
                          val_ratio=0.15, test_ratio=0.15, batch_size=64)

    with open(scl_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_y = scalers["y"]

    model = NARXNet(input_size=data["shapes"]["input_size"], hidden_size=10).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")

    out = evaluate_if_cusum(model, data["raw"], scaler_y, val_fraction=0.15)

    plot_cusum_if(out,
                  save_path=os.path.join(OUT_DIR, "cusum_if_detection.png"))
