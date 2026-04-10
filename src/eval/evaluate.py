"""
IQR-Based FDI Detection — Global (paper baseline) + Session-Aware (improved)
─────────────────────────────────────────────────────────────────
Global IQR (paper baseline):
  • Compute Q1/Q3/IQR across ALL estimation EoE values
  • LB = Q1 - 5·IQR,  UB = Q3 + 5·IQR
  • Flag spikes outside bounds, then sliding window q=5

Session-Aware IQR (this module's improvement):
  • Compute Q1/Q3/IQR independently for each charging session
  • LB_s / UB_s vary per session, eliminating transition spikes
  • Same sliding window q=5 applied within each session

Usage (from narx_ev_fdi/ project root):
    python3 -m src.eval.evaluate
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
)

# ── project root resolution ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset  import build_datasets, TARGET_COL
from src.models.narx   import NARXNet

CKPT_DIR   = os.path.join(ROOT, "checkpoints")
OUT_DIR    = os.path.join(ROOT, "results")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. IQR detection primitives
# ─────────────────────────────────────────────────────────────────
def compute_iqr_bounds(eoe: np.ndarray, k: float = 5.0) -> tuple[float, float]:
    """Return (LB, UB) using the paper's k=5 IQR rule."""
    q1, q3 = np.percentile(eoe, 25), np.percentile(eoe, 75)
    iqr    = q3 - q1
    lb     = q1 - k * iqr
    ub     = q3 + k * iqr
    return lb, ub


def flag_spikes(eoe: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Return boolean array: True where EoE is outside [LB, UB]."""
    return (eoe < lb) | (eoe > ub)


def sliding_window_declare(spikes: np.ndarray, q: int = 3) -> np.ndarray:
    """
    Sliding window of length q.
    Declare FDI=1 at position t if spikes[t-q+1 .. t] are ALL True.
    q=3 (relaxed from 5) reduces false negatives on short bursts.
    Returns integer label array of same length as spikes.
    """
    n       = len(spikes)
    labels  = np.zeros(n, dtype=int)
    for t in range(q - 1, n):
        if spikes[t - q + 1 : t + 1].all():
            labels[t] = 1
    return labels


# ─────────────────────────────────────────────────────────────────
# 1b. Session-aware IQR detection
# ─────────────────────────────────────────────────────────────────
def session_aware_iqr(
    eoe:         np.ndarray,
    session_ids: np.ndarray,
    k:           float = 5.0,
    q:           int   = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-session IQR bounds and apply the sliding-window detector.

    Advantage over global IQR
    ─────────────────────────
    At every session boundary the NARX model's hidden state is cold-started,
    which causes a transient spike in EoE even for legitimate data.  Global
    IQR absorbs these spikes into its per-sample variance — or worse flags
    them as attacks.  Per-session IQR fits its own Q1/Q3 to the local
    error distribution, so boundary artefacts cannot pollute the thresholds.

    Parameters
    ----------
    eoe         : (N,) array of |y_attacked - y_pred|
    session_ids : (N,) array of session identifier per sample
    k           : IQR multiplier (5.0 per paper)
    q           : sliding window length (5 per paper)

    Returns
    -------
    spikes      : bool (N,) — True where sample is a spike within its session
    detected    : int  (N,) — 1 where FDI declared by sliding window
    lb_per_samp : float (N,) — the lower bound assigned to each sample's session
    ub_per_samp : float (N,) — the upper bound assigned to each sample's session
    """
    n           = len(eoe)
    spikes      = np.zeros(n, dtype=bool)
    detected    = np.zeros(n, dtype=int)
    lb_per_samp = np.zeros(n, dtype=float)
    ub_per_samp = np.zeros(n, dtype=float)

    for s_id in np.unique(session_ids):
        mask = (session_ids == s_id)
        idx  = np.where(mask)[0]          # absolute positions in global array
        eoe_s = eoe[mask]

        if len(eoe_s) < 4:
            # Too few points — use global median as a fallback (no spike possible)
            lb_per_samp[mask] = -np.inf
            ub_per_samp[mask] =  np.inf
            continue

        lb_s, ub_s        = compute_iqr_bounds(eoe_s, k=k)
        spk_s             = flag_spikes(eoe_s, lb_s, ub_s)
        det_s             = sliding_window_declare(spk_s, q=q)

        spikes[mask]      = spk_s
        detected[mask]    = det_s
        lb_per_samp[mask] = lb_s
        ub_per_samp[mask] = ub_s

    return spikes, detected, lb_per_samp, ub_per_samp


# ─────────────────────────────────────────────────────────────────
# 2. Synthetic attack injection (for ground-truth labels)
# ─────────────────────────────────────────────────────────────────
def inject_fdi_attacks(
    y_true: np.ndarray,
    attack_fraction: float = 0.10,
    burst_len_range: tuple = (5, 15),
    scale_range: tuple    = (1.0, 2.0),
    seed: int             = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly inject bursts of additive FDI attacks into y_true.

    CRITICAL FIX: Uses additive injection instead of multiplicative.
    The old `y_att = y_true * scale` was invisible for zero-target timesteps
    (73.5% of the multi-site dataset), mathematically capping recall at ~26.5%.

    Additive injection `y_att = y_true + scale * baseline` works on ALL
    timesteps (idle and charging) and creates a clear, detectable anomaly.

    baseline = mean of all positive (active-charging) target values, so
    the injected magnitude is proportional to a real charging signal.

    Returns
    -------
    y_attacked : modified target array
    gt_labels  : 0/1 ground-truth per sample (1 = under attack)
    """
    rng = np.random.default_rng(seed)
    y_att = y_true.copy()
    gt    = np.zeros(len(y_true), dtype=int)

    # Compute positive-charging baseline (mean of non-zero targets)
    pos_vals = y_true[y_true > 1e-6]
    baseline = float(np.mean(pos_vals)) if len(pos_vals) > 0 else 0.05

    n_attacks = max(1, int(len(y_true) * attack_fraction / np.mean(burst_len_range)))
    for _ in range(n_attacks):
        start = rng.integers(0, max(1, len(y_true) - burst_len_range[1]))
        blen  = rng.integers(*burst_len_range)
        scale = rng.uniform(*scale_range)
        end   = min(start + blen, len(y_true))
        # Additive injection: clearly visible regardless of base value
        y_att[start:end] = y_att[start:end] + scale * baseline
        gt[start:end]    = 1
    return y_att, gt


# ─────────────────────────────────────────────────────────────────
# 3. Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────
def evaluate(
    model: NARXNet,
    raw_data: dict,
    scaler_y,
    q: int   = 5,
    k: float = 5.0,
):
    """
    Run IQR detection on the estimation set and report all metrics.

    Parameters
    ----------
    model     : trained NARXNet
    raw_data  : dict returned by build_datasets (key 'raw')
    scaler_y  : fitted MinMaxScaler for target
    q         : sliding window length (paper: 5)
    k         : IQR multiplier (paper: 5)
    """
    model.eval()
    X_es = raw_data["X_estim_w"]    # already scaled
    y_es = raw_data["y_estim_w"]    # already scaled

    with torch.no_grad():
        preds_sc = model(
            torch.tensor(X_es, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()

    y_true = scaler_y.inverse_transform(y_es.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()

    y_attacked, gt_labels = inject_fdi_attacks(y_true, attack_fraction=0.10, seed=42)
    eoe          = np.abs(y_attacked - y_pred)
    eoe_clean    = np.abs(y_true    - y_pred)
    lb, ub       = compute_iqr_bounds(eoe_clean, k=k)
    spikes       = flag_spikes(eoe, lb, ub)
    detected     = sliding_window_declare(spikes, q=q)

    cm   = confusion_matrix(gt_labels, detected)
    acc  = accuracy_score (gt_labels, detected)
    prec = precision_score(gt_labels, detected, zero_division=0)
    rec  = recall_score   (gt_labels, detected, zero_division=0)
    f1   = f1_score       (gt_labels, detected, zero_division=0)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    _print_metrics("GLOBAL IQR", lb, ub, len(gt_labels),
                   int(gt_labels.sum()), int(detected.sum()),
                   int(tp), int(fp), int(fn), int(tn), acc, prec, rec, f1)

    return {
        "eoe": eoe, "eoe_clean": eoe_clean,
        "y_true": y_true, "y_pred": y_pred, "y_attacked": y_attacked,
        "gt_labels": gt_labels, "detected": detected, "spikes": spikes,
        "lb": lb, "ub": ub,
        "lb_per_samp": np.full_like(eoe, lb),
        "ub_per_samp": np.full_like(eoe, ub),
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "confusion_matrix": cm,
    }


# helper shared by both modes
def _print_metrics(label, lb, ub, total, n_true, n_det, tp, fp, fn, tn,
                   acc, prec, rec, f1):
    print("\n" + "═"*56)
    print(f"  {label}")
    print("═"*56)
    print(f"  IQR Bounds (sample)     : LB≈{lb:.4e}  UB≈{ub:.4e}")
    print(f"  Total Samples           : {total}")
    print(f"  True Attacks  (gt=1)    : {n_true}")
    print(f"  Detected      (pred=1)  : {n_det}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy  : {acc:.4f}   Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}   F1 Score  : {f1:.4f}")
    print("═"*56)


# ─────────────────────────────────────────────────────────────────
# 4b. Session-aware evaluation wrapper
# ─────────────────────────────────────────────────────────────────
def evaluate_session_aware(
    model:       NARXNet,
    raw_data:    dict,
    scaler_y,
    df_estim:    pd.DataFrame,
    q:           int   = 5,
    k:           float = 5.0,
):
    """
    Session-aware IQR evaluation.
    Uses df_estim['sessionID'] to recover per-session boundaries on the
    NARX window array (which has lost the first max(mx,my) rows per session).
    """
    model.eval()
    X_es = raw_data["X_estim_w"]
    y_es = raw_data["y_estim_w"]

    with torch.no_grad():
        preds_sc = model(
            torch.tensor(X_es, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()

    y_true = scaler_y.inverse_transform(y_es.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    y_attacked, gt_labels = inject_fdi_attacks(y_true, attack_fraction=0.10, seed=42)
    eoe = np.abs(y_attacked - y_pred)

    # ── Reconstruct session_ids aligned with the NARX window array ──
    # dataset.py builds windows with max(mx,my)=2 warmup per session.
    # For each session we drop the first 2 rows, so the window count
    # per session = numTimeStamps - 2.  We replicate sessionID accordingly.
    MX, MY = 2, 2
    n_delay = max(MX, MY)

    # Count per session in df_estim (after explode, each group = one session)
    if "sessionID" in df_estim.columns:
        sid_col = "sessionID"
    else:
        sid_col = "_id"

    session_ids_full = df_estim[sid_col].values           # one per exploded row
    unique_sids, counts = np.unique(session_ids_full, return_counts=True)

    sid_window = []
    for sid, cnt in zip(unique_sids, counts):
        n_windows = max(0, cnt - n_delay)
        sid_window.extend([sid] * n_windows)
    sid_window = np.array(sid_window)

    # Trim/pad to match actual window array length (guard against off-by-one)
    n_win = len(eoe)
    if len(sid_window) > n_win:
        sid_window = sid_window[:n_win]
    elif len(sid_window) < n_win:
        sid_window = np.pad(sid_window, (0, n_win - len(sid_window)),
                            mode="edge")

    spikes, detected, lb_ps, ub_ps = session_aware_iqr(eoe, sid_window, k=k, q=q)

    cm   = confusion_matrix(gt_labels, detected)
    acc  = accuracy_score (gt_labels, detected)
    prec = precision_score(gt_labels, detected, zero_division=0)
    rec  = recall_score   (gt_labels, detected, zero_division=0)
    f1   = f1_score       (gt_labels, detected, zero_division=0)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    # Representative single LB/UB for print (first session's bounds)
    _print_metrics("SESSION-AWARE IQR",
                   lb_ps[0], ub_ps[0],
                   len(gt_labels), int(gt_labels.sum()), int(detected.sum()),
                   int(tp), int(fp), int(fn), int(tn), acc, prec, rec, f1)

    return {
        "eoe": eoe, "y_true": y_true, "y_pred": y_pred,
        "y_attacked": y_attacked, "gt_labels": gt_labels,
        "spikes": spikes, "detected": detected,
        "lb_per_samp": lb_ps, "ub_per_samp": ub_ps,
        "session_ids": sid_window,
        "metrics": {"accuracy": acc, "precision": prec,
                    "recall": rec,   "f1": f1},
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────
# 5. Plotting (comparison: global vs session-aware)
# ─────────────────────────────────────────────────────────────────
def plot_comparison(glob: dict, sess: dict, save_path: str):
    """
    Four-panel figure:
      (a) EoE with global IQR bounds
      (b) EoE with session-aware IQR bounds (per-sample LB/UB)
      (c) GT vs detected labels — both methods
      (d) Side-by-side confusion matrices
    """
    eoe = glob["eoe"]
    gt  = glob["gt_labels"]
    idx = np.arange(len(eoe))

    fig, axes = plt.subplots(4, 1, figsize=(15, 16),
                             gridspec_kw={"height_ratios": [2.5, 2.5, 1.5, 2]})
    fig.suptitle(
        "Global vs. Session-Aware IQR FDI Detection — Caltech ACN (Dec 2020 – Jan 2021)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── (a) Global IQR ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(idx, eoe, color="#4c9be8", lw=0.75, label="EoE", zorder=2)
    ax.axhline(glob["ub"], color="#e84c4c", lw=1.5, ls="--",
               label=f"Global UB = {glob['ub']:.4f}")
    ax.axhline(glob["lb"], color="#e8a84c", lw=1.5, ls="--",
               label=f"Global LB = {glob['lb']:.4f}")
    ax.fill_between(idx, glob["lb"], glob["ub"], color="#4c9be8", alpha=0.07)
    spk_g = np.where(glob["spikes"])[0]
    if len(spk_g):
        ax.scatter(spk_g, eoe[spk_g], color="#e84c4c", s=10, zorder=3,
                   label=f"Spikes ({len(spk_g)}) incl. boundary artefacts")
    ax.set_title("(a) Global IQR — single LB/UB for all sessions", fontsize=10)
    ax.set_ylabel("EoE", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (b) Session-aware IQR ────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(idx, eoe, color="#4c9be8", lw=0.75, label="EoE", zorder=2)
    ax2.fill_between(idx, sess["lb_per_samp"], sess["ub_per_samp"],
                     color="#2ecc71", alpha=0.12, label="Per-session normal band")
    ax2.plot(idx, sess["ub_per_samp"], color="#27ae60", lw=0.8, ls="--",
             label="Per-session UB")
    ax2.plot(idx, sess["lb_per_samp"], color="#f39c12", lw=0.8, ls="--",
             label="Per-session LB")
    spk_s = np.where(sess["spikes"])[0]
    if len(spk_s):
        ax2.scatter(spk_s, eoe[spk_s], color="#e84c4c", s=10, zorder=3,
                    label=f"Spikes ({len(spk_s)}) — boundary artefacts removed")
    ax2.set_title("(b) Session-Aware IQR — LB/UB recomputed per charging session",
                  fontsize=10)
    ax2.set_ylabel("EoE", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (c) GT vs both detectors ──────────────────────────────────
    ax3 = axes[2]
    ax3.step(idx, gt,              where="post", color="#2ecc71", lw=1.2,
             alpha=0.9, label="Ground Truth")
    ax3.step(idx, glob["detected"], where="post", color="#e74c3c", lw=1.0,
             ls="--", alpha=0.9, label=f"Global IQR  (F1={glob['metrics']['f1']:.3f})")
    ax3.step(idx, sess["detected"], where="post", color="#9b59b6", lw=1.0,
             ls=":",  alpha=0.9, label=f"Session-Aware (F1={sess['metrics']['f1']:.3f})")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Normal", "Attack"])
    ax3.set_xlabel("Sample Index", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.set_title("(c) Detection Labels — Ground Truth vs. Both Methods", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── (d) Side-by-side confusion matrices ───────────────────────
    ax4a = fig.add_subplot(4, 2, 7)
    ax4b = fig.add_subplot(4, 2, 8)
    axes[3].set_visible(False)       # hide the placeholder

    mg, ms = glob["metrics"], sess["metrics"]
    for cm_i, ax_i, ttl, m in [
        (glob["confusion_matrix"], ax4a,
         f"Global IQR\nAcc={mg['accuracy']:.3f}  Prec={mg['precision']:.3f}  Rec={mg['recall']:.3f}  F1={mg['f1']:.3f}", mg),
        (sess["confusion_matrix"], ax4b,
         f"Session-Aware\nAcc={ms['accuracy']:.3f}  Prec={ms['precision']:.3f}  Rec={ms['recall']:.3f}  F1={ms['f1']:.3f}", ms),
    ]:
        ConfusionMatrixDisplay(cm_i, display_labels=["Normal", "Attack"]).plot(
            ax=ax_i, colorbar=False,
            cmap="Blues" if ax_i is ax4a else "Purples",
        )
        ax_i.set_title(ttl, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Comparison plot → {save_path}")


def print_comparison_table(glob: dict, sess: dict):
    """Print a concise side-by-side comparison table."""
    mg, ms = glob["metrics"], sess["metrics"]
    g_cm = glob["confusion_matrix"].ravel()
    s_cm = sess["confusion_matrix"].ravel()
    gtn, gfp, gfn, gtp = (g_cm if len(g_cm)==4 else [g_cm[0],0,0,0])
    stn, sfp, sfn, stp = (s_cm if len(s_cm)==4 else [s_cm[0],0,0,0])

    print("\n" + "═"*64)
    print(f"{'Metric':<20} {'Global IQR':>18} {'Session-Aware IQR':>22}")
    print("─"*64)
    print(f"{'TP':<20} {int(gtp):>18} {int(stp):>22}")
    print(f"{'FP':<20} {int(gfp):>18} {int(sfp):>22}")
    print(f"{'FN':<20} {int(gfn):>18} {int(sfn):>22}")
    print(f"{'TN':<20} {int(gtn):>18} {int(stn):>22}")
    print("─"*64)
    for key in ["accuracy", "precision", "recall", "f1"]:
        delta = ms[key] - mg[key]
        sign  = "+" if delta >= 0 else ""
        print(f"{key.capitalize():<20} {mg[key]:>18.4f} {ms[key]:>22.4f}  ({sign}{delta:.4f})")
    print("═"*64)
    print("  (+) = Session-Aware is better")
    print()


# ─────────────────────────────────────────────────────────────────
# 6. Entry point — runs BOTH modes and compares
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")
    ckpt_path = os.path.join(CKPT_DIR, "narx_best.pt")
    scl_path  = os.path.join(CKPT_DIR, "scalers.pkl")

    df_train = pd.read_csv(train_csv)
    df_estim = pd.read_csv(estim_csv)
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

    # Run both detection modes
    global_results  = evaluate(model, data["raw"], scaler_y, q=5, k=5.0)
    session_results = evaluate_session_aware(model, data["raw"], scaler_y,
                                             df_estim, q=5, k=5.0)

    # Side-by-side comparison table
    print_comparison_table(global_results, session_results)

    # Comparison plot
    plot_path = os.path.join(OUT_DIR, "eoe_iqr_comparison.png")
    plot_comparison(global_results, session_results, save_path=plot_path)
