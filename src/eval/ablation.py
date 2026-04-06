"""
Ablation Study: FDI Detection vs Attack Intensity ϑ
─────────────────────────────────────────────────────────────────
The original paper (Eq. 6) models FDI attacks as manipulation of
minutesAvailable by ϑ minutes. A larger ϑ means a larger mismatch
between the attacked signal and what the NARX model expects, resulting
in larger EoE spikes that are easier to detect.

This script sweeps ϑ ∈ [1, 5, 10, 20, 30, 60] minutes and records{
  - F1 score
  - Recall
  - Precision
} for all three detectors:
  1. Global IQR (paper baseline)
  2. Session-Aware IQR (our improvement)
  3. Isolation Forest (contamination pre-tuned at ϑ=10)

Key insight visualised:
  - At small ϑ (subtle attacks), all detectors degrade in recall
  - The method with highest AUC over the ϑ sweep is the most robust
  - This exposes the paper's admitted weakness at low-intensity attacks

Physical mapping of ϑ → scale factor
─────────────────────────────────────
minutesAvailable   = total minutes the EV is plugged in
FDI attack         = attacker reports minutesAvailable ← minutesAvailable + ϑ
Charging algorithm = delivers more kWh to "use" extra available time
Effect on EoE      = charged kWh exceeds model expectation by ≈ (ϑ / session_minutes)

We approximate this as a multiplicative scale on the attacked samples:
  scale(ϑ) = 1 + (ϑ / REFERENCE_SESSION_MINUTES)
where REFERENCE_SESSION_MINUTES ≈ 60 (median session in Caltech dataset).

Usage (from project root):
    python3 -m src.eval.ablation
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score
)
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset import build_datasets
from src.models.narx  import NARXNet
from src.eval.evaluate import (
    compute_iqr_bounds, flag_spikes, sliding_window_declare,
    session_aware_iqr,
)

CKPT_DIR = os.path.join(ROOT, "checkpoints")
OUT_DIR  = os.path.join(ROOT, "results")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Ablation config ───────────────────────────────────────────────
THETA_VALUES            = [1, 5, 10, 20, 30, 60]   # minutes
REFERENCE_SESSION_MIN   = 60.0                       # median session length
ATTACK_FRACTION         = 0.10                       # 10 % of samples attacked
BURST_LEN               = (5, 15)
SEED                    = 42
IQR_K                   = 5.0
IQR_Q                   = 5
IF_CONTAMINATION        = 0.01    # pre-tuned value from isolation_forest.py
IF_N_ESTIMATORS         = 200


# ─────────────────────────────────────────────────────────────────
# 1.  FDI injection parameterised by ϑ
# ─────────────────────────────────────────────────────────────────
def inject_fdi_theta(
    y_true: np.ndarray,
    theta:  float,
    attack_fraction: float = ATTACK_FRACTION,
    burst_len_range: tuple = BURST_LEN,
    seed:  int             = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject FDI attacks scaled by ϑ (minutesAvailable manipulation).

    scale(ϑ) = 1 + ϑ / REFERENCE_SESSION_MIN

    A charger deceived about having ϑ more minutes will try to deliver
    proportionally more energy, pushing kWhDeliveredPerTimeStamp higher.
    """
    rng   = np.random.default_rng(seed)
    y_att = y_true.copy()
    gt    = np.zeros(len(y_true), dtype=int)

    scale       = 1.0 + theta / REFERENCE_SESSION_MIN
    n_attacks   = max(1, int(len(y_true) * attack_fraction / np.mean(burst_len_range)))

    for _ in range(n_attacks):
        start = rng.integers(0, max(1, len(y_true) - burst_len_range[1]))
        blen  = int(rng.integers(*burst_len_range))
        end   = min(start + blen, len(y_true))
        y_att[start:end] = y_att[start:end] * scale
        gt[start:end]    = 1

    return y_att, gt


# ─────────────────────────────────────────────────────────────────
# 2.  Per-ϑ detection with all three methods
# ─────────────────────────────────────────────────────────────────
def _metrics_from_labels(gt, pred):
    f1   = f1_score       (gt, pred, zero_division=0)
    rec  = recall_score   (gt, pred, zero_division=0)
    prec = precision_score(gt, pred, zero_division=0)
    return f1, rec, prec


def run_ablation(
    model,
    raw_data:  dict,
    scaler_y,
    df_estim:  pd.DataFrame,
    ifo_fitted, # pre-trained IsolationForest on clean training EoE
) -> dict:
    """
    For each ϑ, inject attacks, score with all three detectors, record metrics.
    Returns dict:  {theta: {method: {f1, recall, precision}}}
    """
    MX, MY    = 2, 2
    n_delay   = max(MX, MY)
    model.eval()

    # ── Forward pass on estimation set (clean predictions) ──────
    X_es = raw_data["X_estim_w"]
    y_es = raw_data["y_estim_w"]
    with torch.no_grad():
        preds_sc = model(
            torch.tensor(X_es, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()
    y_true = scaler_y.inverse_transform(y_es.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    eoe_clean = np.abs(y_true - y_pred)

    # ── Session IDs aligned to window array ─────────────────────
    sid_col = "sessionID" if "sessionID" in df_estim.columns else "_id"
    unique_sids, counts = np.unique(df_estim[sid_col].values, return_counts=True)
    sid_window = []
    for sid, cnt in zip(unique_sids, counts):
        sid_window.extend([sid] * max(0, cnt - n_delay))
    sid_window = np.array(sid_window)
    n_win = len(eoe_clean)
    if len(sid_window) > n_win:    sid_window = sid_window[:n_win]
    elif len(sid_window) < n_win:  sid_window = np.pad(sid_window,
                                                        (0, n_win - len(sid_window)),
                                                        mode="edge")

    # ── Global IQR bounds fitted on CLEAN EoE ───────────────────
    lb_g, ub_g = compute_iqr_bounds(eoe_clean, k=IQR_K)

    results = {}

    for theta in THETA_VALUES:
        y_attacked, gt = inject_fdi_theta(y_true, theta=theta)
        eoe_att = np.abs(y_attacked - y_pred)

        # 1. Global IQR
        spikes_g  = flag_spikes(eoe_att, lb_g, ub_g)
        det_g     = sliding_window_declare(spikes_g, q=IQR_Q)
        f1_g, rec_g, prec_g = _metrics_from_labels(gt, det_g)

        # 2. Session-Aware IQR (fits bounds independently per session)
        _, det_s, _, _ = session_aware_iqr(eoe_att, sid_window,
                                           k=IQR_K, q=IQR_Q)
        f1_s, rec_s, prec_s = _metrics_from_labels(gt, det_s)

        # 3. Isolation Forest (predict on attacked EoE)
        scores_if = ifo_fitted.decision_function(eoe_att.reshape(-1, 1))
        det_if    = (scores_if < 0).astype(int)
        f1_i, rec_i, prec_i = _metrics_from_labels(gt, det_if)

        results[theta] = {
            "global_iqr":    {"f1": f1_g, "recall": rec_g, "precision": prec_g},
            "session_iqr":   {"f1": f1_s, "recall": rec_s, "precision": prec_s},
            "isolation_forest": {"f1": f1_i, "recall": rec_i, "precision": prec_i},
        }

        print(f"  ϑ={theta:>3}min  |  "
              f"Global IQR  F1={f1_g:.3f} Rec={rec_g:.3f}  |  "
              f"Session IQR F1={f1_s:.3f} Rec={rec_s:.3f}  |  "
              f"IForest     F1={f1_i:.3f} Rec={rec_i:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────
# 3.  Plotting
# ─────────────────────────────────────────────────────────────────
COLOURS = {
    "global_iqr":      "#e74c3c",
    "session_iqr":     "#9b59b6",
    "isolation_forest": "#f39c12",
}
LABELS = {
    "global_iqr":       "Global IQR  (paper baseline)",
    "session_iqr":      "Session-Aware IQR",
    "isolation_forest": "Isolation Forest",
}
MARKERS = {
    "global_iqr":      "o",
    "session_iqr":     "s",
    "isolation_forest": "^",
}


def plot_ablation(results: dict, save_path: str):
    """
    Two-panel figure:
      Left : F1 score vs ϑ
      Right: Recall vs ϑ
    """
    thetas  = sorted(results.keys())
    methods = ["global_iqr", "session_iqr", "isolation_forest"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   sharey=False, constrained_layout=True)
    fig.suptitle(
        "Ablation: FDI Detection Performance vs. Attack Intensity ϑ\n"
        "(ϑ = extra minutesAvailable injected — larger ϑ = stronger / easier-to-detect attack)",
        fontsize=12, fontweight="bold",
    )

    for m in methods:
        f1s  = [results[t][m]["f1"]     for t in thetas]
        recs = [results[t][m]["recall"] for t in thetas]

        ax1.plot(thetas, f1s,  color=COLOURS[m], marker=MARKERS[m],
                 lw=2, ms=7, label=LABELS[m])
        ax2.plot(thetas, recs, color=COLOURS[m], marker=MARKERS[m],
                 lw=2, ms=7, label=LABELS[m])

    for ax, ylabel, title in [
        (ax1, "F1 Score",  "(a) F1 Score vs. Attack Intensity ϑ"),
        (ax2, "Recall",    "(b) Recall vs. Attack Intensity ϑ"),
    ]:
        ax.set_xlabel("ϑ (extra minutes injected by attacker)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(thetas)
        ax.set_xticklabels([f"ϑ={t}" for t in thetas], rotation=20, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35, ls="--")

    # Highlight "subtle attack" zone
    for ax in [ax1, ax2]:
        ax.axvspan(0, 7, color="#ffe0b2", alpha=0.35, label="Subtle zone (ϑ < 5)")
        ax.axvline(5, color="#e67e22", lw=1.2, ls=":", alpha=0.7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Ablation plot → {save_path}")


def print_ablation_table(results: dict):
    thetas  = sorted(results.keys())
    methods = ["global_iqr", "session_iqr", "isolation_forest"]
    short   = {"global_iqr": "Global-IQR", "session_iqr": "Sess-IQR",
               "isolation_forest": "IForest"}
    W = 78
    print("\n" + "═"*W)
    print("  ABLATION: F1 SCORE vs ATTACK INTENSITY ϑ")
    print(f"  {'ϑ (min)':<10} " +
          "  ".join(f"{short[m]:>15}" for m in methods))
    print("─"*W)
    for t in thetas:
        row = f"  {t:<10} "
        for m in methods:
            row += f"  {results[t][m]['f1']:>12.4f}   "
        print(row)
    print("═"*W)
    print("  ABLATION: RECALL vs ATTACK INTENSITY ϑ")
    print(f"  {'ϑ (min)':<10} " +
          "  ".join(f"{short[m]:>15}" for m in methods))
    print("─"*W)
    for t in thetas:
        row = f"  {t:<10} "
        for m in methods:
            row += f"  {results[t][m]['recall']:>12.4f}   "
        print(row)
    print("═"*W + "\n")


# ─────────────────────────────────────────────────────────────────
# 4.  Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")
    ckpt_path = os.path.join(CKPT_DIR, "narx_best.pt")
    scl_path  = os.path.join(CKPT_DIR, "scalers.pkl")

    # Load data
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

    # Pre-train Isolation Forest on CLEAN training EoE
    # (contamination pre-tuned to 0.01 from isolation_forest.py run)
    model.eval()
    X_tr, y_tr = data["raw"]["X_train_w"], data["raw"]["y_train_w"]
    with torch.no_grad():
        tr_preds_sc = model(
            torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()
    y_tr_true = scaler_y.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    y_tr_pred = scaler_y.inverse_transform(tr_preds_sc.reshape(-1, 1)).flatten()
    eoe_tr_clean = np.abs(y_tr_true - y_tr_pred)

    ifo = IsolationForest(n_estimators=IF_N_ESTIMATORS,
                          contamination=IF_CONTAMINATION, random_state=42)
    ifo.fit(eoe_tr_clean.reshape(-1, 1))
    print(f"[INFO] IsolationForest trained (contamination={IF_CONTAMINATION},"
          f" n_estimators={IF_N_ESTIMATORS})\n")

    # Run ablation
    print("═"*60)
    print("  ABLATION SWEEP: ϑ ∈", THETA_VALUES, "minutes")
    print("═"*60)
    results = run_ablation(model, data["raw"], scaler_y, df_estim, ifo)

    # Print table
    print_ablation_table(results)

    # Plot
    plot_ablation(results,
                  save_path=os.path.join(OUT_DIR, "ablation_intensity.png"))
