"""
Isolation Forest FDI Detector
─────────────────────────────────────────────────────────────────
Drop-in replacement for IQR+window:

  1.  Run NARX on the clean training set  →  build clean EoE distribution
  2.  Tune IsolationForest.contamination on a val slice (maximise F1)
  3.  Fit final IForest on all clean training EoE
  4.  At inference: score each estimation EoE point
        score < 0  →  anomaly label = 1  (FDI suspected)
        score ≥ 0  →  normal     label = 0
  5.  Report confusion matrix + Acc / Prec / Rec / F1
  6.  Three-way comparison: Global IQR vs Session-Aware IQR vs IForest

Why Isolation Forest beats static percentile thresholds
────────────────────────────────────────────────────────
IQR-based detection assumes a single, symmetric, roughly Gaussian
error distribution across all sessions and all time.  Real EoE from
NARX is multi-modal: short sessions produce small, tight errors;
long sessions show a wider error spread; session transitions cause
transient spikes.  A single Q1/Q3/IQR cannot adapt to this
heteroscedastic structure.

Isolation Forest works by:
  • Randomly partitioning the feature space with axis-aligned cuts
  • Anomalies (distant EoE outliers) require fewer cuts to isolate
      → shorter average path length → lower anomaly score
  • The contamination hyperparam is the only prior needed,
    and it is tuned on labelled val data rather than hardcoded

This means IF:
  - Handles EoE distributions with arbitrary shape (multi-modal, fat-tailed)
  - Automatically adapts per feature dimension if you feed richer context
  - Does not suffer from "global band" contamination by session-boundary spikes
  - No manual window size q required

Usage (from narx_ev_fdi/ directory):
    python3 -m src.eval.isolation_forest
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
)
import torch

# ── project root ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset import build_datasets
from src.models.narx  import NARXNet
from src.eval.evaluate import (
    inject_fdi_attacks, evaluate, evaluate_session_aware,
    print_comparison_table,
)

CKPT_DIR   = os.path.join(ROOT, "checkpoints")
OUT_DIR    = os.path.join(ROOT, "results")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────
def _infer_eoe(model, X_sc, y_sc, scaler_y):
    """Forward pass → inverse-transform → EoE and original-scale arrays."""
    model.eval()
    with torch.no_grad():
        preds_sc = model(
            torch.tensor(X_sc, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()
    y_true = scaler_y.inverse_transform(y_sc.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    eoe_clean = np.abs(y_true - y_pred)
    return y_true, y_pred, eoe_clean


def _metrics(gt, pred):
    cm   = confusion_matrix(gt, pred)
    acc  = accuracy_score (gt, pred)
    prec = precision_score(gt, pred, zero_division=0)
    rec  = recall_score   (gt, pred, zero_division=0)
    f1   = f1_score       (gt, pred, zero_division=0)
    return cm, acc, prec, rec, f1


# ─────────────────────────────────────────────────────────────────
# 2. Contamination tuning on validation slice
# ─────────────────────────────────────────────────────────────────
def tune_contamination(
    eoe_train_clean: np.ndarray,
    eoe_val:         np.ndarray,
    gt_val:          np.ndarray,
    candidates:      list = None,
    n_estimators:    int  = 200,
) -> float:
    """
    Grid-search contamination in `candidates` to maximise F1 on val.

    Parameters
    ----------
    eoe_train_clean : clean EoE from training windows (no attacks)
    eoe_val         : EoE on validation slice (may contain injected attacks)
    gt_val          : ground-truth labels for val slice (0/1)
    candidates      : contamination values to try
    n_estimators    : trees per IF model

    Returns
    -------
    best_c : float — contamination value with highest val F1
    """
    if candidates is None:
        candidates = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    best_c, best_f1 = 0.05, -1.0
    print("\n[TUNING] Contamination grid search on validation slice:")
    print(f"  {'Contamination':>15}  {'F1':>8}  {'Recall':>8}  {'Precision':>10}")

    for c in candidates:
        ifo = IsolationForest(n_estimators=n_estimators, contamination=c,
                              random_state=42)
        ifo.fit(eoe_train_clean.reshape(-1, 1))
        scores   = ifo.decision_function(eoe_val.reshape(-1, 1))
        pred_val = (scores < 0).astype(int)   # negative score → anomaly

        _, _, prec, rec, f1 = _metrics(gt_val, pred_val)
        print(f"  {c:>15.3f}  {f1:>8.4f}  {rec:>8.4f}  {prec:>10.4f}")

        if f1 > best_f1:
            best_f1, best_c = f1, c

    print(f"  → Best contamination = {best_c}  (val F1 = {best_f1:.4f})\n")
    return best_c


# ─────────────────────────────────────────────────────────────────
# 3. Main IF evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate_isolation_forest(
    model,
    raw_data:    dict,
    scaler_y,
    val_fraction: float = 0.15,
    n_estimators: int   = 200,
) -> dict:
    """
    Full Isolation Forest FDI detection pipeline.

    Parameters
    ----------
    model        : trained NARXNet
    raw_data     : dict from build_datasets (keys X_train_w, y_train_w,
                   X_estim_w, y_estim_w)
    scaler_y     : fitted MinMaxScaler for the target
    val_fraction : fraction of training windows used for contamination tuning
    n_estimators : number of trees in IsolationForest

    Returns
    -------
    results dict with eoe, detected, gt_labels, metrics, confusion_matrix
    """
    # ── Step 1: Clean train EoE (no attacks injected) ──────────────
    y_tr_true, _, eoe_tr_clean = _infer_eoe(
        model, raw_data["X_train_w"], raw_data["y_train_w"], scaler_y
    )

    # ── Step 2: Attack-injected estimation EoE ─────────────────────
    y_es_true, y_es_pred, eoe_es_clean = _infer_eoe(
        model, raw_data["X_estim_w"], raw_data["y_estim_w"], scaler_y
    )
    y_attacked, gt_labels = inject_fdi_attacks(y_es_true, attack_fraction=0.10, seed=42)
    eoe_attacked = np.abs(y_attacked - y_es_pred)

    # ── Step 3: Carve out a val slice from training for tuning ──────
    n_tr    = len(eoe_tr_clean)
    n_val   = int(n_tr * val_fraction)
    eoe_tr_fit  = eoe_tr_clean[:n_tr - n_val]   # clean portion for fitting
    eoe_val_raw = eoe_tr_clean[n_tr - n_val:]   # val clean EoE

    # Inject attacks into val slice too so we can evaluate detection
    y_val_true = y_tr_true[n_tr - n_val:]
    y_val_attacked, gt_val = inject_fdi_attacks(y_val_true,
                                                attack_fraction=0.10, seed=7)
    y_val_pred           = y_es_pred[:len(y_val_attacked)] if False else (
        scaler_y.inverse_transform(
            model(torch.tensor(raw_data["X_train_w"][n_tr - n_val:],
                               dtype=torch.float32, device=DEVICE)
                  ).detach().cpu().numpy()
        ).flatten()
    )
    eoe_val = np.abs(y_val_attacked - y_val_pred)

    # ── Step 4: Tune contamination ──────────────────────────────────
    best_c = tune_contamination(eoe_tr_fit, eoe_val, gt_val,
                                n_estimators=n_estimators)

    # ── Step 5: Fit final model on ALL clean training EoE ──────────
    ifo = IsolationForest(n_estimators=n_estimators,
                          contamination=best_c, random_state=42)
    ifo.fit(eoe_tr_clean.reshape(-1, 1))

    # ── Step 6: Score estimation set (attacked) ─────────────────────
    scores   = ifo.decision_function(eoe_attacked.reshape(-1, 1))
    detected = (scores < 0).astype(int)

    # ── Metrics ────────────────────────────────────────────────────
    cm, acc, prec, rec, f1 = _metrics(gt_labels, detected)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    print("═"*56)
    print("  ISOLATION FOREST FDI DETECTION RESULTS")
    print("═"*56)
    print(f"  Contamination (tuned): {best_c}")
    print(f"  Total Samples         : {len(gt_labels)}")
    print(f"  True Attacks (gt=1)   : {int(gt_labels.sum())}")
    print(f"  Detected (pred=1)     : {int(detected.sum())}")
    print(f"  TP={int(tp)}  FP={int(fp)}  FN={int(fn)}  TN={int(tn)}")
    print(f"  Accuracy  : {acc:.4f}   Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}   F1 Score  : {f1:.4f}")
    print("═"*56 + "\n")

    return {
        "eoe"             : eoe_attacked,
        "eoe_clean"       : eoe_es_clean,
        "scores"          : scores,
        "y_true"          : y_es_true,
        "y_pred"          : y_es_pred,
        "y_attacked"      : y_attacked,
        "gt_labels"       : gt_labels,
        "detected"        : detected,
        "contamination"   : best_c,
        "iforest"         : ifo,
        "metrics"         : {"accuracy": acc, "precision": prec,
                             "recall": rec,   "f1": f1},
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────
# 4. Three-way comparison print
# ─────────────────────────────────────────────────────────────────
def print_three_way_table(glob: dict, sess: dict, ifo: dict):
    mg, ms, mi = glob["metrics"], sess["metrics"], ifo["metrics"]

    def _unpack(cm):
        r = cm.ravel()
        return (r if len(r) == 4 else [r[0], 0, 0, 0])

    gtn, gfp, gfn, gtp = _unpack(glob["confusion_matrix"])
    stn, sfp, sfn, stp = _unpack(sess["confusion_matrix"])
    itn, ifp, ifn, itp = _unpack(ifo["confusion_matrix"])

    W = 72
    print("\n" + "═"*W)
    print(f"  THREE-WAY FDI DETECTION COMPARISON")
    print("═"*W)
    print(f"  {'Metric':<18} {'Global IQR':>14} {'Session IQR':>14} {'IsolationForest':>18}")
    print("─"*W)
    for label, gv, sv, iv in [
        ("TP",        gtp, stp, itp),
        ("FP",        gfp, sfp, ifp),
        ("FN",        gfn, sfn, ifn),
        ("TN",        gtn, stn, itn),
    ]:
        print(f"  {label:<18} {int(gv):>14} {int(sv):>14} {int(iv):>18}")
    print("─"*W)
    for key, label in [("accuracy","Accuracy"),("precision","Precision"),
                       ("recall","Recall"),("f1","F1")]:
        gi, si, ii = mg[key], ms[key], mi[key]
        best = max(gi, si, ii)
        def fmt(v): return f"{'**' if v==best else '  '}{v:.4f}{'**' if v==best else '  '}"
        print(f"  {label:<18} {fmt(gi):>14} {fmt(si):>14} {fmt(ii):>18}")
    print("═"*W)
    print("  ** = best in row")
    print()


# ─────────────────────────────────────────────────────────────────
# 5. Comparison plot
# ─────────────────────────────────────────────────────────────────
def plot_three_way(glob: dict, sess: dict, ifo: dict, save_path: str):
    """
    Five-panel figure:
      (a) EoE + Global IQR bounds
      (b) EoE + IF anomaly scores (coloured by label)
      (c) GT vs all three detectors
      (d) IF anomaly scores over time with zero threshold
      (e) Side-by-side-by-side confusion matrices
    """
    eoe  = glob["eoe"]
    gt   = glob["gt_labels"]
    idx  = np.arange(len(eoe))
    scores = ifo["scores"]

    fig = plt.figure(figsize=(15, 18))
    gs  = fig.add_gridspec(5, 3, hspace=0.55, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])
    ax_d = fig.add_subplot(gs[3, :])
    ax_e1 = fig.add_subplot(gs[4, 0])
    ax_e2 = fig.add_subplot(gs[4, 1])
    ax_e3 = fig.add_subplot(gs[4, 2])

    fig.suptitle(
        "Three-Way FDI Detection Comparison\n"
        "Global IQR  |  Session-Aware IQR  |  Isolation Forest",
        fontsize=13, fontweight="bold",
    )

    # ── (a) Global IQR EoE ───────────────────────────────────────
    lb_g, ub_g = glob["lb"], glob["ub"]
    ax_a.plot(idx, eoe, color="#4c9be8", lw=0.7, label="EoE")
    ax_a.axhline(ub_g, color="#e84c4c", lw=1.4, ls="--",
                 label=f"Global UB={ub_g:.4f}")
    ax_a.fill_between(idx, lb_g, ub_g, color="#4c9be8", alpha=0.07)
    spk_g = np.where(glob["spikes"])[0]
    if len(spk_g):
        ax_a.scatter(spk_g, eoe[spk_g], color="#e84c4c", s=8, zorder=3,
                     label=f"Spikes ({len(spk_g)})")
    ax_a.set_title("(a) Global IQR — fixed band", fontsize=10)
    ax_a.set_ylabel("EoE"); ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # ── (b) Isolation Forest scores overlaid on EoE ──────────────
    anomaly_mask = scores < 0
    ax_b.plot(idx, eoe, color="#4c9be8", lw=0.7, label="EoE", zorder=2)
    ax_b.scatter(idx[anomaly_mask],  eoe[anomaly_mask],
                 color="#e84c4c", s=10, zorder=3,
                 label=f"IF anomaly ({anomaly_mask.sum()})")
    ax_b.scatter(idx[~anomaly_mask], eoe[~anomaly_mask],
                 color="#2ecc71", s=4, alpha=0.4, zorder=2,
                 label="IF normal")
    ax_b.set_title("(b) Isolation Forest — point-wise anomaly classification",
                   fontsize=10)
    ax_b.set_ylabel("EoE"); ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)

    # ── (c) GT vs all three detectors ────────────────────────────
    mg, ms, mi = glob["metrics"], sess["metrics"], ifo["metrics"]
    ax_c.step(idx, gt,               where="post", color="#2ecc71", lw=1.3,
              label="Ground Truth")
    ax_c.step(idx, glob["detected"], where="post", color="#e74c3c", lw=0.9,
              ls="--", label=f"Global IQR   F1={mg['f1']:.3f}")
    ax_c.step(idx, sess["detected"], where="post", color="#9b59b6", lw=0.9,
              ls=":",  label=f"Session IQR  F1={ms['f1']:.3f}")
    ax_c.step(idx, ifo["detected"],  where="post", color="#f39c12", lw=0.9,
              ls="-.", label=f"IsolForest   F1={mi['f1']:.3f}")
    ax_c.set_yticks([0, 1]); ax_c.set_yticklabels(["Normal", "Attack"])
    ax_c.set_title("(c) Detection labels — all three methods", fontsize=10)
    ax_c.legend(fontsize=8); ax_c.grid(alpha=0.3)

    # ── (d) IF raw anomaly scores with threshold line ─────────────
    ax_d.plot(idx, scores, color="#8e44ad", lw=0.8, label="IF score")
    ax_d.axhline(0, color="#e84c4c", lw=1.4, ls="--", label="Threshold = 0")
    ax_d.fill_between(idx, scores, 0, where=scores < 0,
                      color="#e84c4c", alpha=0.18, label="Anomaly region")
    ax_d.set_title("(d) Isolation Forest raw anomaly scores (negative = anomaly)",
                   fontsize=10)
    ax_d.set_ylabel("IF Score"); ax_d.set_xlabel("Sample Index")
    ax_d.legend(fontsize=8); ax_d.grid(alpha=0.3)

    # ── (e) Three confusion matrices ─────────────────────────────
    cmaps = ["Blues", "Purples", "Oranges"]
    titles = [
        f"Global IQR\nAcc={mg['accuracy']:.3f} F1={mg['f1']:.3f}",
        f"Session IQR\nAcc={ms['accuracy']:.3f} F1={ms['f1']:.3f}",
        f"IsolForest\nAcc={mi['accuracy']:.3f} F1={mi['f1']:.3f}",
    ]
    for cm_d, ax_i, cmp, ttl in zip(
        [glob["confusion_matrix"], sess["confusion_matrix"], ifo["confusion_matrix"]],
        [ax_e1, ax_e2, ax_e3], cmaps, titles,
    ):
        ConfusionMatrixDisplay(cm_d, display_labels=["Normal", "Attack"]).plot(
            ax=ax_i, colorbar=False, cmap=cmp
        )
        ax_i.set_title(ttl, fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Three-way plot → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")
    ckpt_path = os.path.join(CKPT_DIR, "narx_best.pt")
    scl_path  = os.path.join(CKPT_DIR, "scalers.pkl")

    # ── Load data ──────────────────────────────────────────────────
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
    print(f"[INFO] Model loaded from {ckpt_path}\n")

    # ── Run all three detectors ────────────────────────────────────
    print("─"*56)
    print("  1/3  GLOBAL IQR BASELINE")
    print("─"*56)
    glob_results  = evaluate(model, data["raw"], scaler_y, q=5, k=5.0)

    print("─"*56)
    print("  2/3  SESSION-AWARE IQR")
    print("─"*56)
    sess_results  = evaluate_session_aware(model, data["raw"], scaler_y,
                                           df_estim, q=5, k=5.0)

    print("─"*56)
    print("  3/3  ISOLATION FOREST")
    print("─"*56)
    ifo_results   = evaluate_isolation_forest(model, data["raw"], scaler_y,
                                              val_fraction=0.15, n_estimators=200)

    # ── Three-way table ────────────────────────────────────────────
    print_three_way_table(glob_results, sess_results, ifo_results)

    # ── Plot ───────────────────────────────────────────────────────
    plot_three_way(glob_results, sess_results, ifo_results,
                   save_path=os.path.join(OUT_DIR, "three_way_comparison.png"))
