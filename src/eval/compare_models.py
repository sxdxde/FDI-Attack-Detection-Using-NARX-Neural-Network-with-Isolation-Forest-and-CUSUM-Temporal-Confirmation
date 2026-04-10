"""
NARX vs Attention-BiLSTM — Model Comparison & Graph
─────────────────────────────────────────────────────────────────
Loads both trained checkpoints, runs inference on the same data,
injects identical FDI attacks, computes all metrics, and produces
a 6-panel publication-quality comparison figure.

Usage (from narx_ev_fdi/ root):
    python -m src.eval.compare_models

Output:
    results/model_comparison.png
    results/model_comparison_metrics.txt
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, f1_score, recall_score, precision_score, accuracy_score,
    ConfusionMatrixDisplay,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset            import build_datasets, EXOG_COLS
from src.models.narx             import NARXNet
from src.models.attention_bilstm import AttentionBiLSTM
from src.eval.evaluate           import inject_fdi_attacks

CKPT_DIR  = os.path.join(ROOT, "checkpoints")
OUT_DIR   = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── shared attack injection params (must match IF+CUSUM pipeline) ─
ATTACK_FRACTION = 0.10
ATTACK_SEED     = 42

# ── colour palette ────────────────────────────────────────────────
C_GT    = "#2ecc71"   # ground truth — green
C_NARX  = "#e74c3c"   # NARX — red
C_LSTM  = "#3498db"   # BiLSTM — blue
C_BG    = "#1a1a2e"   # dark background
C_PANEL = "#16213e"
C_TEXT  = "#e0e0e0"


# ─────────────────────────────────────────────────────────────────
# 1. Inference helpers
# ─────────────────────────────────────────────────────────────────
def predict_narx(model: NARXNet, X_w: np.ndarray, scaler_y) -> np.ndarray:
    """Run NARX inference on pre-built flat windows."""
    model.eval()
    with torch.no_grad():
        preds_sc = model(
            torch.tensor(X_w, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy().flatten()
    return scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()


def predict_bilstm(model: AttentionBiLSTM, X_w: np.ndarray, scaler_y,
                   return_attn: bool = False):
    """Run BiLSTM inference on pre-built 3-D windows."""
    model.eval()
    all_pred, all_attn = [], []
    batch = 512
    with torch.no_grad():
        for i in range(0, len(X_w), batch):
            xb = torch.tensor(X_w[i:i+batch], dtype=torch.float32, device=DEVICE)
            if return_attn:
                out, attn = model(xb, return_attention=True)
                all_attn.append(attn.cpu().numpy())
            else:
                out = model(xb)
            all_pred.append(out.cpu().numpy())
    preds_sc = np.concatenate(all_pred).flatten()
    y_pred   = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    if return_attn:
        return y_pred, np.concatenate(all_attn, axis=0)
    return y_pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def detection_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    f1   = f1_score(gt, pred, zero_division=0)
    prec = precision_score(gt, pred, zero_division=0)
    rec  = recall_score(gt, pred, zero_division=0)
    acc  = accuracy_score(gt, pred)
    cm   = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
    return {"F1": f1, "Precision": prec, "Recall": rec, "Accuracy": acc,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "cm": cm}


def iqr_detect(eoe_clean: np.ndarray, eoe_attacked: np.ndarray,
               k: float = 5.0, q: int = 5) -> np.ndarray:
    """Simple IQR + sliding-window detector (same as evaluate.py)."""
    q1, q3 = np.percentile(eoe_clean, 25), np.percentile(eoe_clean, 75)
    iqr = q3 - q1
    ub  = q3 + k * iqr
    lb  = q1 - k * iqr
    spikes = (eoe_attacked < lb) | (eoe_attacked > ub)
    n = len(spikes)
    labels = np.zeros(n, dtype=int)
    for t in range(q - 1, n):
        if spikes[t - q + 1: t + 1].all():
            labels[t] = 1
    return labels


# ─────────────────────────────────────────────────────────────────
# 2. Main comparison runner
# ─────────────────────────────────────────────────────────────────
def run_comparison(df_train: pd.DataFrame, df_estim: pd.DataFrame) -> dict:

    # ── Load NARX ────────────────────────────────────────────────
    with open(os.path.join(CKPT_DIR, "scalers.pkl"), "rb") as f:
        narx_scalers = pickle.load(f)

    narx_data = build_datasets(df_train, df_estim, mx=2, my=2,
                               val_ratio=0.15, test_ratio=0.15,
                               batch_size=64, model_type="narx")
    narx_model = NARXNet(
        input_size=narx_data["shapes"]["input_size"], hidden_size=10
    ).to(DEVICE)
    narx_model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "narx_best.pt"), map_location=DEVICE)
    )
    print("[INFO] NARX loaded.")

    # ── Load BiLSTM ───────────────────────────────────────────────
    bilstm_scl_path = os.path.join(CKPT_DIR, "bilstm_scalers.pkl")
    if not os.path.exists(bilstm_scl_path):
        bilstm_scl_path = os.path.join(CKPT_DIR, "scalers.pkl")  # fallback
    with open(bilstm_scl_path, "rb") as f:
        bilstm_scalers = pickle.load(f)

    bilstm_data = build_datasets(df_train, df_estim,
                                 val_ratio=0.15, test_ratio=0.15,
                                 batch_size=64, model_type="bilstm", seq_len=4)
    n_feat = bilstm_data["shapes"].get("n_features", len(EXOG_COLS))
    seq_len = bilstm_data["shapes"].get("seq_len", 4)

    bilstm_model = AttentionBiLSTM(
        n_features=n_feat, seq_len=seq_len, hidden_size=64, num_layers=2, dropout=0.3
    ).to(DEVICE)
    bilstm_model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "bilstm_best.pt"), map_location=DEVICE)
    )
    print("[INFO] Attention-BiLSTM loaded.")

    # ── Get raw targets on estimation set ────────────────────────
    scaler_y_narx  = narx_scalers["y"]
    scaler_y_lstm  = bilstm_scalers["y"]

    # NARX estim predictions
    X_es_narx = narx_data["raw"]["X_estim_w"]
    y_es_narx = narx_data["raw"]["y_estim_w"]
    y_true_narx = scaler_y_narx.inverse_transform(
        y_es_narx.reshape(-1, 1)).flatten()
    y_pred_narx = predict_narx(narx_model, X_es_narx, scaler_y_narx)

    # BiLSTM estim predictions + attention weights
    X_es_bilstm = bilstm_data["raw"]["X_estim_w"]
    y_es_bilstm = bilstm_data["raw"]["y_estim_w"]
    y_true_bilstm = scaler_y_lstm.inverse_transform(
        y_es_bilstm.reshape(-1, 1)).flatten()
    y_pred_bilstm, attn_weights = predict_bilstm(
        bilstm_model, X_es_bilstm, scaler_y_lstm, return_attn=True)

    # ── Align lengths (take the shorter) ─────────────────────────
    n = min(len(y_true_narx), len(y_true_bilstm))
    y_true        = y_true_narx[:n]
    y_pred_narx   = y_pred_narx[:n]
    y_pred_bilstm = y_pred_bilstm[:n]
    attn_weights  = attn_weights[:n]

    # ── Regression metrics ────────────────────────────────────────
    reg_narx  = regression_metrics(y_true, y_pred_narx)
    reg_lstm  = regression_metrics(y_true, y_pred_bilstm)

    print("\n══ Regression Metrics ══")
    for key in ["MSE", "RMSE", "MAE", "R2"]:
        n_val = reg_narx[key]
        l_val = reg_lstm[key]
        delta = l_val - n_val
        sign  = "+" if delta >= 0 else ""
        better = "BiLSTM ↑" if (key == "R2" and delta > 0) or (key != "R2" and delta < 0) else "NARX ↑"
        print(f"  {key:<6}: NARX={n_val:.4e}  BiLSTM={l_val:.4e}  ({sign}{delta:.4e})  {better}")

    # ── FDI detection metrics ─────────────────────────────────────
    y_attacked, gt = inject_fdi_attacks(y_true, attack_fraction=ATTACK_FRACTION,
                                        seed=ATTACK_SEED)
    eoe_narx  = np.abs(y_attacked - y_pred_narx)
    eoe_bilstm = np.abs(y_attacked - y_pred_bilstm)
    eoe_clean_narx  = np.abs(y_true - y_pred_narx)
    eoe_clean_bilstm = np.abs(y_true - y_pred_bilstm)

    det_narx  = iqr_detect(eoe_clean_narx,  eoe_narx,  k=5.0, q=5)
    det_bilstm = iqr_detect(eoe_clean_bilstm, eoe_bilstm, k=5.0, q=5)

    fdi_narx  = detection_metrics(gt, det_narx)
    fdi_lstm  = detection_metrics(gt, det_bilstm)

    print("\n══ FDI Detection Metrics ══")
    for key in ["F1", "Precision", "Recall", "Accuracy"]:
        print(f"  {key:<12}: NARX={fdi_narx[key]:.4f}  BiLSTM={fdi_lstm[key]:.4f}")

    return {
        "y_true": y_true, "y_attacked": y_attacked, "gt": gt,
        "y_pred_narx":   y_pred_narx,
        "y_pred_bilstm": y_pred_bilstm,
        "eoe_narx":    eoe_narx,
        "eoe_bilstm":  eoe_bilstm,
        "eoe_clean_narx":   eoe_clean_narx,
        "eoe_clean_bilstm": eoe_clean_bilstm,
        "det_narx":    det_narx,
        "det_bilstm":  det_bilstm,
        "reg_narx":    reg_narx,
        "reg_lstm":    reg_lstm,
        "fdi_narx":    fdi_narx,
        "fdi_lstm":    fdi_lstm,
        "attn_weights": attn_weights,   # (N, seq_len)
    }


# ─────────────────────────────────────────────────────────────────
# 3. Six-panel comparison figure
# ─────────────────────────────────────────────────────────────────
def plot_comparison(d: dict, save_path: str):
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor":   C_PANEL,
        "axes.edgecolor":   "#444466",
        "axes.labelcolor":  C_TEXT,
        "xtick.color":      C_TEXT,
        "ytick.color":      C_TEXT,
        "text.color":       C_TEXT,
        "grid.color":       "#333355",
        "grid.alpha":       0.4,
        "legend.facecolor": "#0f1124",
        "legend.edgecolor": "#444466",
    })

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor(C_BG)
    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.96, top=0.95, bottom=0.04,
    )

    idx       = np.arange(len(d["y_true"]))
    VIEW      = min(500, len(idx))   # show first 500 points for clarity in panels a/b
    y_true    = d["y_true"]
    y_att     = d["y_attacked"]
    gt        = d["gt"]

    title_kw = dict(fontsize=11, fontweight="bold", color="#aaccff", pad=8)
    lw_thin = 0.7

    # ── (a) NARX — prediction quality ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(idx[:VIEW], y_true[:VIEW],
              color=C_GT, lw=lw_thin, alpha=0.85, label="y_true")
    ax_a.plot(idx[:VIEW], d["y_pred_narx"][:VIEW],
              color=C_NARX, lw=lw_thin, alpha=0.85, label="NARX ŷ", ls="--")
    ax_a.set_title("(a) Prediction — NARX", **title_kw)
    ax_a.set_ylabel("kWh / timestep")
    m = d["reg_narx"]
    ax_a.legend(fontsize=8, title=f"MSE={m['MSE']:.3e}  R²={m['R2']:.3f}",
                title_fontsize=8)
    ax_a.grid(True)

    # ── (b) BiLSTM — prediction quality ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(idx[:VIEW], y_true[:VIEW],
              color=C_GT, lw=lw_thin, alpha=0.85, label="y_true")
    ax_b.plot(idx[:VIEW], d["y_pred_bilstm"][:VIEW],
              color=C_LSTM, lw=lw_thin, alpha=0.85, label="BiLSTM ŷ", ls="--")
    ax_b.set_title("(b) Prediction — Attention-BiLSTM", **title_kw)
    ax_b.set_ylabel("kWh / timestep")
    m = d["reg_lstm"]
    ax_b.legend(fontsize=8, title=f"MSE={m['MSE']:.3e}  R²={m['R2']:.3f}",
                title_fontsize=8)
    ax_b.grid(True)

    # ── (c) EoE violin + box plot ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    eoe_data = [d["eoe_narx"], d["eoe_bilstm"]]
    vp = ax_c.violinplot(eoe_data, positions=[1, 2], showmedians=True,
                         showextrema=True)
    vp["bodies"][0].set_facecolor(C_NARX)
    vp["bodies"][0].set_alpha(0.55)
    vp["bodies"][1].set_facecolor(C_LSTM)
    vp["bodies"][1].set_alpha(0.55)
    for pc in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[pc].set_color(C_TEXT)
        vp[pc].set_linewidth(1.2)
    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels(["NARX", "Attention-BiLSTM"])
    ax_c.set_title("(c) EoE Distribution (|y_attacked - ŷ|)", **title_kw)
    ax_c.set_ylabel("Error of Estimation (EoE)")
    ax_c.grid(True, axis="y")
    # Overlay medians as text
    for xi, eoe, col in zip([1, 2], eoe_data, [C_NARX, C_LSTM]):
        ax_c.text(xi, np.median(eoe) * 1.05,
                  f"med={np.median(eoe):.4f}", ha="center",
                  fontsize=8, color=col, fontweight="bold")

    # ── (d) Metrics grouped bar chart ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    metrics_reg  = ["MSE×1e5", "MAE×1e3", "R²"]
    metrics_fdi  = ["F1", "Precision", "Recall"]

    def _safe(d_model, key):
        v = d_model[key]
        if key.startswith("MSE"):  return v * 1e5
        if key.startswith("MAE"):  return v * 1e3
        return v

    reg_keys  = [("MSE", "MSE×1e5"), ("MAE", "MAE×1e3"), ("R2", "R²")]
    fdi_keys  = [("F1", "F1"), ("Precision", "Precision"), ("Recall", "Recall")]

    labels_bar = [lbl for _, lbl in reg_keys] + [lbl for _, lbl in fdi_keys]
    narx_vals  = ([d["reg_narx"][k] * (1e5 if k == "MSE" else 1e3 if k == "MAE" else 1)
                   for k, _ in reg_keys] +
                  [d["fdi_narx"][k]  for k, _ in fdi_keys])
    lstm_vals  = ([d["reg_lstm"][k] * (1e5 if k == "MSE" else 1e3 if k == "MAE" else 1)
                   for k, _ in reg_keys] +
                  [d["fdi_lstm"][k]  for k, _ in fdi_keys])

    x_bar = np.arange(len(labels_bar))
    bw    = 0.35
    bars_n = ax_d.bar(x_bar - bw/2, narx_vals, bw, label="NARX",
                       color=C_NARX, alpha=0.8, edgecolor="#ff9999")
    bars_l = ax_d.bar(x_bar + bw/2, lstm_vals, bw, label="Attention-BiLSTM",
                       color=C_LSTM, alpha=0.8, edgecolor="#99ccff")

    for bar in list(bars_n) + list(bars_l):
        h = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                  f"{h:.3f}", ha="center", va="bottom", fontsize=7, color=C_TEXT)

    ax_d.set_xticks(x_bar)
    ax_d.set_xticklabels(labels_bar, fontsize=9)
    ax_d.set_title("(d) Quantitative Metrics — NARX vs Attention-BiLSTM", **title_kw)
    ax_d.legend(fontsize=9)
    ax_d.grid(True, axis="y")
    # Add a separator line between regression and detection groups
    ax_d.axvline(x=2.5, color="#888899", lw=1.2, ls="--", alpha=0.7)
    ax_d.text(1.0, ax_d.get_ylim()[1] * 0.92,
              "Regression ↓ better", ha="center", fontsize=8, color="#aaaacc")
    ax_d.text(4.0, ax_d.get_ylim()[1] * 0.92,
              "Detection ↑ better",  ha="center", fontsize=8, color="#aaaacc")

    # ── (e) Attention weight heatmap ─────────────────────────────
    ax_e = fig.add_subplot(gs[2, :])
    attn = d["attn_weights"]          # (N, seq_len)
    VIEW_AT = min(300, len(attn))
    im = ax_e.imshow(
        attn[:VIEW_AT].T,
        aspect="auto", cmap="magma",
        interpolation="nearest",
        origin="lower",
    )
    plt.colorbar(im, ax=ax_e, fraction=0.015, pad=0.01, label="Attention weight")
    ax_e.set_yticks(range(attn.shape[1]))
    ax_e.set_yticklabels([f"t−{attn.shape[1]-i}" for i in range(attn.shape[1])])
    ax_e.set_xlabel("Sample index (first 300 estim samples)")
    ax_e.set_title(
        "(e) Attention-BiLSTM — Learned Attention Weights per Input Timestep\n"
        "(brighter = model attended more to that lag for the prediction)",
        **title_kw,
    )

    # ── (f) Detection labels: GT / NARX / BiLSTM ─────────────────
    ax_f = fig.add_subplot(gs[3, :])
    VIEW_F = min(600, len(gt))
    idx_f  = np.arange(VIEW_F)

    ax_f.fill_between(idx_f, 0, gt[:VIEW_F],
                      step="post", color=C_GT, alpha=0.25, label="Ground Truth")
    ax_f.step(idx_f, gt[:VIEW_F],
              where="post", color=C_GT, lw=1.2, alpha=0.9)
    ax_f.step(idx_f, d["det_narx"][:VIEW_F],
              where="post", color=C_NARX, lw=1.0, ls="--", alpha=0.85,
              label=f"NARX IQR    F1={d['fdi_narx']['F1']:.3f}")
    ax_f.step(idx_f, d["det_bilstm"][:VIEW_F],
              where="post", color=C_LSTM, lw=1.0, ls=":",  alpha=0.85,
              label=f"BiLSTM IQR  F1={d['fdi_lstm']['F1']:.3f}")
    ax_f.set_yticks([0, 1])
    ax_f.set_yticklabels(["Normal", "Attack"])
    ax_f.set_xlabel("Sample Index")
    ax_f.set_title("(f) FDI Detection Labels — Ground Truth vs NARX vs Attention-BiLSTM",
                   **title_kw)
    ax_f.legend(fontsize=9, loc="upper right")
    ax_f.grid(True, alpha=0.3)

    # ── Super-title ───────────────────────────────────────────────
    fig.suptitle(
        "NARX  vs  Attention-BiLSTM — Full Model Comparison\n"
        "Caltech ACN-Data  |  EV Charging FDI Attack Detection",
        fontsize=14, fontweight="bold", color="#ddeeff", y=0.975,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"\n[DONE] Comparison plot saved → {save_path}")
    # Reset rcParams to defaults so other scripts aren't affected
    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────
# 4. Metrics text report
# ─────────────────────────────────────────────────────────────────
def save_metrics_report(d: dict, path: str):
    lines = [
        "═" * 60,
        "  NARX vs Attention-BiLSTM — Comparison Report",
        "═" * 60,
        "",
        "── Regression (Estimation Set) ──────────────────────",
        f"{'Metric':<10} {'NARX':>14} {'Attention-BiLSTM':>18}  Delta",
        "─" * 60,
    ]
    for key in ["MSE", "RMSE", "MAE", "R2"]:
        n_v = d["reg_narx"][key]
        l_v = d["reg_lstm"][key]
        delta = l_v - n_v
        sign  = "+" if delta >= 0 else ""
        lines.append(f"{key:<10} {n_v:>14.6e} {l_v:>18.6e}  {sign}{delta:.6e}")

    lines += [
        "",
        "── FDI Detection (IQR, k=5, q=5 on 10% injected attacks) ─",
        f"{'Metric':<12} {'NARX':>10} {'BiLSTM':>12}  Delta",
        "─" * 60,
    ]
    for key in ["F1", "Precision", "Recall", "Accuracy"]:
        n_v = d["fdi_narx"][key]
        l_v = d["fdi_lstm"][key]
        delta = l_v - n_v
        sign  = "+" if delta >= 0 else ""
        lines.append(f"{key:<12} {n_v:>10.4f} {l_v:>12.4f}  {sign}{delta:.4f}")

    lines += [
        "",
        "── Confusion Matrices ────────────────────────────────",
        "NARX:",
        f"  TP={d['fdi_narx']['TP']}  FP={d['fdi_narx']['FP']}  "
        f"FN={d['fdi_narx']['FN']}  TN={d['fdi_narx']['TN']}",
        "Attention-BiLSTM:",
        f"  TP={d['fdi_lstm']['TP']}  FP={d['fdi_lstm']['FP']}  "
        f"FN={d['fdi_lstm']['FN']}  TN={d['fdi_lstm']['TN']}",
        "",
        "═" * 60,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[DONE] Metrics report → {path}")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")

    if not Path(os.path.join(CKPT_DIR, "narx_best.pt")).exists():
        print("[ERROR] narx_best.pt not found. Run `python -m src.train.train` first.")
        sys.exit(1)
    if not Path(os.path.join(CKPT_DIR, "bilstm_best.pt")).exists():
        print("[ERROR] bilstm_best.pt not found. Run `python -m src.train.train_bilstm` first.")
        sys.exit(1)

    df_train = pd.read_csv(train_csv)
    df_estim = pd.read_csv(estim_csv)
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        for df in [df_train, df_estim]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

    print("\n[INFO] Running model comparison...")
    results = run_comparison(df_train, df_estim)

    plot_path   = os.path.join(OUT_DIR, "model_comparison.png")
    report_path = os.path.join(OUT_DIR, "model_comparison_metrics.txt")

    plot_comparison(results, save_path=plot_path)
    save_metrics_report(results, path=report_path)
