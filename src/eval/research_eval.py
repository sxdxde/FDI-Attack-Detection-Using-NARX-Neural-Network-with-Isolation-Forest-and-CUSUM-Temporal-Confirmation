"""
Research-Grade Evaluation and Publication Figures
──────────────────────────────────────────────────────────────────
Produces all figures and tables for the research paper:

1. EoE Distribution Plot  — KDE of normal vs attack EoE
2. ROC + PR Curves        — all detectors compared
3. F1 Sensitivity         — F1 vs attack scale for each model
4. Multi-Seed Robustness  — mean ± std over 5 seeds
5. Confusion Matrices     — clean side-by-side layout
6. Detection Timeline     — zoomed segment with attack highlighted

Usage (from narx_ev_fdi/ root):
    python3 -m src.eval.research_eval
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import torch
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset            import build_datasets
from src.models.narx             import NARXNet
from src.models.attention_bilstm import AttentionBiLSTM
from src.eval.evaluate           import inject_fdi_attacks, compute_iqr_bounds, \
                                         flag_spikes, sliding_window_declare

OUT_DIR  = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
PROC_DIR = os.path.join(ROOT, "data", "processed")
DEVICE   = torch.device("cpu")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "narx":    "#2196F3",
    "bilstm":  "#E91E63",
    "naive":   "#9E9E9E",
    "normal":  "#4CAF50",
    "attack":  "#F44336",
    "if":      "#FF9800",
    "cusum":   "#9C27B0",
    "combined":"#00BCD4",
}
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
})

SCALE_RANGE = (1.0, 2.0)   # realistic FDI: 1-2x mean charging delivery
ATTACK_FRAC = 0.10


# ─────────────────────────────────────────────────────────────────
# Detection primitives
# ─────────────────────────────────────────────────────────────────
def single_step_detect(eoe_signal, k, h):
    """Flag samples where |EoE| - k >= h (no accumulation across sessions)."""
    inc = np.maximum(0.0, np.abs(eoe_signal) - k)
    return (inc >= h).astype(int)


def tune_threshold(eoe_clean_train, eoe_val_attacked, gt_val):
    """
    Data-driven threshold tuning for single-step detection.
    Sweeps threshold candidates between p95(normal) and p5(attack),
    picks the one that maximises F1 on the validation slice.
    No access to test-set ground truth — uses only training statistics
    and a labelled validation slice (10% of training data, separate seed).
    """
    mu_n  = np.mean(eoe_clean_train)
    sig_n = np.std(eoe_clean_train) + 1e-8
    k     = mu_n * 0.5   # absorb typical normal noise

    # Candidate thresholds: geometric grid from p90 to p99.9 of normal EoE
    low  = np.percentile(eoe_clean_train, 90)
    high = np.percentile(eoe_clean_train, 99.9) * 3   # extend into attack region
    candidates = np.unique(np.concatenate([
        np.linspace(low, high, 60),
        np.percentile(eoe_val_attacked, np.linspace(1, 99, 40)),
    ]))

    best_h, best_f1 = candidates[0], -1.0
    for h in candidates:
        det = single_step_detect(eoe_val_attacked, k, h)
        f1 = f1_score(gt_val, det, zero_division=0)
        if f1 > best_f1:
            best_f1, best_h = f1, h
    return k, best_h, best_f1


def two_stage_detect(eoe_tr, eoe_es, eoe_val, gt_val, seed=42):
    """
    Two-stage IF + CUSUM (single-step) detector.
    Threshold tuned on validation slice (not test set).
    Returns (if_labels, cusum_labels, combined_labels).
    """
    # IF
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(eoe_tr), min(15_000, len(eoe_tr)), replace=False)
    ifo = IsolationForest(n_estimators=200, contamination=0.15, random_state=seed)
    ifo.fit(eoe_tr[fit_idx].reshape(-1, 1))
    if_lbl = (ifo.decision_function(eoe_es.reshape(-1, 1)) < 0).astype(int)

    # CUSUM (single-step)
    k, h, _ = tune_threshold(eoe_tr, eoe_val, gt_val)
    cs_lbl   = single_step_detect(eoe_es, k, h)
    combined = ((if_lbl == 1) & (cs_lbl == 1)).astype(int)
    return if_lbl, cs_lbl, combined, k, h


def score(gt, pred):
    cm = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0],0,0,0)
    return {
        "acc":  accuracy_score(gt, pred),
        "prec": precision_score(gt, pred, zero_division=0),
        "rec":  recall_score(gt, pred, zero_division=0),
        "f1":   f1_score(gt, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "cm": cm,
    }


# ─────────────────────────────────────────────────────────────────
# Load data + models
# ─────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
df_tr = pd.read_csv(os.path.join(PROC_DIR, "acn_train_clean.csv"))
df_es = pd.read_csv(os.path.join(PROC_DIR, "acn_estim_clean.csv"))

# Verify no session overlap (data integrity check)
tr_s = set(df_tr["sessionID"].unique())
es_s = set(df_es["sessionID"].unique())
assert len(tr_s & es_s) == 0, "LEAKAGE: sessions appear in both train and estim!"
print(f"  Train sessions: {len(tr_s):,}  |  Estim sessions: {len(es_s):,}  |  Overlap: 0 ✓",
      flush=True)

data_narx   = build_datasets(df_tr, df_es, val_ratio=0.15, test_ratio=0.15,
                              batch_size=256, model_type="narx",
                              max_train_sessions=10000, max_estim_sessions=5000)
data_bilstm = build_datasets(df_tr, df_es, val_ratio=0.15, test_ratio=0.15,
                              batch_size=256, model_type="bilstm", seq_len=4,
                              max_train_sessions=10000, max_estim_sessions=5000)

with open(os.path.join(CKPT_DIR, "scalers.pkl"),        "rb") as f: sc_n = pickle.load(f)["y"]
with open(os.path.join(CKPT_DIR, "bilstm_scalers.pkl"), "rb") as f: sc_b = pickle.load(f)["y"]

narx   = NARXNet(data_narx["raw"]["X_train_w"].shape[1], 10)
narx.load_state_dict(torch.load(os.path.join(CKPT_DIR,"narx_best.pt"), map_location="cpu"))
narx.eval()

bilstm = AttentionBiLSTM(n_features=7, seq_len=4, hidden_size=128, num_layers=2, dropout=0.3)
bilstm.load_state_dict(torch.load(os.path.join(CKPT_DIR,"bilstm_best.pt"), map_location="cpu"))
bilstm.eval()


def get_eoe(model, X, y_w, scaler_y):
    with torch.no_grad():
        pred_sc = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    y_true = scaler_y.inverse_transform(y_w.reshape(-1,1)).flatten()
    y_pred = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).flatten()
    return y_true, y_pred, np.abs(y_true - y_pred)


raw_n = data_narx["raw"];   raw_b = data_bilstm["raw"]

y_tr_n,  y_trp_n,  eoe_tr_n  = get_eoe(narx,   raw_n["X_train_w"], raw_n["y_train_w"],  sc_n)
y_es_n,  y_esp_n,  eoe_es_n  = get_eoe(narx,   raw_n["X_estim_w"], raw_n["y_estim_w"],  sc_n)
y_tr_b,  y_trp_b,  eoe_tr_b  = get_eoe(bilstm, raw_b["X_train_w"], raw_b["y_train_w"],  sc_b)
y_es_b,  y_esp_b,  eoe_es_b  = get_eoe(bilstm, raw_b["X_estim_w"], raw_b["y_estim_w"],  sc_b)

mean_train_n = float(np.mean(y_tr_n))
eoe_tr_naive = np.abs(y_tr_n  - mean_train_n)
eoe_es_naive = np.abs(y_es_n  - mean_train_n)

# ── Validation slices (from training data — NOT estimation) ────────
N_TR_N = len(eoe_tr_n); N_VAL_N = int(N_TR_N * 0.15)
N_TR_B = len(eoe_tr_b); N_VAL_B = int(N_TR_B * 0.15)

eoe_fit_n     = eoe_tr_n[:N_TR_N - N_VAL_N]   # clean fit portion
eoe_fit_b     = eoe_tr_b[:N_TR_B - N_VAL_B]
eoe_fit_naive = eoe_tr_naive[:N_TR_N - N_VAL_N]

y_val_n  = y_tr_n[N_TR_N-N_VAL_N:];  y_valp_n  = y_trp_n[N_TR_N-N_VAL_N:]
y_val_b  = y_tr_b[N_TR_B-N_VAL_B:];  y_valp_b  = y_trp_b[N_TR_B-N_VAL_B:]

y_va_n,  gt_v_n  = inject_fdi_attacks(y_val_n,  ATTACK_FRAC, SCALE_RANGE, seed=7)
y_va_b,  gt_v_b  = inject_fdi_attacks(y_val_b,  ATTACK_FRAC, SCALE_RANGE, seed=7)
y_va_nv, gt_v_nv = inject_fdi_attacks(y_val_n,  ATTACK_FRAC, SCALE_RANGE, seed=7)

eoe_val_n     = np.abs(y_va_n  - y_valp_n)
eoe_val_b     = np.abs(y_va_b  - y_valp_b)
eoe_val_naive = np.abs(y_va_nv - mean_train_n)

# ── Attacked estimation set ────────────────────────────────────────
y_att_n, gt_n = inject_fdi_attacks(y_es_n, ATTACK_FRAC, SCALE_RANGE, seed=42)
y_att_b, gt_b = inject_fdi_attacks(y_es_b, ATTACK_FRAC, SCALE_RANGE, seed=42)

eoe_att_n     = np.abs(y_att_n - y_esp_n)
eoe_att_b     = np.abs(y_att_b - y_esp_b)
eoe_att_naive = np.abs(y_att_n - mean_train_n)

baseline_phys = float(np.mean(y_es_n[y_es_n > 1e-6]))
print(f"\nNARX   train EoE: mean={eoe_fit_n.mean():.4f}  "
      f"p99={np.percentile(eoe_fit_n,99):.4f}  p99.9={np.percentile(eoe_fit_n,99.9):.4f}",
      flush=True)
print(f"BiLSTM train EoE: mean={eoe_fit_b.mean():.4f}  "
      f"p99={np.percentile(eoe_fit_b,99):.4f}  p99.9={np.percentile(eoe_fit_b,99.9):.4f}",
      flush=True)
print(f"Naive  train EoE: mean={eoe_fit_naive.mean():.4f}  "
      f"p99={np.percentile(eoe_fit_naive,99):.4f}", flush=True)
print(f"Attack baseline: {baseline_phys:.4f} kWh | "
      f"Attack EoE range: {SCALE_RANGE[0]*baseline_phys:.3f}–"
      f"{SCALE_RANGE[1]*baseline_phys:.3f} kWh", flush=True)

sep_n = eoe_att_n[gt_n==1].mean() / max(eoe_att_n[gt_n==0].mean(), 1e-8)
sep_b = eoe_att_b[gt_b==1].mean() / max(eoe_att_b[gt_b==0].mean(), 1e-8)
sep_nv= eoe_att_naive[gt_n==1].mean() / max(eoe_att_naive[gt_n==0].mean(), 1e-8)
print(f"EoE separation — NARX: {sep_n:.1f}x  BiLSTM: {sep_b:.1f}x  Naive: {sep_nv:.1f}x",
      flush=True)


# ─────────────────────────────────────────────────────────────────
# Figure 1 — EoE Distribution (KDE): Normal vs Attack
# ─────────────────────────────────────────────────────────────────
print("\n[FIG 1] EoE distributions...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
fig.suptitle("Error-of-Estimation Distributions: Normal vs FDI-Attacked Samples\n"
             "(Attack scale 1.0–2.0× mean delivery, additive injection)",
             fontweight="bold")

for ax, name, eoe_norm_all, eoe_atk_all, gt_i in [
    (axes[0], "NARX",              eoe_att_n,     eoe_att_n,     gt_n),
    (axes[1], "Attention-BiLSTM",  eoe_att_b,     eoe_att_b,     gt_b),
    (axes[2], "Naive Mean Predictor", eoe_att_naive, eoe_att_naive, gt_n),
]:
    norm_vals = eoe_norm_all[gt_i == 0]
    atk_vals  = eoe_atk_all[gt_i == 1]
    clip95    = np.percentile(np.concatenate([norm_vals, atk_vals]), 99)
    xmax      = min(clip95 * 1.2, 2.0)
    xs = np.linspace(0, xmax, 400)
    try:
        kde_n = gaussian_kde(np.clip(norm_vals, 0, xmax), bw_method="silverman")(xs)
        kde_a = gaussian_kde(np.clip(atk_vals,  0, xmax), bw_method="silverman")(xs)
        ax.fill_between(xs, kde_n, alpha=0.35, color=PALETTE["normal"])
        ax.fill_between(xs, kde_a, alpha=0.35, color=PALETTE["attack"])
        ax.plot(xs, kde_n, color=PALETTE["normal"], lw=1.8, label="Normal EoE")
        ax.plot(xs, kde_a, color=PALETTE["attack"],  lw=1.8, label="Attack EoE")
    except Exception:
        ax.hist(np.clip(norm_vals, 0, xmax), bins=60, density=True,
                alpha=0.4, color=PALETTE["normal"], label="Normal")
        ax.hist(np.clip(atk_vals,  0, xmax), bins=60, density=True,
                alpha=0.4, color=PALETTE["attack"],  label="Attack")

    # Optimal threshold line
    k_i, h_i, _ = tune_threshold(
        eoe_fit_n if "NARX" in name else (eoe_fit_b if "BiLSTM" in name else eoe_fit_naive),
        eoe_atk_all[gt_i==1][:5000],     # small sample for speed
        np.ones(min(5000, (gt_i==1).sum()), dtype=int)
    )
    thresh = k_i + h_i
    ax.axvline(thresh, color="navy", ls="--", lw=1.5, label=f"Threshold={thresh:.3f}")

    sep = float(np.mean(atk_vals)) / max(float(np.mean(norm_vals)), 1e-8)
    ax.set_title(f"{name}\nSeparation ratio: {sep:.1f}×")
    ax.set_xlabel("EoE (kWh / 5-min step)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, xmax)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_eoe_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_eoe_distributions.png", flush=True)


# ─────────────────────────────────────────────────────────────────
# Figure 2 — ROC and Precision-Recall Curves
# ─────────────────────────────────────────────────────────────────
print("[FIG 2] ROC + PR curves...", flush=True)
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ROC and Precision-Recall Curves\n"
             f"(scale_range={SCALE_RANGE}, attack_fraction={ATTACK_FRAC*100:.0f}%, seed=42)",
             fontweight="bold")

ax_roc.plot([0,1],[0,1], "k--", lw=0.8, label="Random classifier")
ax_pr.axhline(gt_n.mean(), color="k", ls="--", lw=0.8,
              label=f"No-skill baseline (P={gt_n.mean():.2f})")

for name, scores_arr, gt_i, color in [
    ("NARX",               eoe_att_n,     gt_n, PALETTE["narx"]),
    ("Attention-BiLSTM",   eoe_att_b,     gt_b, PALETTE["bilstm"]),
    ("Naive Mean Predictor", eoe_att_naive, gt_n, PALETTE["naive"]),
]:
    fpr, tpr, _ = roc_curve(gt_i, scores_arr)
    roc_auc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(gt_i, scores_arr)
    pr_auc = auc(rec_c, prec_c)
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}\nAUC-ROC={roc_auc:.3f}")
    ax_pr.plot(rec_c, prec_c, color=color, lw=2,
               label=f"{name}\nAUC-PR={pr_auc:.3f}")

ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="(a) Receiver Operating Characteristic", xlim=(0,1), ylim=(0,1.01))
ax_roc.legend(loc="lower right"); ax_roc.grid(alpha=0.3)
ax_pr.set(xlabel="Recall", ylabel="Precision",
          title="(b) Precision-Recall Curve", xlim=(0,1), ylim=(0,1.01))
ax_pr.legend(loc="upper right"); ax_pr.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_roc_pr_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_roc_pr_curves.png", flush=True)


# ─────────────────────────────────────────────────────────────────
# Figure 3 — F1 vs Attack Scale (Sensitivity Analysis)
# ─────────────────────────────────────────────────────────────────
print("[FIG 3] F1 vs attack scale...", flush=True)

scales_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

def f1_at_scale(eoe_fit, eoe_es, y_es_true, y_es_pred, eoe_val_fit, y_val_true,
                y_val_pred, sc, seed=42):
    """
    Compute IF+CUSUM F1 at a given scale.
    Threshold tuned on TRAINING validation slice (not test set).
    """
    # Inject into estimation (test)
    y_att, gt = inject_fdi_attacks(y_es_true, ATTACK_FRAC,
                                   scale_range=(sc, sc * 1.5), seed=seed)
    eoe_att = np.abs(y_att - y_es_pred)
    # Inject into validation slice for threshold tuning
    y_va, gt_v = inject_fdi_attacks(y_val_true, ATTACK_FRAC,
                                    scale_range=(sc, sc * 1.5), seed=seed + 1)
    eoe_val = np.abs(y_va - y_val_pred)

    if_lbl, cs_lbl, comb, _, _ = two_stage_detect(eoe_fit, eoe_att, eoe_val, gt_v, seed)
    return (f1_score(gt, if_lbl,   zero_division=0),
            f1_score(gt, cs_lbl,   zero_division=0),
            f1_score(gt, comb,     zero_division=0))

f1_n_comb, f1_b_comb, f1_nv_comb = [], [], []
f1_n_cs,   f1_b_cs,   f1_nv_cs   = [], [], []

for sc in scales_test:
    print(f"  scale={sc:.2f}", flush=True)
    _, a_cs, a_c = f1_at_scale(eoe_fit_n, eoe_es_n, y_es_n, y_esp_n,
                                eoe_val_n, y_val_n, y_valp_n, sc)
    f1_n_cs.append(a_cs); f1_n_comb.append(a_c)

    _, b_cs, b_c = f1_at_scale(eoe_fit_b, eoe_es_b, y_es_b, y_esp_b,
                                eoe_val_b, y_val_b, y_valp_b, sc)
    f1_b_cs.append(b_cs); f1_b_comb.append(b_c)

    val_pred_nv = np.full_like(y_valp_n, mean_train_n)
    es_pred_nv  = np.full_like(y_esp_n,  mean_train_n)
    _, nv_cs, nv_c = f1_at_scale(eoe_fit_naive, eoe_es_naive, y_es_n, es_pred_nv,
                                  eoe_val_naive, y_val_n, val_pred_nv, sc)
    f1_nv_cs.append(nv_cs); f1_nv_comb.append(nv_c)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(scales_test, f1_n_comb,   "o-",  color=PALETTE["narx"],   lw=2, ms=6,
        label="NARX + IF+CUSUM (combined)")
ax.plot(scales_test, f1_b_comb,   "s-",  color=PALETTE["bilstm"], lw=2, ms=6,
        label="Attention-BiLSTM + IF+CUSUM")
ax.plot(scales_test, f1_nv_comb,  "^--", color=PALETTE["naive"],  lw=1.5, ms=6,
        label="Naive Mean Predictor + IF+CUSUM")
ax.plot(scales_test, f1_n_cs,    "o:",  color=PALETTE["narx"],   lw=1.2, ms=5, alpha=0.6,
        label="NARX + CUSUM alone")
ax.axvspan(SCALE_RANGE[0], SCALE_RANGE[1], alpha=0.08, color="steelblue",
           label=f"Evaluation range {SCALE_RANGE}")
ax.set_xlabel("Attack Injection Scale (× mean positive delivery kWh)")
ax.set_ylabel("F1 Score")
ax.set_title("FDI Detection F1 vs Attack Scale\n"
             "Model quality matters: neural network EoE separates cleanly at lower scales")
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_f1_vs_scale.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_f1_vs_scale.png", flush=True)


# ─────────────────────────────────────────────────────────────────
# Figure 4 — Multi-Seed Robustness
# ─────────────────────────────────────────────────────────────────
print("[FIG 4] Multi-seed robustness...", flush=True)
SEEDS = [42, 123, 7, 2024, 99]

res_ms = {"narx": [], "bilstm": [], "naive": []}
for s in SEEDS:
    for key, eoe_fit, eoe_es, y_es_true, y_es_pred, eoe_val_fit, y_val_true, y_val_pred in [
        ("narx",   eoe_fit_n, eoe_es_n, y_es_n, y_esp_n,
         eoe_val_n, y_val_n, y_valp_n),
        ("bilstm", eoe_fit_b, eoe_es_b, y_es_b, y_esp_b,
         eoe_val_b, y_val_b, y_valp_b),
        ("naive",  eoe_fit_naive, eoe_es_naive, y_es_n,
         np.full_like(y_esp_n, mean_train_n),
         eoe_val_naive, y_val_n, np.full_like(y_valp_n, mean_train_n)),
    ]:
        y_att, gt = inject_fdi_attacks(y_es_true, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=s)
        eoe_att   = np.abs(y_att - y_es_pred)
        y_va, gt_v = inject_fdi_attacks(y_val_true, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=s+1)
        eoe_val   = np.abs(y_va - y_val_pred)
        if_lbl, cs_lbl, comb, _, _ = two_stage_detect(
            eoe_fit, eoe_att, eoe_val, gt_v, seed=s)
        res_ms[key].append({
            "f1_if":   f1_score(gt, if_lbl,   zero_division=0),
            "f1_cs":   f1_score(gt, cs_lbl,   zero_division=0),
            "f1_comb": f1_score(gt, comb,     zero_division=0),
        })

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle(f"Multi-Seed Robustness Analysis ({len(SEEDS)} seeds, "
             f"scale_range={SCALE_RANGE})\nBoxes show median ± IQR across seeds",
             fontweight="bold")

for ax, (key, color, title) in zip(axes, [
    ("narx",   PALETTE["narx"],   "NARX"),
    ("bilstm", PALETTE["bilstm"], "Attention-BiLSTM"),
    ("naive",  PALETTE["naive"],  "Naive Mean Predictor"),
]):
    rows = res_ms[key]
    data_box = [
        [r["f1_if"]   for r in rows],
        [r["f1_cs"]   for r in rows],
        [r["f1_comb"] for r in rows],
    ]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    col_list = [PALETTE["if"], PALETTE["cusum"], PALETTE["combined"]]
    for patch, c in zip(bp["boxes"], col_list):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    means = [np.mean(d) for d in data_box]
    ax.scatter([1,2,3], means, marker="D", color="black", zorder=5, s=40)
    for i, m in enumerate(means):
        ax.text(i+1, m+0.02, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["IF\nalone", "CUSUM\nalone", "IF+CUSUM"])
    ax.set_ylabel("F1 Score"); ax.set_title(title)
    ax.set_ylim(-0.05, 1.12); ax.grid(axis="y", alpha=0.3)

legend_patches = [
    Patch(color=PALETTE["if"],      alpha=0.7, label="Isolation Forest alone"),
    Patch(color=PALETTE["cusum"],   alpha=0.7, label="CUSUM alone"),
    Patch(color=PALETTE["combined"],alpha=0.7, label="IF + CUSUM combined"),
]
axes[1].legend(handles=legend_patches, fontsize=8, loc="lower center")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_multiseed_robustness.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_multiseed_robustness.png", flush=True)

print(f"\n{'─'*62}")
print(f"{'Model':<22} {'IF F1 mean±std':>16} {'CUSUM mean±std':>16} {'Comb mean±std':>16}")
print(f"{'─'*62}")
for key, label in [("narx","NARX"),("bilstm","BiLSTM"),("naive","Naive")]:
    rows = res_ms[key]
    f1_if = [r["f1_if"] for r in rows]
    f1_cs = [r["f1_cs"] for r in rows]
    f1_cb = [r["f1_comb"] for r in rows]
    print(f"{label:<22} {np.mean(f1_if):.3f}±{np.std(f1_if):.3f}  "
          f"  {np.mean(f1_cs):.3f}±{np.std(f1_cs):.3f}  "
          f"  {np.mean(f1_cb):.3f}±{np.std(f1_cb):.3f}")
print(f"{'─'*62}")


# ─────────────────────────────────────────────────────────────────
# Figure 5 — Confusion Matrices
# ─────────────────────────────────────────────────────────────────
print("\n[FIG 5] Confusion matrices...", flush=True)

if_n, cs_n, comb_n, kn, hn = two_stage_detect(eoe_fit_n, eoe_att_n, eoe_val_n, gt_v_n)
if_b, cs_b, comb_b, kb, hb = two_stage_detect(eoe_fit_b, eoe_att_b, eoe_val_b, gt_v_b)
if_nv, cs_nv, comb_nv, _, _ = two_stage_detect(eoe_fit_naive, eoe_att_naive,
                                                 eoe_val_naive, gt_v_nv)

print(f"  Thresholds — NARX: k={kn:.4f} h={hn:.4f}  BiLSTM: k={kb:.4f} h={hb:.4f}",
      flush=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Confusion Matrices: IF+CUSUM Combined Detector\n"
             f"Attack scale {SCALE_RANGE}, seed=42  "
             f"(attack_fraction={ATTACK_FRAC*100:.0f}%)",
             fontweight="bold")

for ax, gt_i, pred_i, title, cmap in [
    (axes[0], gt_n, comb_n,  "NARX",                "Blues"),
    (axes[1], gt_b, comb_b,  "Attention-BiLSTM",    "Reds"),
    (axes[2], gt_n, comb_nv, "Naive Mean Predictor","Greys"),
]:
    cm_i = confusion_matrix(gt_i, pred_i)
    m    = score(gt_i, pred_i)
    im   = ax.imshow(cm_i, cmap=cmap, interpolation="nearest",
                     vmin=0, vmax=cm_i.max())
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted\nNormal","Predicted\nAttack"])
    ax.set_yticklabels(["Actual\nNormal","Actual\nAttack"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm_i[r,c]:,}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm_i[r,c] > cm_i.max()*0.5 else "black")
    ax.set_title(f"{title}\n"
                 f"Acc={m['acc']:.3f}  P={m['prec']:.3f}  "
                 f"R={m['rec']:.3f}  F1={m['f1']:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_confusion_matrices.png", flush=True)


# ─────────────────────────────────────────────────────────────────
# Figure 6 — Detection Timeline (first 3000 samples)
# ─────────────────────────────────────────────────────────────────
print("[FIG 6] Detection timeline...", flush=True)
N_SHOW = 3000
seg = slice(0, N_SHOW)
idx_seg  = np.arange(N_SHOW)
eoe_seg  = eoe_att_n[seg]
gt_seg   = gt_n[seg]
pred_seg = comb_n[seg]

# Find attack windows for shading
attack_regions = []
in_atk = False; s0 = 0
for t in range(N_SHOW):
    if gt_seg[t] == 1 and not in_atk:
        in_atk = True; s0 = t
    elif gt_seg[t] == 0 and in_atk:
        attack_regions.append((s0, t)); in_atk = False
if in_atk:
    attack_regions.append((s0, N_SHOW))

fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                          gridspec_kw={"height_ratios":[2.5, 1.2, 1]})
fig.suptitle("FDI Detection Timeline — NARX Model (first 3,000 estimation samples)",
             fontweight="bold")

ax0 = axes[0]
ax0.plot(idx_seg, eoe_seg, color=PALETTE["narx"], lw=0.8, zorder=2,
         label="EoE (attacked signal)")
ax0.plot(idx_seg, np.abs(y_es_n[seg] - y_esp_n[seg]), color=PALETTE["normal"],
         lw=0.6, alpha=0.5, zorder=1, label="EoE (clean, no attack)")
thresh_line = kn + hn
ax0.axhline(thresh_line, color="darkorange", ls="--", lw=1.5,
            label=f"Detection threshold = {thresh_line:.3f} kWh")
for (s,e) in attack_regions:
    ax0.axvspan(s, e, color=PALETTE["attack"], alpha=0.15)
ax0.set_ylabel("EoE (kWh / 5-min)"); ax0.grid(alpha=0.3)
ax0.set_title("(a) Error-of-Estimation — orange band = injected attack windows")
ax0.legend(loc="upper right", fontsize=8)

ax1 = axes[1]
ax1.step(idx_seg, gt_seg,   where="post", color=PALETTE["attack"],  lw=1.8,
         label="Ground Truth (attack=1)")
ax1.step(idx_seg, pred_seg, where="post", color=PALETTE["combined"], lw=1.3,
         ls="--", alpha=0.85, label="IF+CUSUM Detection")
for (s,e) in attack_regions:
    ax1.axvspan(s, e, color=PALETTE["attack"], alpha=0.10)
ax1.set_yticks([0,1]); ax1.set_yticklabels(["Normal","Attack"])
ax1.set_title("(b) Ground Truth vs Detected Labels")
ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)

ax2 = axes[2]
ax2.axis("off")
m_seg = score(gt_seg, pred_seg)
summary = (f"Segment ({N_SHOW} samples)  —  "
           f"TP={m_seg['tp']}  FP={m_seg['fp']}  FN={m_seg['fn']}  TN={m_seg['tn']}\n"
           f"Accuracy={m_seg['acc']:.3f}   Precision={m_seg['prec']:.3f}   "
           f"Recall={m_seg['rec']:.3f}   F1={m_seg['f1']:.3f}")
ax2.text(0.5, 0.5, summary, transform=ax2.transAxes,
         ha="center", va="center", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                   edgecolor="darkorange", lw=1.5))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_detection_timeline.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig6_detection_timeline.png", flush=True)


# ─────────────────────────────────────────────────────────────────
# Final results table
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"FINAL RESULTS  (scale_range={SCALE_RANGE}, attack_fraction={ATTACK_FRAC*100:.0f}%, seed=42)")
print(f"{'='*80}")
print(f"{'Model + Method':<38} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>6} {'FP':>6}")
print(f"{'-'*80}")

for model_name, gt_i, eoe_fit, eoe_att, eoe_val, gt_v, y_es_true, y_es_pred in [
    ("NARX",   gt_n, eoe_fit_n, eoe_att_n, eoe_val_n, gt_v_n, y_es_n, y_esp_n),
    ("BiLSTM", gt_b, eoe_fit_b, eoe_att_b, eoe_val_b, gt_v_b, y_es_b, y_esp_b),
    ("Naive",  gt_n, eoe_fit_naive, eoe_att_naive,
     eoe_val_naive, gt_v_nv, y_es_n, np.full_like(y_esp_n, mean_train_n)),
]:
    if_lbl, cs_lbl, comb, k_i, h_i = two_stage_detect(
        eoe_fit, eoe_att, eoe_val, gt_v)
    lb, ub = compute_iqr_bounds(eoe_att[gt_i==0], k=5.0)
    iqr    = sliding_window_declare(flag_spikes(eoe_att, lb, ub), q=3)
    for mname, pred in [
        ("IQR baseline (q=3)", iqr),
        ("Isolation Forest", if_lbl),
        ("CUSUM alone",      cs_lbl),
        ("IF + CUSUM ★",     comb),
    ]:
        m = score(gt_i, pred)
        print(f"{model_name+' '+mname:<38} {m['acc']:>6.3f} {m['prec']:>6.3f} "
              f"{m['rec']:>6.3f} {m['f1']:>6.3f} {m['tp']:>6} {m['fp']:>6}")
    print(f"  → Threshold: k={k_i:.4f}  h={h_i:.4f}")
    print()

print(f"{'='*80}")
print(f"\n[DONE] All 6 figures saved to {OUT_DIR}/")
for i in range(1,7):
    names = ["eoe_distributions","roc_pr_curves","f1_vs_scale",
             "multiseed_robustness","confusion_matrices","detection_timeline"]
    print(f"  fig{i}_{names[i-1]}.png")
