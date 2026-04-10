"""
Model Comparison Collage — "comparison.png"
────────────────────────────────────────────────────────────────
8-panel collage comparing NARX, Attention-BiLSTM, and Naive baseline:

  (a) Grouped metric bars         (b) ROC curves
  (c) Precision-Recall curves     (d) F1 vs Attack Scale
  (e) EoE KDE distributions       (f) Multi-seed F1 box plots
  (g) TP / FP / FN breakdown      (h) Model performance radar

Usage:
    python3 -m src.eval.comparison
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.stats import gaussian_kde
import torch
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    f1_score, accuracy_score, recall_score,
    precision_score, confusion_matrix,
)
from sklearn.ensemble import IsolationForest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset            import build_datasets
from src.models.narx             import NARXNet
from src.models.attention_bilstm import AttentionBiLSTM
from src.eval.evaluate           import (inject_fdi_attacks, compute_iqr_bounds,
                                          flag_spikes, sliding_window_declare)

OUT_DIR  = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
PROC_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────
C_NARX   = "#1565C0"   # deep blue
C_BILSTM = "#C62828"   # deep red
C_NAIVE  = "#546E7A"   # blue-grey
C_NOR    = "#2E7D32"   # dark green  (normal EoE)
C_ATK    = "#B71C1C"   # dark red    (attack EoE)
C_IF     = "#E65100"   # orange
C_CS     = "#6A1B9A"   # purple
C_CB     = "#00838F"   # teal

ALPHA_FILL = 0.22
SCALE_RANGE = (1.0, 2.0)
ATTACK_FRAC = 0.10
SEEDS = [42, 123, 7, 2024, 99]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def get_eoe(model, X, y_w, sy):
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    yt = sy.inverse_transform(y_w.reshape(-1,1)).flatten()
    yp = sy.inverse_transform(p.reshape(-1,1)).flatten()
    return yt, yp, np.abs(yt - yp)

def single_step(eoe, k, h):
    return (np.maximum(0.0, np.abs(eoe) - k) >= h).astype(int)

def tune_h(eoe_clean, eoe_val_att, gt_v):
    k    = np.mean(eoe_clean) * 0.5
    low  = np.percentile(eoe_clean, 90)
    high = np.percentile(eoe_clean, 99.9) * 3
    cands = np.unique(np.concatenate([
        np.linspace(low, high, 80),
        np.percentile(eoe_val_att, np.linspace(1, 99, 40)),
    ]))
    best_h, best_f1 = cands[0], -1.0
    for h in cands:
        f1 = f1_score(gt_v, single_step(eoe_val_att, k, h), zero_division=0)
        if f1 > best_f1:
            best_f1, best_h = f1, h
    return k, best_h

def two_stage(eoe_fit, eoe_test, eoe_val, gt_v, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eoe_fit), min(15_000, len(eoe_fit)), replace=False)
    ifo = IsolationForest(n_estimators=200, contamination=0.15, random_state=seed)
    ifo.fit(eoe_fit[idx].reshape(-1,1))
    if_lbl = (ifo.decision_function(eoe_test.reshape(-1,1)) < 0).astype(int)
    k, h   = tune_h(eoe_fit, eoe_val, gt_v)
    cs_lbl = single_step(eoe_test, k, h)
    comb   = ((if_lbl==1) & (cs_lbl==1)).astype(int)
    return if_lbl, cs_lbl, comb, k, h

def metrics(gt, pred):
    cm = confusion_matrix(gt, pred)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (cm[0,0],0,0,0)
    return dict(acc=accuracy_score(gt,pred),
                prec=precision_score(gt,pred,zero_division=0),
                rec=recall_score(gt,pred,zero_division=0),
                f1=f1_score(gt,pred,zero_division=0),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ─────────────────────────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────────────────────────
print("Loading data & models...", flush=True)
df_tr = pd.read_csv(os.path.join(PROC_DIR,"acn_train_clean.csv"))
df_es = pd.read_csv(os.path.join(PROC_DIR,"acn_estim_clean.csv"))

dn = build_datasets(df_tr, df_es, val_ratio=0.15, test_ratio=0.15,
                    batch_size=256, model_type="narx",
                    max_train_sessions=10000, max_estim_sessions=5000)
db = build_datasets(df_tr, df_es, val_ratio=0.15, test_ratio=0.15,
                    batch_size=256, model_type="bilstm", seq_len=4,
                    max_train_sessions=10000, max_estim_sessions=5000)

with open(os.path.join(CKPT_DIR,"scalers.pkl"),       "rb") as f: sn = pickle.load(f)["y"]
with open(os.path.join(CKPT_DIR,"bilstm_scalers.pkl"),"rb") as f: sb = pickle.load(f)["y"]

narx = NARXNet(dn["raw"]["X_train_w"].shape[1], 10)
narx.load_state_dict(torch.load(os.path.join(CKPT_DIR,"narx_best.pt"),map_location="cpu"))
narx.eval()

bilstm = AttentionBiLSTM(n_features=7,seq_len=4,hidden_size=128,num_layers=2,dropout=0.3)
bilstm.load_state_dict(torch.load(os.path.join(CKPT_DIR,"bilstm_best.pt"),map_location="cpu"))
bilstm.eval()

rn = dn["raw"]; rb = db["raw"]
ytn, ypn, en = get_eoe(narx,   rn["X_train_w"], rn["y_train_w"], sn)
yen, ypne, een= get_eoe(narx,   rn["X_estim_w"], rn["y_estim_w"], sn)
ytb, ypb, eb = get_eoe(bilstm, rb["X_train_w"], rb["y_train_w"], sb)
yeb, ypbe, eeb= get_eoe(bilstm, rb["X_estim_w"], rb["y_estim_w"], sb)

mean_tr = float(np.mean(ytn))
etn_nv = np.abs(ytn - mean_tr); een_nv = np.abs(yen - mean_tr)

Ntrn=len(en); Nvaln=int(Ntrn*0.15)
Ntrb=len(eb); Nvalb=int(Ntrb*0.15)

efit_n = en[:Ntrn-Nvaln]; efit_b = eb[:Ntrb-Nvalb]
efit_nv= etn_nv[:Ntrn-Nvaln]

yval_n=ytn[Ntrn-Nvaln:]; yvp_n=ypn[Ntrn-Nvaln:]
yval_b=ytb[Ntrb-Nvalb:]; yvp_b=ypb[Ntrb-Nvalb:]

yva_n, gtv_n = inject_fdi_attacks(yval_n, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=7)
yva_b, gtv_b = inject_fdi_attacks(yval_b, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=7)
yva_nv,gtv_nv= inject_fdi_attacks(yval_n, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=7)

eval_n = np.abs(yva_n  - yvp_n)
eval_b = np.abs(yva_b  - yvp_b)
eval_nv= np.abs(yva_nv - mean_tr)

yatt_n, gt_n = inject_fdi_attacks(yen, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=42)
yatt_b, gt_b = inject_fdi_attacks(yeb, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=42)

eatt_n = np.abs(yatt_n - ypne)
eatt_b = np.abs(yatt_b - ypbe)
eatt_nv= np.abs(yatt_n - mean_tr)

print("Running two-stage detection...", flush=True)
IF_n, CS_n, CB_n, kn, hn = two_stage(efit_n, eatt_n, eval_n, gtv_n)
IF_b, CS_b, CB_b, kb, hb = two_stage(efit_b, eatt_b, eval_b, gtv_b)
IF_nv,CS_nv,CB_nv,knv,hnv= two_stage(efit_nv,eatt_nv,eval_nv,gtv_nv)

m_n  = metrics(gt_n, CB_n)
m_b  = metrics(gt_b, CB_b)
m_nv = metrics(gt_n, CB_nv)

print(f"  NARX   F1={m_n['f1']:.3f}  P={m_n['prec']:.3f}  R={m_n['rec']:.3f}",flush=True)
print(f"  BiLSTM F1={m_b['f1']:.3f}  P={m_b['prec']:.3f}  R={m_b['rec']:.3f}",flush=True)
print(f"  Naive  F1={m_nv['f1']:.3f}  P={m_nv['prec']:.3f}  R={m_nv['rec']:.3f}",flush=True)

# ── Multi-seed ────────────────────────────────────────────────────
print("Multi-seed evaluation...", flush=True)
ms = {"narx": [], "bilstm": [], "naive": []}
for s in SEEDS:
    for key, ef, ye_t, ye_p, yv_t, yv_p in [
        ("narx",   efit_n, yen, ypne, yval_n, yvp_n),
        ("bilstm", efit_b, yeb, ypbe, yval_b, yvp_b),
        ("naive",  efit_nv, yen,
         np.full_like(ypne, mean_tr), yval_n,
         np.full_like(yvp_n, mean_tr)),
    ]:
        ya,gt  = inject_fdi_attacks(ye_t, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=s)
        ea     = np.abs(ya - ye_p)
        yv,gv  = inject_fdi_attacks(yv_t, ATTACK_FRAC, scale_range=SCALE_RANGE, seed=s+1)
        ev     = np.abs(yv - yv_p)
        _,cs,cb,_,_ = two_stage(ef, ea, ev, gv, seed=s)
        ms[key].append(dict(
            f1_cs=f1_score(gt,cs,zero_division=0),
            f1_cb=f1_score(gt,cb,zero_division=0),
        ))

# ── Scale sensitivity ─────────────────────────────────────────────
print("Scale sensitivity...", flush=True)
scales = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
sc_f1 = {"narx":[], "bilstm":[], "naive":[]}
for sc in scales:
    for key, ef, ye_t, ye_p, yv_t, yv_p in [
        ("narx",   efit_n, yen, ypne, yval_n, yvp_n),
        ("bilstm", efit_b, yeb, ypbe, yval_b, yvp_b),
        ("naive",  efit_nv, yen,
         np.full_like(ypne,mean_tr), yval_n,
         np.full_like(yvp_n,mean_tr)),
    ]:
        ya,gt = inject_fdi_attacks(ye_t,ATTACK_FRAC,scale_range=(sc,sc*1.5),seed=42)
        ea    = np.abs(ya - ye_p)
        yv,gv = inject_fdi_attacks(yv_t,ATTACK_FRAC,scale_range=(sc,sc*1.5),seed=7)
        ev    = np.abs(yv - yv_p)
        _,_,cb,_,_ = two_stage(ef,ea,ev,gv)
        sc_f1[key].append(f1_score(gt,cb,zero_division=0))


# ═════════════════════════════════════════════════════════════════
# BUILD COLLAGE
# ═════════════════════════════════════════════════════════════════
print("Building comparison collage...", flush=True)

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#FAFAFA")

gs = gridspec.GridSpec(
    4, 4,
    figure=fig,
    hspace=0.52, wspace=0.42,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.04,
)

ax_bar  = fig.add_subplot(gs[0, :2])   # (a) metric bars
ax_roc  = fig.add_subplot(gs[0, 2])    # (b) ROC
ax_pr   = fig.add_subplot(gs[0, 3])    # (c) PR

ax_kde  = fig.add_subplot(gs[1, :2])   # (d) EoE KDE
ax_sc   = fig.add_subplot(gs[1, 2])    # (e) F1 vs scale
ax_box  = fig.add_subplot(gs[1, 3])    # (f) multi-seed boxes

ax_break= fig.add_subplot(gs[2, :2])   # (g) TP/FP/FN breakdown
ax_sep  = fig.add_subplot(gs[2, 2])    # (h) EoE separation bars
ax_rad  = fig.add_subplot(gs[2, 3], projection="polar")  # (i) radar

ax_summ = fig.add_subplot(gs[3, :])    # (j) summary table

# ── Main title ────────────────────────────────────────────────────
fig.suptitle(
    "FDI Attack Detection — Model Comparison: NARX  vs  Attention-BiLSTM  vs  Naive Baseline\n"
    "ACN-Data-Static  |  Two-Stage IF+CUSUM Detector  |  "
    f"Attack scale {SCALE_RANGE[0]}–{SCALE_RANGE[1]}× mean delivery  |  10% attack fraction",
    fontsize=13, fontweight="bold", y=0.965,
)

MODEL_LABELS = ["NARX", "Attention-\nBiLSTM", "Naive\nBaseline"]
COLORS       = [C_NARX, C_BILSTM, C_NAIVE]

# ──────────────────────────────────────────────────────────────────
# (a) Grouped metric bars
# ──────────────────────────────────────────────────────────────────
metric_keys = ["acc","prec","rec","f1"]
metric_names= ["Accuracy","Precision","Recall","F1 Score"]
vals = np.array([[m_n[k], m_b[k], m_nv[k]] for k in metric_keys])   # (4,3)

x  = np.arange(len(metric_keys))
bw = 0.22
for i,(col,lbl) in enumerate(zip(COLORS, MODEL_LABELS)):
    bars = ax_bar.bar(x + (i-1)*bw, vals[:,i], bw,
                      color=col, alpha=0.85, label=lbl.replace("\n"," "),
                      edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals[:,i]):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=col)

ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_names)
ax_bar.set_ylim(0, 1.13)
ax_bar.set_ylabel("Score")
ax_bar.set_title("(a)  Detection Performance Metrics (IF+CUSUM Combined)")
ax_bar.legend(loc="upper left", framealpha=0.85)
ax_bar.grid(axis="y", alpha=0.25)
ax_bar.axhline(0.9, color="grey", ls=":", lw=0.8)
ax_bar.text(3.35, 0.91, "0.90", fontsize=7.5, color="grey")

# ──────────────────────────────────────────────────────────────────
# (b) ROC curves
# ──────────────────────────────────────────────────────────────────
ax_roc.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
for score_arr, gt_i, col, lbl in [
    (eatt_n, gt_n, C_NARX,   "NARX"),
    (eatt_b, gt_b, C_BILSTM, "BiLSTM"),
    (eatt_nv,gt_n, C_NAIVE,  "Naive"),
]:
    fpr,tpr,_ = roc_curve(gt_i, score_arr)
    ax_roc.plot(fpr,tpr,color=col,lw=2,label=f"{lbl} (AUC={auc(fpr,tpr):.3f})")
ax_roc.set(xlabel="FPR", ylabel="TPR",
           title="(b)  ROC Curve", xlim=(0,1), ylim=(0,1.01))
ax_roc.legend(loc="lower right", fontsize=8); ax_roc.grid(alpha=0.25)

# ──────────────────────────────────────────────────────────────────
# (c) Precision-Recall
# ──────────────────────────────────────────────────────────────────
ax_pr.axhline(gt_n.mean(), color="grey", ls=":", lw=0.8)
for score_arr, gt_i, col, lbl in [
    (eatt_n, gt_n, C_NARX,   "NARX"),
    (eatt_b, gt_b, C_BILSTM, "BiLSTM"),
    (eatt_nv,gt_n, C_NAIVE,  "Naive"),
]:
    pr,rc,_ = precision_recall_curve(gt_i, score_arr)
    ax_pr.plot(rc,pr,color=col,lw=2,label=f"{lbl} (AUC={auc(rc,pr):.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision",
          title="(c)  Precision-Recall Curve", xlim=(0,1), ylim=(0,1.01))
ax_pr.legend(loc="upper right", fontsize=8); ax_pr.grid(alpha=0.25)

# ──────────────────────────────────────────────────────────────────
# (d) EoE KDE comparison (all 3 models, normal & attack overlaid)
# ──────────────────────────────────────────────────────────────────
ax_kde.set_title("(d)  EoE Distributions — Normal vs Attack (all models)")
xmax = 1.2
xs = np.linspace(0, xmax, 500)
linestyles = ["-","--","-."]
for eoe_all, gt_i, col, lbl, ls in [
    (eatt_n,  gt_n, C_NARX,   "NARX",   "-"),
    (eatt_b,  gt_b, C_BILSTM, "BiLSTM", "--"),
    (eatt_nv, gt_n, C_NAIVE,  "Naive",  "-."),
]:
    nv = np.clip(eoe_all[gt_i==0], 0, xmax)
    av = np.clip(eoe_all[gt_i==1], 0, xmax)
    try:
        kn_  = gaussian_kde(nv, bw_method="silverman")(xs)
        ka_  = gaussian_kde(av, bw_method="silverman")(xs)
        ax_kde.plot(xs, kn_, color=col, lw=1.8, ls=ls, alpha=0.9,
                    label=f"{lbl} — normal")
        ax_kde.fill_between(xs, kn_, alpha=ALPHA_FILL, color=col)
        ax_kde.plot(xs, ka_, color=col, lw=1.8, ls=ls, alpha=0.55,
                    label=f"{lbl} — attack")
    except Exception:
        pass

# Vertical threshold markers
for k_i, h_i, col, lbl in [
    (kn, hn, C_NARX,   "NARX θ"),
    (kb, hb, C_BILSTM, "BiLSTM θ"),
]:
    ax_kde.axvline(k_i+h_i, color=col, ls=":", lw=1.5, alpha=0.7)
    ax_kde.text(k_i+h_i+0.01, ax_kde.get_ylim()[1]*0.9 if ax_kde.get_ylim()[1]>0 else 5,
                lbl, color=col, fontsize=7.5, va="top")

ax_kde.set_xlabel("EoE (kWh / 5-min step)")
ax_kde.set_ylabel("Density")
ax_kde.set_xlim(0, xmax)
ax_kde.legend(ncol=2, fontsize=7.5, loc="upper right")
ax_kde.grid(alpha=0.25)

legend_lines = [
    plt.Line2D([0],[0],color="black",lw=1.8,ls="-",  label="NARX"),
    plt.Line2D([0],[0],color="black",lw=1.8,ls="--", label="BiLSTM"),
    plt.Line2D([0],[0],color="black",lw=1.8,ls="-.", label="Naive"),
    Patch(color=C_NOR,alpha=0.5,label="Normal fill"),
    Patch(color=C_ATK,alpha=0.25,label="Attack fill (lighter)"),
]
ax_kde.legend(handles=legend_lines, fontsize=7.5, loc="upper right", ncol=2)

# ──────────────────────────────────────────────────────────────────
# (e) F1 vs Attack Scale
# ──────────────────────────────────────────────────────────────────
ax_sc.plot(scales, sc_f1["narx"],   "o-",  color=C_NARX,   lw=2, ms=5,
           label="NARX")
ax_sc.plot(scales, sc_f1["bilstm"], "s-",  color=C_BILSTM, lw=2, ms=5,
           label="BiLSTM")
ax_sc.plot(scales, sc_f1["naive"],  "^--", color=C_NAIVE,  lw=1.5, ms=5,
           label="Naive")
ax_sc.axvspan(SCALE_RANGE[0], SCALE_RANGE[1], alpha=0.08,
              color="steelblue", label="Eval range")
ax_sc.set(xlabel="Attack Scale (× mean delivery)",
          ylabel="F1 Score",
          title="(e)  F1 vs Attack Scale",
          ylim=(0,1.05), xlim=(min(scales)-0.1, max(scales)+0.1))
ax_sc.legend(fontsize=8); ax_sc.grid(alpha=0.25)

# Annotation: NARX advantage at low scale
diff_low = sc_f1["narx"][3] - sc_f1["naive"][3]
ax_sc.annotate(f"NARX−Naive\n+{diff_low:.2f} @ scale=1",
               xy=(1.0, sc_f1["narx"][3]),
               xytext=(1.4, sc_f1["narx"][3]-0.15),
               arrowprops=dict(arrowstyle="->", color="grey", lw=1),
               fontsize=7.5, color="grey")

# ──────────────────────────────────────────────────────────────────
# (f) Multi-seed F1 box plots — grouped bars by model
# ──────────────────────────────────────────────────────────────────
all_cb = {k: [r["f1_cb"] for r in ms[k]] for k in ["narx","bilstm","naive"]}
all_cs = {k: [r["f1_cs"] for r in ms[k]] for k in ["narx","bilstm","naive"]}

positions = [1, 1.5, 2,   3, 3.5, 4]
box_data  = [all_cb["narx"], all_cb["bilstm"], all_cb["naive"],
             all_cs["narx"], all_cs["bilstm"], all_cs["naive"]]
box_cols  = [C_NARX, C_BILSTM, C_NAIVE,
             C_NARX, C_BILSTM, C_NAIVE]
bps = ax_box.boxplot(box_data, positions=positions, patch_artist=True,
                     widths=0.35, medianprops=dict(color="black",lw=2))
for patch, col in zip(bps["boxes"], box_cols):
    patch.set_facecolor(col); patch.set_alpha(0.7)

for d, pos, col in zip(box_data, positions, box_cols):
    ax_box.scatter([pos]*len(d), d, color=col, s=25, zorder=5, alpha=0.8)
    ax_box.text(pos, np.mean(d)-0.055, f"{np.mean(d):.3f}",
                ha="center", fontsize=7.5, fontweight="bold", color=col)

ax_box.set_xticks([1.5, 3.5])
ax_box.set_xticklabels(["IF+CUSUM\nCombined", "CUSUM\nAlone"], fontsize=9)
ax_box.set_ylabel("F1 Score"); ax_box.set_ylim(0, 1.12)
ax_box.set_title(f"(f)  Multi-Seed F1 ({len(SEEDS)} seeds)")
ax_box.grid(axis="y", alpha=0.25)
ax_box.axvline(2.5, color="grey", ls=":", lw=0.8)
legend_patches = [Patch(color=C_NARX, alpha=0.7, label="NARX"),
                  Patch(color=C_BILSTM,alpha=0.7, label="BiLSTM"),
                  Patch(color=C_NAIVE, alpha=0.7, label="Naive")]
ax_box.legend(handles=legend_patches, fontsize=7.5, loc="lower right")

# ──────────────────────────────────────────────────────────────────
# (g) TP / FP / FN stacked horizontal bars
# ──────────────────────────────────────────────────────────────────
labels_g = ["NARX\nIF+CUSUM","BiLSTM\nIF+CUSUM","Naive\nIF+CUSUM"]
tps = [m_n["tp"],  m_b["tp"],  m_nv["tp"]]
fps = [m_n["fp"],  m_b["fp"],  m_nv["fp"]]
fns = [m_n["fn"],  m_b["fn"],  m_nv["fn"]]
tns = [m_n["tn"],  m_b["tn"],  m_nv["tn"]]
y_g = np.arange(len(labels_g))

ax_break.barh(y_g, tps, color="#2E7D32", alpha=0.85, label="True Positives (TP)")
ax_break.barh(y_g, fps, left=tps,  color="#E53935", alpha=0.7,  label="False Positives (FP)")
ax_break.barh(y_g, fns, left=[t+f for t,f in zip(tps,fps)],
              color="#FB8C00", alpha=0.7, label="False Negatives (FN)")

for i,(tp,fp,fn,tn) in enumerate(zip(tps,fps,fns,tns)):
    total = tp+fp+fn+tn
    ax_break.text(tp/2, i, f"TP\n{tp:,}", ha="center", va="center",
                  fontsize=8, color="white", fontweight="bold")
    if fp > 0:
        ax_break.text(tp + fp/2, i, f"FP\n{fp:,}", ha="center", va="center",
                      fontsize=8, color="white", fontweight="bold")
    if fn > 0:
        ax_break.text(tp+fp + fn/2, i, f"FN\n{fn:,}", ha="center", va="center",
                      fontsize=8, color="white", fontweight="bold")

ax_break.set_yticks(y_g); ax_break.set_yticklabels(labels_g)
ax_break.set_xlabel("Sample Count")
ax_break.set_title("(g)  TP / FP / FN Breakdown (IF+CUSUM Combined Detector)")
ax_break.legend(loc="lower right", fontsize=8); ax_break.grid(axis="x", alpha=0.2)

# ──────────────────────────────────────────────────────────────────
# (h) EoE separation bar chart
# ──────────────────────────────────────────────────────────────────
sep_vals = []
for eoe_all, gt_i in [(eatt_n,gt_n),(eatt_b,gt_b),(eatt_nv,gt_n)]:
    nm = eoe_all[gt_i==0].mean(); am = eoe_all[gt_i==1].mean()
    sep_vals.append(am / max(nm, 1e-8))

ax_sep.bar(MODEL_LABELS, sep_vals, color=COLORS, alpha=0.82, edgecolor="white")
for i, (lbl, v) in enumerate(zip(MODEL_LABELS, sep_vals)):
    ax_sep.text(i, v+0.4, f"{v:.1f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLORS[i])

ax_sep.set_ylabel("EoE Separation Ratio\n(mean attack / mean normal)")
ax_sep.set_title("(h)  EoE Separation by Model\n(higher = cleaner signal)")
ax_sep.set_ylim(0, max(sep_vals)*1.25)
ax_sep.grid(axis="y", alpha=0.25)
ax_sep.axhline(10, color="grey", ls=":", lw=0.8)
ax_sep.text(2.4, 10.2, "10× threshold", fontsize=7.5, color="grey")

# ──────────────────────────────────────────────────────────────────
# (i) Radar chart
# ──────────────────────────────────────────────────────────────────
radar_metrics = ["F1", "Precision", "Recall", "Accuracy", "Separation\n(norm.)"]
sep_norm = [min(v/max(sep_vals),1.0) for v in sep_vals]   # normalise to [0,1]

N_rad = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N_rad, endpoint=False).tolist()
angles += angles[:1]  # close

for model_vals, col, lbl in [
    ([m_n["f1"], m_n["prec"], m_n["rec"], m_n["acc"], sep_norm[0]],
     C_NARX, "NARX"),
    ([m_b["f1"], m_b["prec"], m_b["rec"], m_b["acc"], sep_norm[1]],
     C_BILSTM, "BiLSTM"),
    ([m_nv["f1"],m_nv["prec"],m_nv["rec"],m_nv["acc"],sep_norm[2]],
     C_NAIVE, "Naive"),
]:
    v = model_vals + model_vals[:1]
    ax_rad.plot(angles, v, color=col, lw=2, label=lbl)
    ax_rad.fill(angles, v, color=col, alpha=0.10)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(radar_metrics, fontsize=8.5)
ax_rad.set_ylim(0, 1.0)
ax_rad.set_yticks([0.25,0.5,0.75,1.0])
ax_rad.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=7)
ax_rad.set_title("(i)  Radar: Multi-Metric Profile", pad=15, fontsize=10.5)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8.5)
ax_rad.grid(alpha=0.3)

# ──────────────────────────────────────────────────────────────────
# (j) Summary table
# ──────────────────────────────────────────────────────────────────
ax_summ.axis("off")

col_labels = ["Model", "Detector", "Accuracy", "Precision", "Recall", "F1 Score",
              "TP", "FP", "FN",
              "EoE Sep.", "Multi-seed F1\n(mean ± std)"]
rows_data = []

for model_name, m_i, sep_i, ms_key, gt_i, efit, eatt, evl, gv, seed in [
    ("NARX",   m_n,  sep_vals[0], "narx",   gt_n, efit_n, eatt_n, eval_n,  gtv_n,  42),
    ("BiLSTM", m_b,  sep_vals[1], "bilstm", gt_b, efit_b, eatt_b, eval_b,  gtv_b,  42),
    ("Naive",  m_nv, sep_vals[2], "naive",  gt_n, efit_nv,eatt_nv,eval_nv, gtv_nv, 42),
]:
    # IQR row
    lb,ub = compute_iqr_bounds(eatt[gt_i==0], k=5.0)
    iqr   = sliding_window_declare(flag_spikes(eatt,lb,ub), q=3)
    mi    = metrics(gt_i, iqr)
    ms_cb = [r["f1_cb"] for r in ms[ms_key]]
    rows_data.append([model_name, "IQR baseline",
                      f"{mi['acc']:.3f}", f"{mi['prec']:.3f}",
                      f"{mi['rec']:.3f}", f"{mi['f1']:.3f}",
                      f"{mi['tp']:,}", f"{mi['fp']:,}", f"{mi['fn']:,}",
                      f"{sep_i:.1f}×", "—"])
    # IF+CUSUM combined row
    rows_data.append([model_name, "IF+CUSUM ★",
                      f"{m_i['acc']:.3f}", f"{m_i['prec']:.3f}",
                      f"{m_i['rec']:.3f}", f"{m_i['f1']:.3f}",
                      f"{m_i['tp']:,}", f"{m_i['fp']:,}", f"{m_i['fn']:,}",
                      f"{sep_i:.1f}×",
                      f"{np.mean(ms_cb):.3f} ± {np.std(ms_cb):.3f}"])
    rows_data.append(["", "", "", "", "", "", "", "", "", "", ""])  # spacer

table = ax_summ.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False); table.set_fontsize(8.8)

# Style header
for c in range(len(col_labels)):
    table[0, c].set_facecolor("#263238")
    table[0, c].set_text_props(color="white", fontweight="bold")

# Style data rows
narx_rows   = [1, 2]
bilstm_rows = [4, 5]
naive_rows  = [7, 8]
highlight_col = "#E3F2FD"
for ridx, col_bg in [(narx_rows, "#E3F2FD"), (bilstm_rows,"#FCE4EC"),
                      (naive_rows,"#ECEFF1")]:
    for r in ridx:
        for c in range(len(col_labels)):
            table[r+1, c].set_facecolor(col_bg)

# Bold the IF+CUSUM rows
for r in [2, 5, 8]:
    for c in range(len(col_labels)):
        table[r+1, c].set_text_props(fontweight="bold")

ax_summ.set_title("(j)  Complete Results Summary Table  (seed=42, scale_range=[1.0, 2.0])",
                  fontsize=10.5, pad=8, fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────
save_path = os.path.join(OUT_DIR, "comparison.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[DONE] Saved → {save_path}", flush=True)
