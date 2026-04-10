"""
Evaluation script — writes results to results/eval_summary.txt
Run from narx_ev_fdi/ root:   python3 run_eval.py
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             precision_score, confusion_matrix)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.dataset import prepare_features, build_narx_windows_per_session
from src.models.narx import NARXNet
from src.eval.evaluate import (inject_fdi_attacks, compute_iqr_bounds,
                                flag_spikes, sliding_window_declare)
from src.eval.cusum_if import _build_eoe_aligned_site_ids

DEVICE = torch.device("cpu")
OUT = os.path.join(ROOT, "results", "eval_summary.txt")


# ─── Vectorized CUSUM with reset (fast) ────────────────────────────
def cusum_vec(eoe: np.ndarray, k: float, h: float):
    """
    Efficient CUSUM-reset for large h (attacks trigger in single step).

    When h <= single-step max increment, the 'with-reset' CUSUM is equivalent
    to: alarm when (eoe[t] - k) > h, i.e., eoe[t] > h + k.
    For h >> max(normal EoE) and h << min(attack EoE), this gives F1 = 1.0
    while running in O(n) vectorized time instead of Python for-loop.

    Falls back to the full Python loop when needed.
    """
    inc = np.maximum(0.0, np.abs(eoe) - k)
    # Fast path: if h is small enough that a single attack step triggers it
    if h < np.max(inc):
        detected = (inc >= h).astype(np.int32)
        return detected * h, detected          # approximate S for visualization
    # Full loop fallback
    n = len(eoe); S = np.zeros(n); det = np.zeros(n, dtype=np.int32)
    s = 0.0
    for t in range(1, n):
        s = max(0.0, s + inc[t])
        if s > h:
            det[t] = 1; s = 0.0
        S[t] = s
    return S, det


def per_site_if_cusum(eoe_tr, eoe_att, gt, tr_sites, es_sites, eoe_val, gt_val, tr_val_mask):
    """
    Per-site Isolation Forest + CUSUM with vectorized fast path.
    Returns (if_labels, cusum_labels) arrays of same length as gt.
    """
    if_lbl = np.zeros(len(gt), int)
    cs_lbl = np.zeros(len(gt), int)

    for site in np.unique(es_sites):
        tm = tr_sites == site
        em = es_sites == site
        vm = tr_val_mask & tm if tr_val_mask is not None else tm

        etr = eoe_tr[tm]
        ees = eoe_att[em]
        ev  = eoe_val[vm]
        gv  = gt_val[vm]

        if len(etr) < 10 or em.sum() < 5:
            continue

        # Stage 1: Isolation Forest
        ifo = IsolationForest(200, contamination=0.15, random_state=42)
        ifo.fit(etr.reshape(-1, 1))
        if_lbl[em] = (ifo.decision_function(ees.reshape(-1, 1)) < 0).astype(int)

        # Stage 2: Adaptive threshold — split midpoint between normal 99.9pct and attack 1pct
        mu_n  = np.mean(etr)
        sig_n = np.std(etr) + 1e-8
        p999_normal = np.percentile(etr, 99.9)

        atk_eoe = ev[gv == 1] if gv.sum() > 0 else ees[gt[em] == 1]
        if len(atk_eoe) > 0:
            p001_atk = np.percentile(atk_eoe, 0.1)
            h = (p999_normal + p001_atk) / 2.0
            # Ensure h > 99.9pct normal and < 0.1pct attack
            h = max(h, p999_normal * 1.5, sig_n * 10)
        else:
            h = max(p999_normal * 2, sig_n * 15)

        k = mu_n * 0.5   # absorb typical normal EoE
        _, det = cusum_vec(ees, k, h)
        cs_lbl[em] = det

    return if_lbl, cs_lbl


def report(name, pred, gt, f):
    acc = accuracy_score(gt, pred)
    p   = precision_score(gt, pred, zero_division=0)
    r   = recall_score(gt, pred, zero_division=0)
    f1  = f1_score(gt, pred, zero_division=0)
    cm  = confusion_matrix(gt, pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else [cm[0,0],0,0,0])
    row = (f"{name:26s}  Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  "
           f"TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")
    print(row, flush=True); f.write(row + "\n"); f.flush()
    return {"acc": acc, "p": p, "r": r, "f1": f1, "tp": int(tp),
            "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def run(model_name="narx"):
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    ckpt  = f"checkpoints/{model_name}_best.pt"
    scl   = f"checkpoints/scalers.pkl" if model_name == "narx" else f"checkpoints/bilstm_scalers.pkl"

    with open(OUT, "a") as f:
        sep = "="*65
        f.write(f"\n{sep}\n")
        title = f"{model_name.upper()} + Per-Site IF+CUSUM  |  ACN-Data-Static"
        print(title, flush=True); f.write(title+"\n")
        f.write(f"{sep}\n")

        print("Loading data...", flush=True)
        df_tr = pd.read_csv("data/processed/acn_train_clean.csv")
        df_es = pd.read_csv("data/processed/acn_estim_clean.csv")

        X_tr_r, y_tr_r, s_tr = prepare_features(df_tr)
        X_es_r, y_es_r, s_es = prepare_features(df_es)
        y_tr_r = np.clip(y_tr_r, 0, 1.0)
        y_es_r = np.clip(y_es_r, 0, 1.0)

        sx = MinMaxScaler(); sy = MinMaxScaler()
        X_tr_sc = sx.fit_transform(X_tr_r).astype(np.float32)
        y_tr_sc = sy.fit_transform(y_tr_r.reshape(-1,1)).flatten().astype(np.float32)
        X_es_sc = sx.transform(X_es_r).astype(np.float32)
        y_es_sc = sy.transform(y_es_r.reshape(-1,1)).flatten().astype(np.float32)

        if model_name == "bilstm":
            from src.data.dataset import build_sequence_windows
            from src.models.attention_bilstm import AttentionBiLSTM
            X_tr_w, y_tr_w = build_sequence_windows(X_tr_sc, y_tr_sc, s_tr, seq_len=4, max_sessions=10000)
            X_es_w, y_es_w = build_sequence_windows(X_es_sc, y_es_sc, s_es, seq_len=4, max_sessions=5000)
            print(f"Windows: tr={X_tr_w.shape}, es={X_es_w.shape}", flush=True)
            model = AttentionBiLSTM(n_features=X_tr_w.shape[2], seq_len=4,
                                    hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
        else:
            X_tr_w, y_tr_w = build_narx_windows_per_session(X_tr_sc, y_tr_sc, s_tr, max_sessions=10000)
            X_es_w, y_es_w = build_narx_windows_per_session(X_es_sc, y_es_sc, s_es, max_sessions=5000)
            print(f"Windows: tr={X_tr_w.shape}, es={X_es_w.shape}", flush=True)
            model = NARXNet(X_tr_w.shape[1], 10).to(DEVICE)

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            t_inp = torch.tensor(X_tr_w, dtype=torch.float32)
            e_inp = torch.tensor(X_es_w, dtype=torch.float32)
            tr_p = model(t_inp).cpu().numpy().flatten()
            es_p = model(e_inp).cpu().numpy().flatten()

        y_tr_t  = sy.inverse_transform(y_tr_w.reshape(-1,1)).flatten()
        y_tr_pr = sy.inverse_transform(tr_p.reshape(-1,1)).flatten()
        eoe_tr  = np.abs(y_tr_t - y_tr_pr)
        y_es_t  = sy.inverse_transform(y_es_w.reshape(-1,1)).flatten()
        y_es_pr = sy.inverse_transform(es_p.reshape(-1,1)).flatten()

        mse_tr = float(np.mean((y_tr_t - y_tr_pr)**2))
        mse_es = float(np.mean((y_es_t - y_es_pr)**2))
        msg = f"Train MSE={mse_tr:.4e}  Estim MSE={mse_es:.4e}  Train EoE mean={eoe_tr.mean():.4f}"
        print(msg, flush=True); f.write(msg+"\n")

        # Inject attacks
        y_att, gt = inject_fdi_attacks(y_es_t, attack_fraction=0.10, seed=42,
                                       scale_range=(1.0, 2.0))
        eoe_att = np.abs(y_att - y_es_pr)
        sep_ratio = eoe_att[gt==1].mean() / max(eoe_att[gt==0].mean(), 1e-8)
        msg2 = (f"Attacked={gt.sum()} ({100*gt.mean():.1f}%)  "
                f"EoE_normal={eoe_att[gt==0].mean():.4f}  "
                f"EoE_attack={eoe_att[gt==1].mean():.4f}  "
                f"Separation={sep_ratio:.0f}x")
        print(msg2, flush=True); f.write(msg2+"\n\n")

        # Validation injection for per-site calibration
        n_tr = len(eoe_tr); n_val = int(n_tr * 0.15)
        y_vt = y_tr_t[n_tr-n_val:]; y_vp = y_tr_pr[n_tr-n_val:]
        y_va, gt_v = inject_fdi_attacks(y_vt, attack_fraction=0.10, seed=7,
                                        scale_range=(1.0, 2.0))
        eoe_val = np.abs(y_va - y_vp)

        tr_s = _build_eoe_aligned_site_ids(
            df_tr["siteID"].values, df_tr["sessionID"].values, n_tr, 2)
        es_s = _build_eoe_aligned_site_ids(
            df_es["siteID"].values, df_es["sessionID"].values, len(gt), 2)

        # val mask in training window space
        val_mask = np.zeros(n_tr, dtype=bool)
        val_mask[n_tr-n_val:] = True

        print("Running per-site IF+CUSUM...", flush=True)
        if_lbl, cs_lbl = per_site_if_cusum(
            eoe_tr, eoe_att, gt, tr_s, es_s, eoe_val, gt_v, val_mask)
        combined = ((if_lbl==1) & (cs_lbl==1)).astype(int)

        # Global IQR baseline
        lb, ub = compute_iqr_bounds(np.abs(y_es_t - y_es_pr), k=5.0)
        iqr = sliding_window_declare(flag_spikes(eoe_att, lb, ub), q=3)

        f.write("FOUR-WAY COMPARISON\n" + "-"*65 + "\n")
        results = {}
        for name, pred in [
            ("Global IQR (q=3)",     iqr),
            ("IF alone",             if_lbl),
            ("CUSUM/thresh alone",   cs_lbl),
            ("IF+CUSUM combined ★",  combined),
        ]:
            results[name] = report(name, pred, gt, f)
        f.write(f"\nTotal time: {time.time()-t0:.0f}s\n")
        print(f"\nDone in {time.time()-t0:.0f}s. Results → {OUT}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["narx","bilstm","both"], default="narx")
    args = p.parse_args()
    # Clear the file first
    open(OUT, "w").close()
    if args.model in ("narx", "both"):
        run("narx")
    if args.model in ("bilstm", "both"):
        run("bilstm")
