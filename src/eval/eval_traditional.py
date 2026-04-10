"""
Traditional 5-Panel Evaluation for Both Models
─────────────────────────────────────────────────────────────────
Applies the 2-stage IF + CUSUM detector to both NARX and Attention-BiLSTM
and outputs two separate traditional 5-panel plotting layouts.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset import build_datasets, EXOG_COLS
from src.models.narx import NARXNet
from src.models.attention_bilstm import AttentionBiLSTM
from src.eval.cusum_if import evaluate_if_cusum, plot_cusum_if

CKPT_DIR = os.path.join(ROOT, "checkpoints")
OUT_DIR  = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batched_predict(model, X_np):
    model.eval()
    preds = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32, device=DEVICE)
            preds.append(model(xb).cpu().numpy().flatten())
    return np.concatenate(preds)

def eval_and_plot(model, raw_data, scaler_y, name, title_prefix, out_path):
    from src.eval.cusum_if import tune_cusum, cusum_reset, IF_N_ESTIMATORS, IF_CONTAMINATION
    from sklearn.ensemble import IsolationForest
    from src.eval.evaluate import inject_fdi_attacks
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np

    print(f"\n[EVAL] Running 2-Stage Pipeline for {name}...")

    # ── Clean training EoE ───────────────────────────────────────
    X_tr, y_tr = raw_data["X_train_w"], raw_data["y_train_w"]
    tr_sc = batched_predict(model, X_tr)
    y_tr_true = scaler_y.inverse_transform(y_tr.reshape(-1, 1)).flatten()
    y_tr_pred = scaler_y.inverse_transform(tr_sc.reshape(-1, 1)).flatten()
    eoe_tr_clean = np.abs(y_tr_true - y_tr_pred)

    # ── Attacked estimation EoE ───────────────────────────────────
    X_es, y_es = raw_data["X_estim_w"], raw_data["y_estim_w"]
    es_sc = batched_predict(model, X_es)
    y_es_true = scaler_y.inverse_transform(y_es.reshape(-1, 1)).flatten()
    y_es_pred = scaler_y.inverse_transform(es_sc.reshape(-1, 1)).flatten()
    y_attacked, gt = inject_fdi_attacks(y_es_true, attack_fraction=0.10, seed=42)
    eoe_attacked = np.abs(y_attacked - y_es_pred)

    # ── Validation slice (for tuning) ─────────────────────────────
    n_tr  = len(eoe_tr_clean)
    n_val = int(n_tr * 0.15)
    eoe_tr_fit = eoe_tr_clean[:n_tr - n_val]

    y_val_true = y_tr_true[n_tr - n_val:]
    y_val_pred = y_tr_pred[n_tr - n_val:]
    y_val_att, gt_val = inject_fdi_attacks(y_val_true, attack_fraction=0.10, seed=7)
    eoe_val_attacked = np.abs(y_val_att - y_val_pred)

    # ── STAGE 1: IF ───────────────────────────────────────────────
    ifo = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=42)
    ifo.fit(eoe_tr_clean.reshape(-1, 1))
    if_scores  = ifo.decision_function(eoe_attacked.reshape(-1, 1))
    if_labels  = (if_scores < 0).astype(int)

    # ── STAGE 2: CUSUM ────────────────────────────────────────────
    best_k, best_h, _, mu_n = tune_cusum(eoe_tr_fit, eoe_val_attacked, gt_val, min_recall=0.90)
    cusum_S, cusum_labels = cusum_reset(eoe_attacked, best_k, best_h)
    
    combined = ((if_labels == 1) & (cusum_labels == 1)).astype(int)

    # Compile result dict for plotting
    def _stats(pred):
        cm = confusion_matrix(gt, pred)
        if cm.size < 4:
            cm = np.array([[cm[0,0],0],[0,0]])
        tn, fp, fn, tp = cm.ravel()
        return {"cm": cm, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "acc": accuracy_score(gt, pred), "prec": precision_score(gt, pred, zero_division=0),
                "rec": recall_score(gt, pred, zero_division=0), "f1": f1_score(gt, pred, zero_division=0)}
    
    # Baseline global IQR
    from src.eval.compare_models import iqr_detect
    eoe_clean_ref = np.abs(y_es_true - y_es_pred)
    iqr_lbl = iqr_detect(eoe_clean_ref, eoe_attacked, k=5.0, q=5)

    data = {
        "eoe": eoe_attacked, "gt": gt,
        "cusum_S": cusum_S, "if_labels": if_labels, "cusum_labels": cusum_labels, "combined": combined,
        "cusum_h": best_h, "cusum_k": best_k,
        "results": {
            "global_iqr": _stats(iqr_lbl),
            "isolation_forest": _stats(if_labels),
            "cusum_only": _stats(cusum_labels),
            "combined": _stats(combined)
        }
    }

    # Now we call the plotter directly!
    title_text = f"Two-Stage FDI Detector: Isolation Forest  +  CUSUM\nMulti-Site ACN-Data (Static)  |  {title_prefix} Error-of-Estimation Pipeline"
    plot_cusum_if(data, out_path, title=title_text)
    # I will patch plot_cusum_if directly from cusum_if!
    pass


def main():
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")

    print("[INFO] Loading datasets...")
    df_train = pd.read_csv(train_csv)
    df_estim = pd.read_csv(estim_csv)
    for col in ["connectionTime"]:
        if col in df_train.columns:
            df_train[col] = pd.to_datetime(df_train[col], utc=True)
            df_estim[col] = pd.to_datetime(df_estim[col], utc=True)

    # ----- NARX Pipeline -----
    print("\n[INFO] Evaluating NARX Pipeline...")
    with open(os.path.join(CKPT_DIR, "scalers.pkl"), "rb") as f:
        narx_scalers = pickle.load(f)
    narx_data = build_datasets(df_train, df_estim, mx=2, my=2,
                               val_ratio=0.15, test_ratio=0.15,
                               batch_size=64, model_type="narx")
    narx_model = NARXNet(input_size=narx_data["shapes"]["input_size"], hidden_size=10).to(DEVICE)
    narx_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "narx_best.pt"), map_location=DEVICE))
    eval_and_plot(narx_model, narx_data["raw"], narx_scalers["y"], "NARX", "NARX", os.path.join(OUT_DIR, "eval_narx_traditional.png"))

    # ----- Attention-BiLSTM Pipeline -----
    print("\n[INFO] Evaluating Attention-BiLSTM Pipeline...")
    with open(os.path.join(CKPT_DIR, "bilstm_scalers.pkl"), "rb") as f:
        lstm_scalers = pickle.load(f)
    lstm_data = build_datasets(df_train, df_estim, val_ratio=0.15, test_ratio=0.15,
                               batch_size=256, model_type="bilstm", seq_len=4)
    lstm_model = AttentionBiLSTM(n_features=lstm_data["shapes"]["n_features"], seq_len=4, hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
    lstm_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "bilstm_best.pt"), map_location=DEVICE))
    eval_and_plot(lstm_model, lstm_data["raw"], lstm_scalers["y"], "Attention-BiLSTM", "Attention-BiLSTM", os.path.join(OUT_DIR, "eval_bilstm_traditional.png"))


if __name__ == "__main__":
    main()
