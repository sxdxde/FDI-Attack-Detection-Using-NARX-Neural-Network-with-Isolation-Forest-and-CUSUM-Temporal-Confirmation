"""
Two-Stage FDI Detector: Isolation Forest → CUSUM  (Attention-BiLSTM variant)
─────────────────────────────────────────────────────────────────
Identical pipeline to cusum_if.py but loads AttentionBiLSTM.
Both models use the same per-site IF + CUSUM evaluation so results
are directly comparable.

Usage (from narx_ev_fdi/ project root):
    python -m src.eval.cusum_if_bilstm
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.dataset            import build_datasets
from src.models.attention_bilstm import AttentionBiLSTM
from src.eval.cusum_if           import evaluate_if_cusum, plot_cusum_if

CKPT_DIR = os.path.join(ROOT, "checkpoints")
OUT_DIR  = os.path.join(ROOT, "results")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN     = 4
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3


if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    ckpt_path = os.path.join(CKPT_DIR, "bilstm_best.pt")
    scl_path  = os.path.join(CKPT_DIR, "bilstm_scalers.pkl")

    df_train = pd.read_csv(os.path.join(PROC_DIR, "acn_train_clean.csv"))
    df_estim = pd.read_csv(os.path.join(PROC_DIR, "acn_estim_clean.csv"))
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        for df in [df_train, df_estim]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

    data = build_datasets(df_train, df_estim,
                          val_ratio=0.15, test_ratio=0.15,
                          batch_size=256,
                          model_type="bilstm",
                          seq_len=SEQ_LEN,
                          max_train_sessions=10000, max_estim_sessions=5000)

    with open(scl_path, "rb") as f:
        scalers = pickle.load(f)
    scaler_y = scalers["y"]

    shapes = data["shapes"]
    n_feat = shapes.get("n_features", 7)

    model = AttentionBiLSTM(
        n_features=n_feat,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded BiLSTM from {ckpt_path}")

    # BiLSTM uses 3-D sequence input but evaluate_if_cusum calls model(X_tr)
    # where X_tr is the flat NARX window array — we need the bilstm raw arrays.
    # Re-extract the bilstm raw windows from the returned data dict.
    out = evaluate_if_cusum(model, data["raw"], scaler_y, val_fraction=0.15,
                            df_train=df_train, df_estim=df_estim)

    plot_cusum_if(
        out,
        save_path=os.path.join(OUT_DIR, "cusum_if_bilstm_detection.png"),
        title=(
            "Two-Stage FDI Detector: Isolation Forest + CUSUM\n"
            "ACN-Data-Static  |  Attention-BiLSTM Error-of-Estimation Pipeline"
        ),
    )

    r = out["results"]["combined"]
    print("\n" + "═" * 52)
    print("  ATTENTION-BiLSTM: COMBINED IF+CUSUM RESULT")
    print("═" * 52)
    print(f"  Accuracy  : {r['acc']:.4f}")
    print(f"  Precision : {r['prec']:.4f}")
    print(f"  Recall    : {r['rec']:.4f}")
    print(f"  F1 Score  : {r['f1']:.4f}")
    print("═" * 52)
