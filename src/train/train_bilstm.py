"""
Attention-BiLSTM Training Script
─────────────────────────────────────────────────────────────────
Usage:
    python -m src.train.train_bilstm
    (run from the narx_ev_fdi/ project root)

Reads:
    data/processed/acn_train_clean.csv
    data/processed/acn_estim_clean.csv

Saves:
    checkpoints/bilstm_best.pt   ← best val-loss model weights
    checkpoints/bilstm_scalers.pkl
"""

import os
import sys
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd
from src.data.dataset       import build_datasets
from src.models.attention_bilstm import AttentionBiLSTM

# ─────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 4       # rolling window of 4 past timesteps
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
LEARNING_RATE = 1e-3
EPOCHS       = 30
BATCH_SIZE   = 256
PATIENCE     = 5
CKPT_DIR     = os.path.join(ROOT, "checkpoints")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def mse_inverse(pred: torch.Tensor, target: torch.Tensor, scaler_y) -> float:
    p = scaler_y.inverse_transform(pred.cpu().numpy())
    t = scaler_y.inverse_transform(target.cpu().numpy())
    return float(np.mean((p - t) ** 2))


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)   # (batch, seq_len, n_features)
            y_batch = y_batch.to(DEVICE)   # (batch, 1)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — important for LSTM stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            n          += len(y_batch)

    return total_loss / n


def evaluate_mse(model, loader, scaler_y):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            pred = model(X_b.to(DEVICE))
            all_pred.append(pred.cpu())
            all_true.append(y_b)
    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    return mse_inverse(pred_cat, true_cat, scaler_y)


# ─────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────
def train(df_train: pd.DataFrame, df_estim: pd.DataFrame):
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Build sequence datasets (3-D windows)
    data = build_datasets(
        df_train, df_estim,
        val_ratio=0.15, test_ratio=0.15,
        batch_size=BATCH_SIZE,
        model_type="bilstm",
        seq_len=SEQ_LEN,
        max_train_sessions=10000,
        max_estim_sessions=5000,
    )
    loaders  = data["loaders"]
    scaler_y = data["scalers"]["y"]
    shapes   = data["shapes"]    # {"seq_len": ..., "n_features": ...}
    n_feat   = shapes.get("n_features", 7)

    # 2. Model, loss, optimiser, scheduler
    model     = AttentionBiLSTM(
        n_features=n_feat,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=8)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Device      : {DEVICE}")
    print(f"[INFO] Model       : AttentionBiLSTM  seq_len={SEQ_LEN}  "
          f"n_features={n_feat}  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}")
    print(f"[INFO] Parameters  : {n_params:,}\n")

    # 3. Training loop with early stopping
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss   = run_epoch(model, loaders["val"],   criterion)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, "bilstm_best.pt"))
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>4d}/{EPOCHS}  "
                  f"train={train_loss:.6e}  val={val_loss:.6e}  "
                  f"EarlyStop={patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"\n[INFO] Training finished in {elapsed:.1f}s")

    # 4. Load best weights and report MSE
    model.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "bilstm_best.pt"), map_location=DEVICE)
    )

    mse_train = evaluate_mse(model, loaders["train"], scaler_y)
    mse_val   = evaluate_mse(model, loaders["val"],   scaler_y)
    mse_test  = evaluate_mse(model, loaders["test"],  scaler_y)
    mse_estim = evaluate_mse(model, loaders["estim"], scaler_y)

    print("\n══════════════════════════════════════════")
    print("  Attention-BiLSTM Performance (MSE, original scale, kWh/timestamp)")
    print("══════════════════════════════════════════")
    print(f"  Train  MSE : {mse_train:.6e}")
    print(f"  Val    MSE : {mse_val:.6e}")
    print(f"  Test   MSE : {mse_test:.6e}")
    print(f"  Estim  MSE : {mse_estim:.6e}")
    print("══════════════════════════════════════════\n")

    # 5. Save scalers
    scl_path = os.path.join(CKPT_DIR, "bilstm_scalers.pkl")
    with open(scl_path, "wb") as f:
        pickle.dump(data["scalers"], f)
    print(f"[DONE] Best model : {os.path.join(CKPT_DIR, 'bilstm_best.pt')}")
    print(f"[DONE] Scalers    : {scl_path}")

    return model, data


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR  = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")

    if not os.path.exists(train_csv):
        print(
            "[ERROR] Preprocessed CSVs not found.\n"
            "  Run `python src/data/preprocess.py` first."
        )
        sys.exit(1)

    df_train = pd.read_csv(train_csv)
    df_estim = pd.read_csv(estim_csv)
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        if col in df_train.columns:
            df_train[col] = pd.to_datetime(df_train[col], utc=True)
            df_estim[col] = pd.to_datetime(df_estim[col], utc=True)

    train(df_train, df_estim)
