"""
NARX Training Script
─────────────────────────────────────────────────────────────────
Usage:
    python -m src.train.train
    (run from the narx_ev_fdi/ project root)

Reads:
    data/processed/acn_train_clean.csv
    data/processed/acn_estim_clean.csv

Saves:
    checkpoints/narx_best.pt   ← best val-loss model weights
    checkpoints/scalers.pkl    ← fitted MinMaxScalers
"""

import os
import sys
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ── resolve project root so we can import sibling packages ───────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd
from src.data.dataset import build_datasets, EXOG_COLS, TARGET_COL
from src.models.narx   import NARXNet

# ─────────────────────────────────────────────────────────────────
# Hyper-parameters (matching the paper)
# ─────────────────────────────────────────────────────────────────
MX           = 2        # exogenous input delay order
MY           = 2        # output feedback delay order
HIDDEN_SIZE  = 10       # neurons in the single hidden layer
LEARNING_RATE = 1e-3
EPOCHS       = 200
BATCH_SIZE   = 64
PATIENCE     = 20       # early-stopping patience (epochs without val-loss improvement)
CKPT_DIR     = os.path.join(ROOT, "checkpoints")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def mse_inverse(pred: torch.Tensor, target: torch.Tensor,
                scaler_y) -> float:
    """MSE in original (un-scaled) kWh/timestamp space."""
    p = scaler_y.inverse_transform(pred.cpu().numpy())
    t = scaler_y.inverse_transform(target.cpu().numpy())
    return float(np.mean((p - t) ** 2))


def run_epoch(model, loader, criterion, optimizer=None):
    """One training or evaluation epoch.  Returns mean loss (scaled space)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            n          += len(y_batch)

    return total_loss / n


def evaluate_mse(model, loader, scaler_y):
    """MSE in original (un-scaled) space over a full DataLoader."""
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

    # 1. Build datasets / loaders
    data = build_datasets(
        df_train, df_estim,
        mx=MX, my=MY,
        val_ratio=0.15, test_ratio=0.15,
        batch_size=BATCH_SIZE,
    )
    loaders   = data["loaders"]
    scaler_y  = data["scalers"]["y"]
    input_dim = data["shapes"]["input_size"]

    # 2. Model, loss, optimiser
    model     = NARXNet(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[INFO] Device : {DEVICE}")
    print(f"[INFO] Model  : NARXNet  input={input_dim}  hidden={HIDDEN_SIZE}")
    print(f"[INFO] Params : {sum(p.numel() for p in model.parameters())}\n")

    # 3. Training loop with early stopping
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss   = run_epoch(model, loaders["val"],   criterion)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "narx_best.pt"))
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>4d}/{EPOCHS}  "
                  f"train_loss={train_loss:.6e}  val_loss={val_loss:.6e}  "
                  f"EarlyStop={patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"\n[INFO] Training finished in {elapsed:.1f}s")

    # 4. Load best weights and evaluate
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "narx_best.pt"),
                                     map_location=DEVICE))

    mse_train = evaluate_mse(model, loaders["train"], scaler_y)
    mse_val   = evaluate_mse(model, loaders["val"],   scaler_y)
    mse_test  = evaluate_mse(model, loaders["test"],  scaler_y)
    mse_estim = evaluate_mse(model, loaders["estim"], scaler_y)

    print("\n══════════════════════════════════════════")
    print("  NARX Open-Loop Performance (MSE, original scale, kWh/timestamp)")
    print("══════════════════════════════════════════")
    print(f"  Train  MSE : {mse_train:.6e}")
    print(f"  Val    MSE : {mse_val:.6e}")
    print(f"  Test   MSE : {mse_test:.6e}")
    print(f"  Estim  MSE : {mse_estim:.6e}   ← paper benchmark ~ 1.99e-5")
    print("══════════════════════════════════════════\n")

    # 5. Save scalers alongside checkpoint
    scalers_path = os.path.join(CKPT_DIR, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(data["scalers"], f)
    print(f"[DONE] Best model : {os.path.join(CKPT_DIR, 'narx_best.pt')}")
    print(f"[DONE] Scalers    : {scalers_path}")

    return model, data


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROC_DIR = os.path.join(ROOT, "data", "processed")
    train_csv = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_csv = os.path.join(PROC_DIR, "acn_estim_clean.csv")

    if not os.path.exists(train_csv):
        print(
            "[ERROR] Preprocessed CSVs not found.\n"
            "  Run `python src/data/preprocess.py` first to generate:\n"
            f"    {train_csv}\n"
            f"    {estim_csv}"
        )
        sys.exit(1)

    df_train = pd.read_csv(train_csv)
    df_estim = pd.read_csv(estim_csv)

    # Parse datetime columns that may have been saved as strings
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        if col in df_train.columns:
            df_train[col] = pd.to_datetime(df_train[col], utc=True)
            df_estim[col] = pd.to_datetime(df_estim[col], utc=True)

    train(df_train, df_estim)
