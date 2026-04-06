"""
NARX Dataset Builder
Creates time-delayed input windows (open-loop / series-parallel) for NARX training.

  x(t) = [u(t), u(t-1), ..., u(t-mx+1),   <- exogenous inputs
           y(t-1), y(t-2), ..., y(t-my)]    <- past target values

MUST prevent cross-session leakage by building windows per session.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "kWhDeliveredPerTimeStamp"

EXOG_COLS = [
    "stationID",
    "siteID",
    "connectionTime_unix",
    "doneChargingTime_unix",
    "kWhDelivered",
    "timestamps",                 # The 5-minute timestep offset
    "modifiedAt_unix",
    "chargingCurrent",
    "pilotSignal",
    "userID",
    "WhPerMile",
    "milesRequested",
    "minutesAvailable",
    "requestedDeparture_unix",
    "kWhRequested",
]

def _to_unix(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype(np.int64) / 1e9
    try:
        parsed = pd.to_datetime(series, utc=True)
        return parsed.astype(np.int64) / 1e9
    except Exception:
        return series.astype(float)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features, targets, and session IDs.
    Returns: X_raw, y_raw, session_ids
    """
    work = df.copy()

    dt_map = {
        "connectionTime":     "connectionTime_unix",
        "doneChargingTime":   "doneChargingTime_unix",
        "modifiedAt":         "modifiedAt_unix",
        "requestedDeparture": "requestedDeparture_unix",
    }
    for src, dst in dt_map.items():
        if src in work.columns:
            work[dst] = _to_unix(work[src])

    # Hash string categorical features for the neural net (simple numerical encoding)
    for col in ["stationID", "siteID", "userID"]:
        if col in work.columns:
            work[col] = work[col].astype(str).apply(hash) % 10000

    present_exog = [c for c in EXOG_COLS if c in work.columns]
    missing = set(EXOG_COLS) - set(present_exog)
    if missing:
        for c in missing:
            work[c] = 0.0

    X_raw = work[EXOG_COLS].values.astype(np.float32)
    y_raw = work[TARGET_COL].values.astype(np.float32)
    session_ids = work["sessionID"].values
    return X_raw, y_raw, session_ids


def build_narx_windows_per_session(
    X_sc: np.ndarray,
    y_sc: np.ndarray,
    session_ids: np.ndarray,
    mx: int = 2,
    my: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct NARX input/output pairs grouped by sessionID to avoid leakage.
    If a session is shorter than max(mx, my), it's skipped.
    """
    n_delay = max(mx, my)
    inputs, targets = [], []

    # Get indices for each session
    unique_sessions = np.unique(session_ids)
    
    for s_id in unique_sessions:
        mask = (session_ids == s_id)
        X_s = X_sc[mask]
        y_s = y_sc[mask]
        
        n = len(y_s)
        if n <= n_delay:
            continue
            
        for t in range(n_delay, n):
            exog_window = X_s[t - mx + 1 : t + 1][::-1].flatten()
            past_y      = y_s[t - my : t][::-1]
            inp = np.concatenate([exog_window, past_y])
            inputs.append(inp)
            targets.append(y_s[t])

    if len(inputs) == 0:
        # Fallback if somehow things are too small
        return np.zeros((0, EXOG_COLS.__len__()*mx + my)), np.zeros((0,))
        
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


class NARXDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(inputs,  dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_datasets(
    df_train: pd.DataFrame,
    df_estim: pd.DataFrame,
    mx: int = 2,
    my: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 64,
) -> dict:

    X_tr_raw, y_tr_raw, s_tr = prepare_features(df_train)
    X_es_raw, y_es_raw, s_es = prepare_features(df_estim)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_tr_sc = scaler_X.fit_transform(X_tr_raw).astype(np.float32)
    y_tr_sc = scaler_y.fit_transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float32)

    X_es_sc = scaler_X.transform(X_es_raw).astype(np.float32)
    y_es_sc = scaler_y.transform(y_es_raw.reshape(-1, 1)).flatten().astype(np.float32)

    X_tr_w, y_tr_w = build_narx_windows_per_session(X_tr_sc, y_tr_sc, s_tr, mx=mx, my=my)
    X_es_w, y_es_w = build_narx_windows_per_session(X_es_sc, y_es_sc, s_es, mx=mx, my=my)

    # 4. Temporal split on the training windows
    n = len(y_tr_w)
    
    # Paper uses physical sample sizes: if enough data, take them directly
    # 32966 + 14129 = 47095. Total 47k.
    n_train = int(n * (1 - val_ratio - test_ratio))
    n_val   = int(n * val_ratio)

    X_train, y_train = X_tr_w[:n_train],            y_tr_w[:n_train]
    X_val,   y_val   = X_tr_w[n_train:n_train+n_val], y_tr_w[n_train:n_train+n_val]
    X_test,  y_test  = X_tr_w[n_train+n_val:],       y_tr_w[n_train+n_val:]

    def loader(x, y, shuffle=False):
        return DataLoader(NARXDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    print(f"[INFO] NARX input size: {X_tr_w.shape[1] if len(X_tr_w) else 0}")
    print(f"[INFO] Split sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"       estim: {len(y_es_w)}")

    return {
        "loaders": {
            "train": loader(X_train, y_train, shuffle=False),
            "val":   loader(X_val,   y_val),
            "test":  loader(X_test,  y_test),
            "estim": loader(X_es_w,  y_es_w),
        },
        "scalers": {"X": scaler_X, "y": scaler_y},
        "shapes":  {"input_size": X_tr_w.shape[1] if len(X_tr_w) else 0},
        "raw": {
            "X_train_w": X_tr_w, "y_train_w": y_tr_w,
            "X_estim_w": X_es_w, "y_estim_w": y_es_w,
        },
    }
