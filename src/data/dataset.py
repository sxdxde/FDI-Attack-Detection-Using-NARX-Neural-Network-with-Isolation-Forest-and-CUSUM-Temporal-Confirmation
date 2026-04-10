"""
Dataset Builder
Creates time-delayed input windows for NARX and Attention-BiLSTM training.

NARX (flat vectors):
  x(t) = [u(t), u(t-1), ..., u(t-mx+1),   <- exogenous inputs
           y(t-1), y(t-2), ..., y(t-my)]    <- past target values

Attention-BiLSTM (3-D sequences):
  x(t) = (seq_len, n_features) window of past exogenous rows

MUST prevent cross-session leakage by building windows per session.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "kWhDeliveredPerTimeStamp"

EXOG_COLS = [
    "siteID",
    "connectionTime_unix",
    "timestamps",                 # The 5-minute timestep offset
    "Charging Current (A)",
    "Voltage (V)",
    "Energy Delivered (kWh)",
    "Power (kW)",
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
    max_sessions: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct NARX input/output pairs grouped by sessionID to avoid leakage.
    If a session is shorter than max(mx, my), it's skipped.

    Uses sort-based grouping (O(n log n)) instead of per-session boolean masking
    (O(n_sessions × n_total)) for efficiency with large datasets.

    max_sessions: if set, randomly subsample to this many sessions (for speed).
    """
    n_delay  = max(mx, my)
    n_feat   = X_sc.shape[1]
    inp_size = n_feat * mx + my

    # Sort by session ID for O(1) group slicing
    sort_idx = np.argsort(session_ids, kind="stable")
    s_sorted = session_ids[sort_idx]
    X_sorted = X_sc[sort_idx]
    y_sorted = y_sc[sort_idx]

    _, first_occ, counts = np.unique(s_sorted, return_index=True, return_counts=True)

    if max_sessions is not None and max_sessions < len(first_occ):
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(first_occ), size=max_sessions, replace=False)
        chosen.sort()
        first_occ = first_occ[chosen]
        counts    = counts[chosen]

    inputs, targets = [], []

    for start, cnt in zip(first_occ, counts):
        if cnt <= n_delay:
            continue
        X_s = X_sorted[start : start + cnt]
        y_s = y_sorted[start : start + cnt]

        for t in range(n_delay, cnt):
            exog_window = X_s[t - mx + 1 : t + 1][::-1].flatten()
            past_y      = y_s[t - my : t][::-1]
            inputs.append(np.concatenate([exog_window, past_y]))
            targets.append(y_s[t])

    if len(inputs) == 0:
        return np.zeros((0, inp_size), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def build_sequence_windows(
    X_sc: np.ndarray,
    y_sc: np.ndarray,
    session_ids: np.ndarray,
    seq_len: int = 4,
    max_sessions: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (seq_len, n_features) 3-D windows for Attention-BiLSTM.

    Each sample is a rolling window of `seq_len` past exogenous rows
    within the same charging session (no cross-session leakage).

    Uses sort-based grouping for efficiency with large datasets.
    max_sessions: if set, randomly subsample to this many sessions.

    Returns
    -------
    inputs  : (N, seq_len, n_features)  float32
    targets : (N,)                      float32
    """
    n_feat = X_sc.shape[1]

    sort_idx = np.argsort(session_ids, kind="stable")
    s_sorted = session_ids[sort_idx]
    X_sorted = X_sc[sort_idx]
    y_sorted = y_sc[sort_idx]

    _, first_occ, counts = np.unique(s_sorted, return_index=True, return_counts=True)

    if max_sessions is not None and max_sessions < len(first_occ):
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(first_occ), size=max_sessions, replace=False)
        chosen.sort()
        first_occ = first_occ[chosen]
        counts    = counts[chosen]

    inputs, targets = [], []

    for start, cnt in zip(first_occ, counts):
        if cnt <= seq_len:
            continue
        X_s = X_sorted[start : start + cnt]
        y_s = y_sorted[start : start + cnt]

        for t in range(seq_len, cnt):
            inputs.append(X_s[t - seq_len : t])
            targets.append(y_s[t])

    if len(inputs) == 0:
        return np.zeros((0, seq_len, n_feat), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


class NARXDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(inputs,  dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Dataset for Attention-BiLSTM: inputs are (seq_len, n_features) tensors."""
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        # inputs: (N, seq_len, n_features)
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
    model_type: str = "narx",    # "narx" | "bilstm"
    seq_len: int = 4,            # only used when model_type="bilstm"
    max_train_sessions: int = 10000,   # cap training sessions for tractability
    max_estim_sessions: int = 5000,    # cap estimation sessions for tractability
) -> dict:

    X_tr_raw, y_tr_raw, s_tr = prepare_features(df_train)
    X_es_raw, y_es_raw, s_es = prepare_features(df_estim)

    # Clip outlier targets: max L2 charger output is ~0.64 kWh/5min (32A×240V×5/60/1000).
    # Values up to 1.0 kWh add a safety margin. Without clipping, outliers (up to 39 kWh)
    # compress all normal values into a tiny [0, 0.016] region of scaled space.
    Y_CLIP = 1.0
    y_tr_raw = np.clip(y_tr_raw, 0.0, Y_CLIP)
    y_es_raw = np.clip(y_es_raw, 0.0, Y_CLIP)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_tr_sc = scaler_X.fit_transform(X_tr_raw).astype(np.float32)
    y_tr_sc = scaler_y.fit_transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float32)

    X_es_sc = scaler_X.transform(X_es_raw).astype(np.float32)
    y_es_sc = scaler_y.transform(y_es_raw.reshape(-1, 1)).flatten().astype(np.float32)

    if model_type == "bilstm":
        X_tr_w, y_tr_w = build_sequence_windows(
            X_tr_sc, y_tr_sc, s_tr, seq_len=seq_len,
            max_sessions=max_train_sessions)
        X_es_w, y_es_w = build_sequence_windows(
            X_es_sc, y_es_sc, s_es, seq_len=seq_len,
            max_sessions=max_estim_sessions)
        DatasetCls = SequenceDataset
    else:
        X_tr_w, y_tr_w = build_narx_windows_per_session(
            X_tr_sc, y_tr_sc, s_tr, mx=mx, my=my,
            max_sessions=max_train_sessions)
        X_es_w, y_es_w = build_narx_windows_per_session(
            X_es_sc, y_es_sc, s_es, mx=mx, my=my,
            max_sessions=max_estim_sessions)
        DatasetCls = NARXDataset

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
        return DataLoader(DatasetCls(x, y), batch_size=batch_size, shuffle=shuffle)

    if model_type == "bilstm":
        shape_info = {"seq_len": X_tr_w.shape[1] if len(X_tr_w) else 0,
                      "n_features": X_tr_w.shape[2] if len(X_tr_w) else 0}
        print(f"[INFO] BiLSTM window: seq_len={shape_info['seq_len']}  n_features={shape_info['n_features']}")
    else:
        shape_info = {"input_size": X_tr_w.shape[1] if len(X_tr_w) else 0}
        print(f"[INFO] NARX input size: {shape_info['input_size']}")
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
        "shapes":  shape_info,
        "raw": {
            "X_train_w": X_tr_w, "y_train_w": y_tr_w,
            "X_estim_w": X_es_w, "y_estim_w": y_es_w,
            # Pass through site IDs for per-site evaluation
            "site_ids_train": df_train["siteID"].values if "siteID" in df_train.columns else None,
            "site_ids_estim": df_estim["siteID"].values if "siteID" in df_estim.columns else None,
        },
    }
