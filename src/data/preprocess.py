"""
ACN-Data Preprocessing for NARX-based FDI Attack Detection
Dataset: Caltech ACN, Dec 2020 – Jan 2021
Paper replication: NARX neural network for FDI attack detection in EV charging networks
"""

"""
ACN-Data Preprocessing for NARX-based FDI Attack Detection
Dataset: Caltech ACN, Dec 2020 – Jan 2021
Paper replication: Explode each session into discrete 5-minute stamps.
"""

import os
import json
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "sessions.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Run download_acn.py first.")

df = pd.read_csv(CSV_PATH)
print(f"[INFO] Raw dataset shape (sessions): {df.shape}")

# ─────────────────────────────────────────────
# 2. PARSE TIMESTAMPS
# ─────────────────────────────────────────────
for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True)

# ─────────────────────────────────────────────
# 3. EXTRACT USER INPUTS
# ─────────────────────────────────────────────
# The 'userInputs' field might be a JSON list or empty
def parse_ui(row, field):
    ui = row.get("userInputs")
    if pd.isna(ui) or ui == "none" or ui == "[]" or not ui:
        return np.nan
    try:
        # Some are serialized JSON
        if isinstance(ui, str):
            parsed = json.loads(ui.replace("'", '"'))
        else:
            parsed = ui
            
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0].get(field, np.nan)
        return np.nan
    except Exception:
        return np.nan

df["WhPerMile"]          = df.apply(lambda r: parse_ui(r, "WhPerMile"), axis=1)
df["milesRequested"]     = df.apply(lambda r: parse_ui(r, "milesRequested"), axis=1)
df["minutesAvailable"]   = df.apply(lambda r: parse_ui(r, "minutesAvailable"), axis=1)
df["kWhRequested"]       = df.apply(lambda r: parse_ui(r, "kWhRequested"), axis=1)
df["userID"]             = df.apply(lambda r: parse_ui(r, "userID"), axis=1)
df["requestedDeparture"] = df.apply(lambda r: parse_ui(r, "requestedDeparture"), axis=1)
df["modifiedAt"]         = df.apply(lambda r: parse_ui(r, "modifiedAt"), axis=1)

df["requestedDeparture"] = pd.to_datetime(df["requestedDeparture"], utc=True)
df["modifiedAt"]         = pd.to_datetime(df["modifiedAt"], utc=True)

# ─────────────────────────────────────────────
# 4. REMOVE SESSIONS WITH MISSING USER INPUTS & NULLS
# ─────────────────────────────────────────────
# If any of these are NaN, user didn't provide inputs
df = df.dropna(subset=["kWhRequested", "minutesAvailable"])
df = df.dropna(subset=["connectionTime", "doneChargingTime", "kWhDelivered"])

# We don't have chargingCurrent / pilotSignal natively in the sessions API
# (those are internal to ACN-Sim / raw time-series). We will mock them
# or derive them as average delivered since we are predicting delivered.
if "chargingCurrent" not in df.columns:
    df["chargingCurrent"] = 32.0  # default pilot signal in ACN
if "pilotSignal" not in df.columns:
    df["pilotSignal"] = 32.0
if "stationID" not in df.columns:
    df["stationID"] = "Unknown"
if "siteID" not in df.columns:
    df["siteID"] = "0001"

# ─────────────────────────────────────────────
# 5. EXPLODE INTO 5-MINUTE TIMESTAMPS
# ─────────────────────────────────────────────
TIMESTEP_MINS = 5

# Derive numTimeStamps
duration_mins = (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 60.0
df["numTimeStamps"] = (duration_mins / TIMESTEP_MINS).clip(lower=1).round().astype(int)
df["kWhDeliveredPerTimeStamp"] = df["kWhDelivered"] / df["numTimeStamps"]

print(f"[INFO] Total sessions after cleaning: {len(df)}")

exploded_rows = []
for idx, row in df.iterrows():
    n_steps = int(row["numTimeStamps"])
    # Replicate row n_steps times
    for t in range(n_steps):
        new_row = row.copy()
        new_row["timestamps"] = t  # The 'timestamps' feature is the offset
        exploded_rows.append(new_row)

df_exp = pd.DataFrame(exploded_rows)
print(f"[INFO] Exploded shape (samples): {df_exp.shape}")

# ─────────────────────────────────────────────
# 6. TRAIN / ESTIMATION SPLIT
# ─────────────────────────────────────────────
N_TRAIN = 32966
N_ESTIM = 14129

# If we have less than N_TRAIN, we just use all as train and warn
# Or we pad/scale up if strictly replicating counts!
total = len(df_exp)
if total < N_TRAIN + N_ESTIM:
    print(f"[WARN] Paper used ~47k samples. We only got {total} from this 2-month window via API.")
    print(f"       We will split 70% / 30% dynamically instead of hardcoded numbers.")
    split_idx = int(total * 0.70)
else:
    split_idx = N_TRAIN

df_exp = df_exp.sort_values(["connectionTime", "timestamps"]).reset_index(drop=True)

df_train = df_exp.iloc[:split_idx].copy()
df_estim = df_exp.iloc[split_idx:].copy()

print(f"[INFO] Training set  : {df_train.shape}")
print(f"[INFO] Estimation set: {df_estim.shape}")

train_out = os.path.join(BASE_DIR, "data", "processed", "acn_train_clean.csv")
estim_out = os.path.join(BASE_DIR, "data", "processed", "acn_estim_clean.csv")
os.makedirs(os.path.dirname(train_out), exist_ok=True)

df_train.to_csv(train_out, index=False)
df_estim.to_csv(estim_out, index=False)
print(f"\n[DONE] Saved exploded datasets to {os.path.dirname(train_out)}")
