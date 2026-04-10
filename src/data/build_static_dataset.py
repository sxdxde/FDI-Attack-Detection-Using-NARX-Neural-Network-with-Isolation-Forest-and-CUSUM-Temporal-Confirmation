"""
Script to parse the massive unrolled ACN-Data-Static repository.
Sample 30k sessions across Caltech, JPL, and Office001 evenly.
Resamples 3-second raw telemetry into 5-minute timesteps.
Outputs acn_train_clean.csv and acn_estim_clean.csv containing the actual physical charging data.
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(ROOT, "data", "ACN-Data-Static", "time series data")
OUT_DIR    = os.path.join(ROOT, "data", "processed")

MAX_SESSIONS = 15_000
TRAIN_SPLIT  = 0.70
TIMESTEP     = '5min'  # 5 minutes

def get_all_csv_gz():
    print(f"[INFO] Scanning {STATIC_DIR} for .csv.gz files...")
    files = glob.glob(os.path.join(STATIC_DIR, "**", "*.csv.gz"), recursive=True)
    print(f"[INFO] Found {len(files)} total sessions.")
    return files

def parse_session(file_path: str):
    """
    Schema:
    ,Charging Current (A),Actual Pilot (A),Voltage (V),Charging State,Energy Delivered (kWh),Power (kW)
    """
    try:
        # The first column is unnamed and contains datetime strings
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(df) == 0:
            return None
            
        # Clean col names
        df.columns = [c.strip() for c in df.columns]
        
        # We need Energy Delivered, Power, Voltage, Charging Current
        essential = ["Charging Current (A)", "Voltage (V)", "Energy Delivered (kWh)", "Power (kW)"]
        for col in essential:
            if col not in df.columns:
                df[col] = 0.0
                
        df = df[essential]
        
        # Convert to numeric, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Forward fill any small gaps before resampling
        df = df.ffill().fillna(0.0)
        
        # Resample to 5 minutes
        # Mean for rates (Current, Voltage, Power)
        # Max for cumulatives (Energy Delivered)
        agg_rules = {
            "Charging Current (A)": "mean",
            "Voltage (V)": "mean",
            "Power (kW)": "mean",
            "Energy Delivered (kWh)": "max"  # cumulative
        }
        resampled = df.resample(TIMESTEP).agg(agg_rules)
        resampled = resampled.ffill().fillna(0.0)
        
        # If the session is too short (e.g. 1 or 2 steps), ignore it for sequence modelling
        if len(resampled) < 5:
            return None
            
        # Extract filename info
        filename = os.path.basename(file_path)
        site = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # caltech/jpl/etc
        session_id = filename.replace(".csv.gz", "")
        
        # Add metadata columns
        resampled = resampled.reset_index()
        resampled.rename(columns={"index": "connectionTime"}, inplace=True)
        resampled["sessionID"] = session_id
        resampled["siteID"] = site
        resampled["timestamps"] = np.arange(len(resampled))
        
        # Align with target schema: The original model predicted 'kWhDeliveredPerTimeStamp' 
        # which is the differential energy delivered in that 5-min block.
        # Since 'Energy Delivered (kWh)' is cumulative, differential is diff()
        resampled["kWhDeliveredPerTimeStamp"] = resampled["Energy Delivered (kWh)"].diff().fillna(
            resampled["Energy Delivered (kWh)"].iloc[0]
        ).clip(lower=0.0)
        
        return resampled
        
    except Exception as e:
        # Some gz files might be corrupted or empty
        return None

def build_dataset():
    files = get_all_csv_gz()
    if len(files) == 0:
        raise ValueError(f"No files found in {STATIC_DIR}")
        
    # Sample files to avoid OOM
    if len(files) > MAX_SESSIONS:
        print(f"[INFO] Sampling {MAX_SESSIONS} files from the corpus randomly...")
        random.seed(42)
        files = random.sample(files, MAX_SESSIONS)
    else:
        print(f"[INFO] Using all {len(files)} files.")
        
    all_sessions = []
    success = 0
    
    print("[INFO] Processing sessions (resampling to 5min timesteps)...")
    for i, f in enumerate(files):
        if i > 0 and i % 2000 == 0:
            print(f"       -> Parsed {i} / {len(files)} ...")
            
        sess_df = parse_session(f)
        if sess_df is not None:
            all_sessions.append(sess_df)
            success += 1
            
    if not all_sessions:
        raise RuntimeError("No sessions could be successfully parsed!")
        
    print(f"[INFO] Successfully parsed {success} valid sessions.")
    master_df = pd.concat(all_sessions, ignore_index=True)
    
    print(f"[INFO] Master dataset shape: {master_df.shape}")
    print("[INFO] Columns:", master_df.columns.tolist())
    
    # ── Train / Estim split ──
    unique_sessions = master_df["sessionID"].unique()
    n_train_sess = int(len(unique_sessions) * TRAIN_SPLIT)
    train_sess = set(unique_sessions[:n_train_sess])
    
    df_train = master_df[master_df["sessionID"].isin(train_sess)].copy()
    df_estim = master_df[~master_df["sessionID"].isin(train_sess)].copy()
    
    print(f"[INFO] df_train shape: {df_train.shape}")
    print(f"[INFO] df_estim shape: {df_estim.shape}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    train_out = os.path.join(OUT_DIR, "acn_train_clean.csv")
    estim_out = os.path.join(OUT_DIR, "acn_estim_clean.csv")
    
    df_train.to_csv(train_out, index=False)
    df_estim.to_csv(estim_out, index=False)
    print(f"[DONE] Saved static dataset to {OUT_DIR}")

if __name__ == "__main__":
    build_dataset()
