"""
ACN-Data-Static Preprocessor
─────────────────────────────────────────────────────────────────
Reads the compressed per-session CSV files from data/ACN-Data-Static/
and produces the clean train/estimation CSVs used by the model.

Key fixes over the old preprocess.py (which used a small sessions.csv API dump):
  1. Uses REAL measured Charging Current from each session file.
  2. Derives kWhDeliveredPerTimeStamp from Energy Delivered delta when available,
     falling back to Current × 240V × (5/60 h) / 1000 for sessions without it.
  3. Resamples each session to 5-minute bins (matching the NARX paper).
  4. Clips unrealistically high targets (outliers from DC fast chargers / data
     artifacts) to a maximum of 1.0 kWh per 5-min step.
  5. Filters sessions where > 95% of timesteps are idle (zero current), which
     contribute only noise to the zero-target problem.

Output columns (matches dataset.py EXOG_COLS + target):
  connectionTime, Charging Current (A), Voltage (V), Power (kW),
  Energy Delivered (kWh), sessionID, siteID, timestamps,
  kWhDeliveredPerTimeStamp

Usage (from narx_ev_fdi/ project root):
    python src/data/preprocess_static.py
"""

import os
import sys
import gzip
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "data", "ACN-Data-Static", "time series data")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# Physics constants
TIMESTEP_MINS = 5
VOLTAGE_V     = 240.0        # Standard L2 charger voltage (US)
MIN_ACTIVE_RATIO = 0.05      # Skip sessions where <5% of timesteps have current>0
Y_CLIP        = 1.0          # Max kWh per 5-min step (32A×240V×5/60/1000 ≈ 0.64)


def read_session_gz(filepath: str) -> pd.DataFrame | None:
    """Read a single .csv.gz session file. Returns None on failure."""
    try:
        with gzip.open(filepath, "rt") as gz:
            df = pd.read_csv(gz, index_col=0, parse_dates=True)
        if df.empty or "Charging Current (A)" not in df.columns:
            return None
        return df
    except Exception:
        return None


def process_session(filepath: str, site: str, station: str) -> list[dict] | None:
    """
    Process one session file → list of 5-minute row dicts.

    Returns None if the session should be skipped.
    """
    df = read_session_gz(filepath)
    if df is None:
        return None

    # Drop rows with null current (shouldn't happen, but be safe)
    df = df.dropna(subset=["Charging Current (A)"])
    if len(df) < 2:
        return None

    # ── Skip idle-dominated sessions ────────────────────────────
    active_frac = (df["Charging Current (A)"] > 0.5).mean()
    if active_frac < MIN_ACTIVE_RATIO:
        return None

    # ── Resample to 5-minute bins ───────────────────────────────
    # Use UTC-aware index — handle mixed ISO8601 formats (with/without sub-seconds)
    df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
    rule = f"{TIMESTEP_MINS}min"

    agg_dict = {"Charging Current (A)": "mean"}
    has_energy = (
        "Energy Delivered (kWh)" in df.columns
        and df["Energy Delivered (kWh)"].notna().any()
    )
    if has_energy:
        agg_dict["Energy Delivered (kWh)"] = "last"  # cumulative → take last

    resampled = df.resample(rule).agg(agg_dict).dropna(subset=["Charging Current (A)"])
    if len(resampled) < 2:
        return None

    current = resampled["Charging Current (A)"].fillna(0.0).values

    # ── Derive target ────────────────────────────────────────────
    if has_energy:
        energy_cum = resampled["Energy Delivered (kWh)"].fillna(method="ffill").fillna(0.0).values
        # Delta of cumulative energy per 5-min step
        energy_delta = np.diff(energy_cum, prepend=0.0)
        energy_delta = np.clip(energy_delta, 0.0, None)   # no negative energy
    else:
        # Derive from current: P = I × V, E = P × t / 1000
        dt_hours = TIMESTEP_MINS / 60.0
        energy_delta = current * VOLTAGE_V * dt_hours / 1000.0

    # Clip outliers
    energy_delta = np.clip(energy_delta, 0.0, Y_CLIP)

    # ── Build output rows ────────────────────────────────────────
    session_id  = os.path.basename(filepath).replace(".csv.gz", "")
    conn_time   = resampled.index[0]
    n_steps     = len(resampled)

    rows = []
    for t in range(n_steps):
        # Cumulative energy at this 5-min step
        cum_energy = float(np.sum(energy_delta[:t + 1]))
        rows.append({
            "connectionTime":            str(conn_time),
            "Charging Current (A)":      float(current[t]),
            "Voltage (V)":               VOLTAGE_V,
            "Power (kW)":                float(current[t] * VOLTAGE_V / 1000.0),
            "Energy Delivered (kWh)":    cum_energy,
            "sessionID":                 session_id,
            "siteID":                    site,
            "stationID":                 station,
            "timestamps":                t,
            "kWhDeliveredPerTimeStamp":  float(energy_delta[t]),
        })
    return rows


def collect_all_files() -> list[tuple[str, str, str]]:
    """Walk STATIC_DIR and return (filepath, site, station) tuples."""
    entries = []
    for site in sorted(os.listdir(STATIC_DIR)):
        site_path = os.path.join(STATIC_DIR, site)
        if not os.path.isdir(site_path) or site.startswith("."):
            continue
        for station in sorted(os.listdir(site_path)):
            st_path = os.path.join(site_path, station)
            if not os.path.isdir(st_path) or station.startswith("."):
                continue
            for fname in sorted(os.listdir(st_path)):
                if fname.endswith(".csv.gz") and not fname.startswith("."):
                    entries.append((os.path.join(st_path, fname), site, station))
    return entries


def process_batch(entries: list[tuple]) -> list[dict]:
    """Process a batch of session files and return all rows."""
    all_rows = []
    for filepath, site, station in entries:
        rows = process_session(filepath, site, station)
        if rows:
            all_rows.extend(rows)
    return all_rows


def main():
    print("[INFO] Collecting session file paths …")
    all_files = collect_all_files()
    print(f"[INFO] Found {len(all_files):,} session files across all sites")

    # ── Process in parallel (8 workers) ──────────────────────────
    # Split into chunks for ProcessPoolExecutor
    CHUNK = 500
    chunks = [all_files[i:i + CHUNK] for i in range(0, len(all_files), CHUNK)]
    print(f"[INFO] Processing {len(chunks)} chunks of ≤{CHUNK} sessions each …")

    all_rows = []
    n_processed = 0
    n_skipped   = 0

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(process_batch, chunk): idx
                   for idx, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)
                n_processed += len(rows) // 10   # approx sessions
            else:
                n_skipped += CHUNK
            done = futures[fut]
            if (done + 1) % 10 == 0:
                print(f"  Chunk {done + 1}/{len(chunks)}: "
                      f"{len(all_rows):,} rows collected so far")

    if not all_rows:
        print("[ERROR] No rows collected. Check STATIC_DIR path.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"\n[INFO] Total rows collected: {len(df):,}")
    print(f"[INFO] Sites: {df['siteID'].value_counts().to_dict()}")
    print(f"[INFO] Unique sessions: {df['sessionID'].nunique():,}")
    print(f"[INFO] Target stats (kWhDeliveredPerTimeStamp):")
    print(f"       mean={df['kWhDeliveredPerTimeStamp'].mean():.4f}  "
          f"std={df['kWhDeliveredPerTimeStamp'].std():.4f}  "
          f"max={df['kWhDeliveredPerTimeStamp'].max():.4f}  "
          f"% zero={(df['kWhDeliveredPerTimeStamp']==0).mean()*100:.1f}%")

    # ── Sort chronologically and split 70/30 ─────────────────────
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True)
    df = df.sort_values(["connectionTime", "timestamps"]).reset_index(drop=True)

    # Per-session split: assign sessions to train or estim set
    # This prevents leakage — all timesteps of a session go to the same split
    unique_sessions = df["sessionID"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_sessions)
    n_train_sess = int(len(unique_sessions) * 0.70)
    train_sessions = set(unique_sessions[:n_train_sess])
    estim_sessions = set(unique_sessions[n_train_sess:])

    df_train = df[df["sessionID"].isin(train_sessions)].reset_index(drop=True)
    df_estim = df[df["sessionID"].isin(estim_sessions)].reset_index(drop=True)

    print(f"\n[INFO] Training set  : {df_train.shape} "
          f"({df_train['sessionID'].nunique():,} sessions)")
    print(f"[INFO] Estimation set: {df_estim.shape} "
          f"({df_estim['sessionID'].nunique():,} sessions)")

    # ── Save ─────────────────────────────────────────────────────
    train_out = os.path.join(PROC_DIR, "acn_train_clean.csv")
    estim_out = os.path.join(PROC_DIR, "acn_estim_clean.csv")

    df_train.to_csv(train_out, index=False)
    df_estim.to_csv(estim_out, index=False)
    print(f"\n[DONE] Saved:")
    print(f"  Training  → {train_out}")
    print(f"  Estimation → {estim_out}")


if __name__ == "__main__":
    main()
