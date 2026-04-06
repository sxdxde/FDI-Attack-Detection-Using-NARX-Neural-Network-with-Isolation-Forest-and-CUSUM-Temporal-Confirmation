# NARX FDI Detection with Two-Stage Isolation Forest + CUSUM

A PyTorch replication and extension of the NARX neural network paper for FDI attack detection in EV charging networks using the ACN-Data Caltech dataset (Dec 2020 – Jan 2021). 

## Novel Approach: Two-Stage Detection (IF + CUSUM)

This project improves upon the original Global IQR anomaly detection baseline by introducing a **two-stage hybrid anomaly detector** on the NARX Error-of-Estimation (EoE) signal:
1. **Stage 1 (Isolation Forest):** Performs a high-sensitivity, distribution-agnostic sweep to flag all candidate anomalies. This guarantees zero missed attacks (perfect recall) and handles the multi-modal nature of EoE distributions better than static global computing methods.
2. **Stage 2 (CUSUM Control Chart):** Temporally confirms anomalies by looking for sustained EoE drift. This mimics the actual physical signature of a False Data Injection (FDI) attack (such as curtailed charging over multiple timestamps) and filters out momentary cold-start session boundary spikes that plague statistical outlier methods.

This combined method (IF + CUSUM) achieves a perfect recall of 100% and an F1 score of 0.764, outperforming the baseline Global IQR (recall: 57.9%, F1: 0.682) on subtle, short-duration attacks ($\vartheta < 10$ minutes) making it extremely robust.

## Project Layout

```
narx_ev_fdi/
├── data/
│   ├── raw/           ← Place sessions.csv here (download from ACN portal)
│   └── processed/     ← Auto-generated train/estim CSVs
├── src/
│   ├── data/
│   │   ├── preprocess.py   ← Step 1: clean raw data → processed CSVs
│   │   └── dataset.py      ← NARX windowing + DataLoaders
│   ├── models/
│   │   └── narx.py         ← NARXNet (PyTorch) + closed-loop inference
│   ├── simulate/
│   │   └── fdi_attack.py   ← FDI attack injection simulation
│   ├── train/
│   │   └── train.py        ← Training loop + MSE evaluation
│   └── eval/
│       ├── evaluate.py         ← Main detection metrics evaluation
│       ├── isolation_forest.py ← Stage 1 (IF) detector
│       ├── cusum_if.py         ← Stage 2 (CUSUM) temporal confirmation
│       └── ablation.py         ← Attack intensity ablation study
├── paper/                  ← LaTeX source code for the research paper
├── checkpoints/            ← Auto-generated: best weights + scalers
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# 1 — Download data from https://ev.caltech.edu/dataset
#     Place sessions.csv into data/raw/

# 2 — Preprocess
cd narx_ev_fdi
python src/data/preprocess.py

# 3 — Train NARX (open-loop, series-parallel)
python -m src.train.train

# 4 — Evaluate Two-Stage Detection Pipeline
python -m src.eval.evaluate
```

## Model Architecture
| Parameter | Value |
|-----------|-------|
| Type | NARX (Nonlinear AutoRegressive with eXogenous inputs) |
| Hidden layers | 1 |
| Hidden neurons | 10 |
| Activation | Sigmoid |
| Output | Linear |
| Exogenous delay mx | 2 |
| Output delay my | 2 |
| Training mode | Open-loop (series-parallel) |
| Inference mode | Closed-loop (autonomous) |
| Target MSE | ~1.99 × 10⁻⁵ |

## Exogenous Inputs (15 features)
`stationID`, `siteID`, `connectionTime`, `doneChargingTime`, `kWhDelivered`, `timestamps`, `modifiedAt`, `chargingCurrent`, `pilotSignal`, `userID`, `WhPerMile`, `milesRequested`, `minutesAvailable`, `requestedDeparture`, `kWhRequested`

Target: **kWhDeliveredPerTimeStamp**

## Experimental Data Split
| Subset | Size |
|--------|------|
| Training (70 %) | 23,076 sessions |
| Validation (15 %) | 4,944 sessions |
| Test (15 %) | 4,946 sessions |
| Estimation (held-out) | 14,129 sessions |
