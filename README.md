# NARX FDI Detection Project
A PyTorch replication of the NARX neural network paper for FDI attack detection in EV charging networks using the ACN-Data Caltech dataset (Dec 2020 – Jan 2021).

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
│   │   └── fdi_attack.py   ← (coming) FDI attack injection
│   ├── train/
│   │   └── train.py        ← Training loop + MSE evaluation
│   └── eval/
│       └── evaluate.py     ← (coming) detection metrics
├── tests/
├── notebooks/
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

## Split
| Subset | Size |
|--------|------|
| Training (70 %) | 23,076 sessions |
| Validation (15 %) | 4,944 sessions |
| Test (15 %) | 4,946 sessions |
| Estimation (held-out) | 14,129 sessions |
