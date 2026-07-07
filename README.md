# Drift-Aware Time-Series Forecasting for Building Energy Consumption

An end-to-end pipeline for forecasting building energy consumption (HVAC, lighting, MELs) that detects **concept drift** using PELT change-point detection, classifies the type of drift (abrupt / gradual / recurrent / stable), and adaptively retrains forecasting models (XGBoost, LSTM, GRU) as new drift segments appear. A hybrid GRU + XGBoost residual-correction model is also included as a lighter-weight alternative.

## Overview

Building energy time series are non-stationary — HVAC schedules change, occupancy patterns shift, and equipment behavior drifts over time. A model trained once on historical data degrades as these shifts occur. This project addresses that by:

1. **Detecting** structural breaks (change points) in the energy signal using the PELT algorithm.
2. **Classifying** each detected drift as `abrupt`, `gradual`, `recurrent`, or `stable` based on mean shift, slope change, and autocorrelation between segments.
3. **Retraining** forecasting models (XGBoost, LSTM, GRU) incrementally on each new drift segment, using a replay-buffer strategy that mixes new data with a sample of historical data to avoid catastrophic forgetting.
4. **Comparing** baseline (train-once) performance against drift-aware retrained performance.

A second, simpler approach (`scripts/hybrid_model.py`) combines a GRU base forecaster with an XGBoost model trained on the GRU's residuals near known change points, blended using a distance-based drift weight.

## Repository Structure

\```
Adaptive Forecasting Under Data Drift/

├── README.md
|
├── requirements.txt
|
├── .gitignore
|
├── notebooks/
|
   └── SDP_Full_code.ipynb        # Main analysis notebook (full pipeline, see below)

├── scripts/
|
   └── hybrid_model.py       # Standalone hybrid GRU + XGBoost residual model
|
├── data/                     # Place input Excel/CSV files here (not tracked by git)
|
└── results/                  # Saved plots, metrics, and model outputs

\```

### Main notebook (`notebooks/MJJ_SDP.ipynb`) sections

| Section | What it does |
|---|---|
| 1. Loading Libraries and Data | Imports, loads the raw Excel energy dataset |
| 2. Data Preprocessing | Builds `total_energy` target, fills missing values, removes outliers |
| 3. Feature Engineering | Time-of-day/week cyclic encodings, lag features, rolling stats |
| 4. Model Training (Baseline) | Trains XGBoost, LSTM, and GRU on a single train/test split |
| 5. Drift Detection | Custom PELT implementation to find change points in the energy signal |
| 6. Segmentation | Splits the series into segments at each change point |
| 7. Drift Classification | Labels each change point as abrupt / gradual / recurrent / stable |
| 8. Model Retraining | Incrementally retrains XGBoost, LSTM, GRU on each drift segment with replay |
| 9. Combining(GRU + XGboost) hybrid model | Trained the GRU with raw data and trained XGBoost with residual data at drifted sections |
| 10. Evaluation | Compares MAE / RMSE / R² before vs. after drift-aware retraining vs. Hybrid |

## Dataset

The pipeline expects an Excel file with a `date` column plus sub-metered energy columns (in the sample data: `mels_s`, `lig_s`, `mels_n`, `hvac_n`, `hvac_s`), which are summed into a `total_energy` target. The raw data file itself is **not included** in this repo (see `.gitignore`) — add your own file under `data/` and update the file path in the notebook's first section.

## Installation

\```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
\```

## Usage

**Run the full pipeline (recommended):**
\```bash
jupyter notebook notebooks/SDP_Full_code.ipynb
\```
Update the data path in Section 1 to point to your own Excel file, then run all cells top to bottom.

**Run the standalone hybrid model:**
\```bash
python scripts/hybrid_model.py
\```
Note: this script currently uses Colab's `files.upload()` for data input — replace that line with a local file path (e.g. `pd.read_excel("data/your_file.xlsx")`) before running outside Colab.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (baseline) | 10.907 | 15.413 | 0.635 |
| LSTM (baseline) | 8.205 | 11.733 | 0.789 |
| GRU (baseline) | 8.487 | 11.853 | 0.784 |
| XGBoost (after drift retraining) | 8.655 | 12.024 | 0.764 |
| LSTM (after drift retraining) | 7.522 | 10.713 | 0.813 |
| GRU (after drift retraining) | 7.834 | 10.933 | 0.803 |
| Hybrid (GRU + XGBoost Residual) | 5.607 | 7.699 | 0.906 |


## Tech Stack

- Python, pandas, numpy
- scikit-learn (scaling, metrics)
- XGBoost
- TensorFlow / Keras (LSTM, GRU)
- ruptures (change-point detection reference implementation)
- matplotlib, seaborn (visualization)

## Future Work

- Refactor notebook logic into reusable modules under `src/` (e.g. `preprocessing.py`, `drift.py`, `models.py`)
- Add unit tests for the PELT and drift classification functions
- Parameterize file paths and hyperparameters via a config file (`config.yaml`) instead of hardcoding


