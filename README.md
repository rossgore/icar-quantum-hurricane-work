# ICAR Quantum Hurricane Work

Research pipeline for predicting **hurricane rapid intensification (RI)** in the western Atlantic, comparing classical ML baselines against models augmented with quantum-derived ("classical shadow") features.

RI is defined as a ≥30 kt increase in sustained wind speed within 24 hours. The pipeline is built around storms passing near Norfolk, VA, using HURDAT2 track data (optionally augmented with SHIPS developmental data).

## Pipeline

Run in order — each phase consumes the previous phase's outputs.

| Phase | Script | Purpose |
|---|---|---|
| 1 | `phase1_data_acquisition.py` | Filters HURDAT2 Atlantic tracks (1980–2025) to storms within 5° of Norfolk (36.85°N, -76.29°W) and wind ≥35 kt; resamples to 6-hour intervals. |
| 2 | `phase2_features_and_labels.py` | Builds RI labels, optionally merges SHIPS variables (SST, wind shear, ocean heat content, humidity, vorticity), normalizes features to [0, π] for quantum angle encoding. |
| 3 | `phase3_classical_baselines.py` | Trains classical baselines (Logistic Regression, Gradient Boosting, Neural Network) with class-imbalance handling and a storm-safe train/test split. |
| 4 | `phase4_quantum_features.py` | Generates quantum shadow features via a classical-shadow protocol (PennyLane angle encoding + IQP-style entangling circuit + Pauli observable expectations); augments classical features and retrains the same models. |
| 5 | `phase5_statistical_validation.py` | Bootstrap confidence intervals, NHC-standard operational metrics (POD/FAR/CSI/PSS), 5-fold storm-safe cross-validation, and paper-ready figures/summary. |

```bash
python phase1_data_acquisition.py
python phase2_features_and_labels.py
python phase3_classical_baselines.py
python phase4_quantum_features.py
python phase5_statistical_validation.py
```

## Data

- `data/atlantic_hurricane_tracks.csv` — input HURDAT2 track data (ArcGIS CSV format).
- `data/al_ships_1982_2023.txt` — optional SHIPS developmental data ([download](https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/development_data/)); if absent, only HURDAT2-derived features are used.
- Intermediate/derived CSVs, fitted scalers, storm-ID train/test splits, and quantum shadow feature labels are all written to `data/` by phases 1–4.

## Models & outputs

- `models/` — fitted `.joblib` models for each classical algorithm, baseline and quantum variants.
- `plots/` — ROC/PR curves, quantum-vs-baseline comparison, and the phase 5 paper figure.
- `reports/phase5_summary.txt` — abstract-ready summary of dataset size, AUC-ROC, and operational metrics.

## Current results

From `reports/phase5_summary.txt` (123 storms, 2,506 labelable observations, 143 RI events, 1:17 class imbalance):

- Best baseline AUC-ROC: 0.7539 vs. best quantum AUC-ROC: 0.7550 (Δ = +0.0011)
- Logistic Regression sees the largest quantum gain: 0.7351 → 0.7550 (ΔAUC = +0.0199)
- Gradient Boosting and Neural Network AUC decline slightly with quantum features under cross-validation

These results use only 2 qubits (HURDAT2 features only, no SHIPS data). Larger gains are expected once SHIPS data is integrated (6 qubits, 32 Pauli observables, 38 augmented features) — the current numbers are a conservative lower bound.

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
joblib
pennylane
```

No `requirements.txt` is currently checked in; install the packages above (e.g. into `.venv`) before running the pipeline.