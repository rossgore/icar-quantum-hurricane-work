# ICAR Quantum Hurricane Work

Research pipeline for predicting **hurricane rapid intensification (RI)** in the western Atlantic, comparing classical ML baselines against models augmented with quantum-derived ("classical shadow") features.

RI is defined as a ≥30 kt increase in sustained wind speed within 24 hours. The pipeline is built around storms passing near Norfolk, VA, using HURDAT2 track data (optionally augmented with SHIPS developmental data).

## Phase 4 vs. Phase 4b: online vs. offline quantum features

Both phases augment the classical HURDAT2/SHIPS features with the same quantum "classical shadow" features (Pauli observable expectation values from a simulated PennyLane circuit), but they use them completely differently:

- **Phase 4 ("quantum-online"):** the shadow features are part of the model's input vector, full stop. To classify a *new* observation, you must first run it through the quantum circuit to get its shadow features, then hand `[classical features, shadow features]` to the classifier. Quantum computation (or a quantum simulation) is required at deployment time, not just during training.
- **Phase 4b ("LUQPI-offline"):** the shadow features are only ever used to help *train* the model — specifically, SVM+ ([luqpi_svm.py](luqpi_svm.py)) uses them to shape the slack variables that decide how much margin violation to tolerate for each training point. The resulting decision function depends only on the ordinary classical features; the trained SVM+ model never touches a shadow feature again after `fit()`. So a Phase 4b model can be deployed with zero quantum computation, exactly like the Phase 3 classical baseline — it just got to be *smarter* about the difficult/boundary cases by having seen quantum-derived information during training.

This is the "Learning Under Quantum Privileged Information" (LUQPI) idea from [arXiv:2601.22006](https://arxiv.org/abs/2601.22006): get some benefit from a quantum computer while confining its use entirely to a one-time training step.

## Pipeline

Run in order — each phase consumes the previous phase's outputs.

| Phase | Script | Purpose |
|---|---|---|
| 1 | `phase1_data_acquisition.py` | Filters HURDAT2 Atlantic tracks (1980–2025) to storms within 5° of Norfolk (36.85°N, -76.29°W) and wind ≥35 kt; resamples to 6-hour intervals. |
| 2 | `phase2_features_and_labels.py` | Builds RI labels, optionally merges SHIPS variables (SST, wind shear, ocean heat content, humidity, vorticity), normalizes features to [0, π] for quantum angle encoding. |
| 3 | `phase3_classical_baselines.py` | Trains classical baselines (Logistic Regression, Gradient Boosting, Neural Network) with class-imbalance handling and a storm-safe train/test split. |
| 4 | `phase4_quantum_features.py` | Generates quantum shadow features via a classical-shadow protocol (PennyLane angle encoding + IQP-style entangling circuit + Pauli observable expectations); augments classical features and retrains the same models. Quantum shadow computation is required both at training **and** at deployment/test time ("online" use of quantum features). |
| 4b | `phase4b_luqpi.py` | Learning Under Quantum Privileged Information ([LUQPI](https://arxiv.org/abs/2601.22006)): reuses Phase 4's shadow-PCA features, but only as *privileged information* available during training (Vapnik & Vashist's SVM+ algorithm). Deployment uses classical features only — no quantum computation needed at inference, unlike Phase 4. Sweeps training-set size (50–400 obs) since the LUQPI advantage is concentrated in the low-data regime, tracking AUC-ROC, Recall *and* Precision at every size (`plots/luqpi_svm_comparison.png`). Also produces `plots/luqpi_vs_quantum_vs_baseline.png`, an apples-to-apples 3-way comparison (baseline / quantum-online / LUQPI, all retrained on the identical n=400 subsample). |
| 5 | `phase5_statistical_validation.py` | Bootstrap confidence intervals, NHC-standard operational metrics (POD/FAR/CSI/PSS), 5-fold storm-safe cross-validation (Phase 3 vs. Phase 4), plus bootstrap CI/operational metrics for the Phase 4b LUQPI models — all folded into a 6-panel `plots/phase5_paper_figure.png` and `reports/phase5_summary.txt`. |

```bash
python phase1_data_acquisition.py
python phase2_features_and_labels.py
python phase3_classical_baselines.py
python phase4_quantum_features.py
python phase4b_luqpi.py
python phase5_statistical_validation.py
```

## Data

- `data/atlantic_hurricane_tracks.csv` — input HURDAT2 track data (ArcGIS CSV format).
- `data/lsdiaga_1982_2023_sat_ts_7day.txt.gz` — optional SHIPS developmental data, gzip-compressed ([download](https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/development_data/)); if absent, only HURDAT2-derived features are used.
- Intermediate/derived CSVs, fitted scalers, storm-ID train/test splits, and quantum shadow feature labels are all written to `data/` by phases 1–4.

## Models & outputs

- `models/` — fitted `.joblib` models for each classical algorithm, baseline and quantum variants, plus the Phase 4b `svm_baseline_luqpi.joblib` / `svm_plus_luqpi.joblib` pair.
- `plots/` — ROC/PR curves, quantum-vs-baseline comparison, the Phase 4b LUQPI training-size sweep, `luqpi_vs_quantum_vs_baseline.png`, and the 6-panel phase 5 paper figure.
- `reports/phase5_summary.txt` — abstract-ready summary of dataset size, AUC-ROC, operational metrics, and LUQPI results.

## Current results

From `reports/phase5_summary.txt` (123 storms, 2,506 labelable observations, 143 RI events, 1:17 class imbalance, 6-qubit circuit with HURDAT2 + SHIPS features, 32 Pauli shadow observables):

- Best baseline AUC-ROC: 0.7539 vs. best quantum AUC-ROC: 0.7550 (Δ = +0.0011)
- Logistic Regression sees the largest quantum gain: 0.7351 → 0.7550 (ΔAUC = +0.0199)
- Gradient Boosting and Neural Network AUC decline slightly with quantum features under cross-validation

### Phase 4b (LUQPI)

From `data/luqpi_results.csv` and `data/phase4b_comparison.csv`, sweeping training-set size from 50 to 400 storm-safe-sampled observations (10 random seeds each), with the same fixed test set as Phases 3/4 (SVM and SVM+ always trained on the *same* size at every point — this comparison was already apples-to-apples):

- AUC-ROC is roughly comparable between plain SVM and SVM+ across all training sizes (both ~0.65–0.74) — privileged information doesn't make SVM+ fundamentally better at *ranking* observations.
- RI Recall — the operationally critical metric — is dramatically higher for SVM+ at a naive 0.5/zero-crossing threshold: e.g. at 400 training observations, SVM recall is 0.086 vs. SVM+ recall of 0.807. **This comes at a real precision cost**, though: precision drops from ~0.45 (SVM) to ~0.09 (SVM+) at the same threshold (`plots/luqpi_svm_comparison.png`, bottom-left panel). SVM+ isn't smarter, it's operating at a much more liberal decision threshold — it catches almost every RI event but with many more false alarms. Privileged shadow features let SVM+ avoid the total recall collapse that plain SVM suffers under severe imbalance (~6% RI rate) with no class-weight correction, consistent with the LUQPI paper's own finding that privileged information most helps underrepresented classes/boundary cases — but that recall gain and the precision cost are two sides of the same coin, not a free win.
- At each model's own CSI-optimal threshold (Phase 5's convention, `plots/phase5_paper_figure.png` panel F), both the recall gap and the precision gap narrow (plain SVM can also reach high recall by choosing a very permissive threshold) — the two views aren't contradictory, just different operating points; see `reports/phase5_summary.txt` for both.
- Unlike Phase 4's quantum-augmented models, Phase 4b's deployment path is 100% classical — quantum shadow features are used only while fitting SVM+, never at prediction time.

**`plots/luqpi_vs_quantum_vs_baseline.png`** puts all three pipeline stages side by side, but — unlike a naive comparison of each stage's own headline model — retrains a classical Logistic Regression (with and without quantum shadow features) and SVM/SVM+ **all on the identical n=400 subsample**, so the only thing that differs between bars is *how* quantum information is used (never / online / offline-privileged), not how much training data was available. At that matched size, AUC-ROC is similar across all four (~0.68–0.71); LUQPI's SVM+ again stands out only on recall (0.64 vs. 0.36–0.39 for the others), at a similar precision cost to the sweep above. The full-data (n=2004) Phase 3/4 numbers are noted for reference in the figure's caption but are *not* plotted as bars, since comparing them directly against n=400 models would conflate "more data" with "quantum information."

## Dependencies

Run
```
pip install -r requirements.txt
```