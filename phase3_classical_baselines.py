"""
Phase 3: Classical Baseline Models
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Input:  data/normalized_dataset.csv    (Phase 2 output)

Outputs:
  data/train_storm_ids.txt            (storm IDs in training set — reused in all phases)
  data/test_storm_ids.txt             (storm IDs in test set)
  data/baseline_results.csv           (per-model metrics table)
  models/logreg_baseline.joblib
  models/xgb_baseline.joblib
  models/nn_baseline.joblib
  plots/baseline_roc_pr_curves.png

Primary metric:  AUC-ROC  (class-imbalance robust)
Secondary metric: Recall on RI=1 (operationally critical)

Class imbalance handling:
  Logistic Regression : class_weight='balanced'
  XGBoost             : scale_pos_weight = n_negative / n_positive
  Neural Network      : sample_weight proportional to inverse class frequency
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve,
                              precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

NORMALIZED_CSV = "data/normalized_dataset.csv"
OUT_TRAIN_IDS  = "data/train_storm_ids.txt"
OUT_TEST_IDS   = "data/test_storm_ids.txt"
OUT_RESULTS    = "data/baseline_results.csv"
MODELS_DIR     = "models"
PLOTS_DIR      = "plots"
RANDOM_STATE   = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Step 1: Load data ──────────────────────────────────────────────────────────
print("── Step 1: Loading normalized dataset ───────────────────────────────")
df = pd.read_csv(NORMALIZED_CSV)
df_labeled = df[df['ri_label'].notna()].copy()
df_labeled['ri_label'] = df_labeled['ri_label'].astype(int)

norm_features = [c for c in df_labeled.columns if c.endswith('_norm')]
X      = df_labeled[norm_features].values
y      = df_labeled['ri_label'].values
groups = df_labeled['storm_id'].values

print(f"  Labelable observations: {len(df_labeled):,}")
print(f"  RI positive:            {y.sum():,}  ({y.mean()*100:.1f}%)")
print(f"  RI negative:            {(y==0).sum():,}")
print(f"  Features:               {norm_features}")
print(f"  Class imbalance ratio:  1 : {(y==0).sum() / y.sum():.1f}")

# ── Step 2: Storm-based train/test split ───────────────────────────────────────
print("\n── Step 2: Storm-based train/test split (80/20) ─────────────────────")
# Split by storm_id, not by observation, to prevent data leakage.
# StratifiedGroupKFold ensures no storm appears in both train and test,
# while preserving the RI class ratio in both splits.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
splits = list(sgkf.split(X, y, groups))
train_idx, test_idx = splits[0]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_storms    = np.unique(groups[train_idx])
test_storms     = np.unique(groups[test_idx])

assert len(set(train_storms) & set(test_storms)) == 0, "Storm leakage detected!"

np.savetxt(OUT_TRAIN_IDS, train_storms, fmt='%s')
np.savetxt(OUT_TEST_IDS,  test_storms,  fmt='%s')

print(f"  Training storms:  {len(train_storms)}  |  observations: {len(X_train):,}")
print(f"  Test storms:      {len(test_storms)}   |  observations: {len(X_test):,}")
print(f"  Train RI rate:    {y_train.mean()*100:.1f}%")
print(f"  Test  RI rate:    {y_test.mean()*100:.1f}%")
print(f"  Split saved → {OUT_TRAIN_IDS}, {OUT_TEST_IDS}")

# ── Step 3: Class imbalance weights ───────────────────────────────────────────
n_pos = y_train.sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / n_pos
sample_weights   = np.where(y_train == 1, scale_pos_weight, 1.0)
print(f"\n  scale_pos_weight: {scale_pos_weight:.1f}")

# ── Step 4: Define models ──────────────────────────────────────────────────────
print("\n── Step 4: Training baseline models ────────────────────────────────")
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE
    ),
    'Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        verbose=0
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        learning_rate_init=0.001
    ),
}

trained_models = {}
for name, model in models.items():
    print(f"  Training {name}...", end=' ', flush=True)
    if name == 'Neural Network':
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    trained_models[name] = model
    print("done.")

# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
print("\n── Step 5: Evaluating models ────────────────────────────────────────")

def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc_roc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    report   = classification_report(y_test, y_pred,
                                     target_names=['No RI', 'RI'],
                                     output_dict=True, zero_division=0)
    ri_prec   = report['RI']['precision']
    ri_recall = report['RI']['recall']
    ri_f1     = report['RI']['f1-score']
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n  ── {name} ──────────────────────────────────")
    print(f"    AUC-ROC:        {auc_roc:.4f}")
    print(f"    Avg Precision:  {avg_prec:.4f}  (imbalance-adjusted)")
    print(f"    RI Recall:      {ri_recall:.4f}  <- primary operational metric")
    print(f"    RI Precision:   {ri_prec:.4f}")
    print(f"    RI F1:          {ri_f1:.4f}")
    print(f"    Confusion:      TN={tn}  FP={fp}  |  FN={fn}  TP={tp}")

    return {
        'model':         name,
        'auc_roc':       round(auc_roc,    4),
        'avg_precision': round(avg_prec,   4),
        'ri_recall':     round(ri_recall,  4),
        'ri_precision':  round(ri_prec,    4),
        'ri_f1':         round(ri_f1,      4),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'features_used': len(norm_features),
        'n_train':       len(X_train),
        'n_test':        len(X_test),
    }

results  = []
roc_data = {}
for name, model in trained_models.items():
    metrics         = evaluate_model(name, model, X_test, y_test)
    results.append(metrics)
    roc_data[name]  = model.predict_proba(X_test)[:, 1]

# ── Step 6: Save models ────────────────────────────────────────────────────────
print("\n── Step 6: Saving models ────────────────────────────────────────────")
for name, model in trained_models.items():
    fname = name.lower().replace(' ', '_')
    path  = os.path.join(MODELS_DIR, f"{fname}_baseline.joblib")
    joblib.dump(model, path)
    print(f"  Saved → {path}")

# ── Step 7: Save results ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_RESULTS, index=False)
print(f"\n  Results saved → {OUT_RESULTS}")

# ── Step 8: ROC + PR curves ────────────────────────────────────────────────────
print("\n── Step 8: Generating plots ─────────────────────────────────────────")
colors = ['steelblue', 'darkorange', 'forestgreen']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for (name, y_prob), color in zip(roc_data.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Classical Baselines')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ri_rate = y_test.mean()
for (name, y_prob), color in zip(roc_data.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)
ax.axhline(ri_rate, color='k', linestyle='--', lw=1,
           label=f'Random baseline ({ri_rate:.3f})')
ax.set_xlabel('Recall (RI Events Detected)')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Classical Baselines')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "baseline_roc_pr_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot saved → {plot_path}")

# ── Step 9: Summary ────────────────────────────────────────────────────────────
print("\n── Baseline Results Summary ─────────────────────────────────────────")
summary_cols = ['model', 'auc_roc', 'avg_precision', 'ri_recall', 'ri_precision', 'ri_f1']
print(results_df[summary_cols].to_string(index=False))
print(f"  Note: These are the 6-feature baselines (HURDAT2 + SHIPS).")
print(f"  The quantum-augmented models in Phase 4 must beat these scores.")
