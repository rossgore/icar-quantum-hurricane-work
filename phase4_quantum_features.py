"""
Phase 4: Quantum Feature Engineering via Classical Shadow Protocol
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Inputs:
  data/normalized_dataset.csv         (Phase 2 output)
  data/train_storm_ids.txt            (Phase 3 storm split — must reuse exactly)
  data/test_storm_ids.txt
  data/baseline_results.csv           (Phase 3 metrics for comparison)

Outputs:
  data/augmented_dataset.csv          (original + quantum shadow features)
  data/shadow_feature_labels.txt      (names of shadow feature columns)
  data/quantum_results.csv            (per-model metrics)
  data/phase4_comparison.csv          (baseline vs quantum side-by-side)
  models/logreg_quantum.joblib
  models/gradboost_quantum.joblib
  models/nn_quantum.joblib
  plots/quantum_vs_baseline.png

Quantum approach (LUQPI framework):
  1. Encode normalized features into n_qubits via angle encoding (RY + RZ)
  2. Apply IQP-style ZZ entangling layer (captures pairwise feature correlations)
  3. Apply second rotation layer for expressivity
  4. Compute Pauli string expectation values as quantum-derived shadow features
  5. Augment original feature vector: x_aug = [x_classical, x_shadow]
  6. Retrain same classical models on augmented data using identical train/test split

Quantum hardware: required at training time only.
Deployment: fully classical — shadow features are precomputed and stored.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from itertools import combinations

from sklearn.decomposition import PCA

import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve,
                              precision_recall_curve, average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

NORMALIZED_CSV   = "data/normalized_dataset.csv"
TRAIN_IDS_FILE   = "data/train_storm_ids.txt"
TEST_IDS_FILE    = "data/test_storm_ids.txt"
BASELINE_CSV     = "data/baseline_results.csv"
OUT_AUGMENTED    = "data/augmented_dataset.csv"
OUT_SHADOW_LBLS  = "data/shadow_feature_labels.txt"
OUT_RESULTS      = "data/quantum_results.csv"
OUT_COMPARISON   = "data/phase4_comparison.csv"
MODELS_DIR       = "models"
PLOTS_DIR        = "plots"
RANDOM_STATE     = 42
MAX_OBSERVABLES  = 32    # target number of Pauli string features (matches TornadoQ)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Build Pauli Observable Set
# ══════════════════════════════════════════════════════════════════════════════

def generate_observables(n_qubits, max_obs=MAX_OBSERVABLES):
    """
    Generate a prioritized set of Pauli string observables for shadow extraction.

    Priority:
      1. All single-qubit Z, X, Y  (most informative marginals)
      2. All ZZ qubit pairs         (captures pairwise correlations, key for IQP circuits)
      3. XX, YY pairs               (complementary correlations)
      4. ZX, ZY, XZ, XY, YZ, YX    (cross-basis pairs, fill to max_obs)
      5. Three-body ZZZ triples     (higher-order correlations, n>=3 only)

    Returns:
      observables : list of PennyLane observables
      labels      : list of human-readable strings e.g. 'Z0', 'Z0_Z1'
    """
    observables = []
    labels      = []
    p_ops = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}

    def add(obs, lbl):
        if len(observables) < max_obs:
            observables.append(obs)
            labels.append(lbl)

    # Single-qubit terms
    for name in ['Z', 'X', 'Y']:
        for q in range(n_qubits):
            add(p_ops[name](q), f'{name}{q}')

    # Two-qubit terms in priority order
    two_body_order = [
        ('Z','Z'), ('X','X'), ('Y','Y'),
        ('Z','X'), ('Z','Y'), ('X','Z'),
        ('X','Y'), ('Y','Z'), ('Y','X'),
    ]
    for p1, p2 in two_body_order:
        for q1, q2 in combinations(range(n_qubits), 2):
            add(p_ops[p1](q1) @ p_ops[p2](q2), f'{p1}{q1}_{p2}{q2}')

    # Three-body ZZZ triples
    if n_qubits >= 3:
        for q1, q2, q3 in combinations(range(n_qubits), 3):
            add(qml.PauliZ(q1) @ qml.PauliZ(q2) @ qml.PauliZ(q3),
                f'Z{q1}_Z{q2}_Z{q3}')

    return observables, labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Build Quantum Encoding Circuit
# ══════════════════════════════════════════════════════════════════════════════

def build_shadow_circuit(n_qubits, observables):
    """
    Build a PennyLane QNode that encodes n_qubits features and returns
    the expectation values of all Pauli string observables.

    Encoding strategy (from slide deck):
      Layer 1 - Angle encoding : RY(feature[i]) + RZ(feature[i]) on each qubit
      Layer 2 - IQP entangling : ZZ coupling between adjacent qubits
                                  = CNOT -- RZ(feat[i]*feat[j]) -- CNOT
      Layer 3 - Second rotation: RY(feature[i]) on each qubit for expressivity
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features):
        # Layer 1: Angle encoding
        for i in range(n_qubits):
            qml.RY(float(features[i]), wires=i)
            qml.RZ(float(features[i]), wires=i)

        # Layer 2: IQP-style ZZ entangling
        for i in range(n_qubits - 1):
            theta = float(features[i]) * float(features[(i + 1) % n_qubits])
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(theta, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
        if n_qubits > 2:
            theta = float(features[-1]) * float(features[0])
            qml.CNOT(wires=[n_qubits - 1, 0])
            qml.RZ(theta, wires=0)
            qml.CNOT(wires=[n_qubits - 1, 0])

        # Layer 3: Second rotation layer
        for i in range(n_qubits):
            qml.RY(float(features[i]), wires=i)

        return [qml.expval(obs) for obs in observables]

    return circuit


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Extract Shadow Features
# ══════════════════════════════════════════════════════════════════════════════

def extract_shadow_features(X, circuit, shadow_labels, batch_size=200):
    """
    Compute Pauli string expectation values for every observation.
    Processes in batches and prints progress.

    Returns: shadow_matrix of shape (n_samples, n_observables)
    """
    n_samples  = len(X)
    n_obs      = len(shadow_labels)
    shadow_mat = np.zeros((n_samples, n_obs))

    n_batches = (n_samples + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, n_samples)
        for i in range(start, end):
            result = circuit(X[i])
            shadow_mat[i] = np.array(result)
        pct = (end / n_samples) * 100
        print(f"    Shadow extraction: {end:,}/{n_samples:,}  ({pct:.0f}%)", end='\r')

    print(f"    Shadow extraction: {n_samples:,}/{n_samples:,}  (100%) — done.  ")
    return shadow_mat


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Train & Evaluate Hybrid Models
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc_roc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    report   = classification_report(y_test, y_pred,
                                     target_names=['No RI', 'RI'],
                                     output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n  ── {name} ──────────────────────────────────")
    print(f"    AUC-ROC:        {auc_roc:.4f}")
    print(f"    Avg Precision:  {avg_prec:.4f}")
    print(f"    RI Recall:      {report['RI']['recall']:.4f}  <- primary operational metric")
    print(f"    RI Precision:   {report['RI']['precision']:.4f}")
    print(f"    RI F1:          {report['RI']['f1-score']:.4f}")
    print(f"    Confusion:      TN={tn}  FP={fp}  |  FN={fn}  TP={tp}")

    return {
        'model':         name,
        'auc_roc':       round(auc_roc,                   4),
        'avg_precision': round(avg_prec,                  4),
        'ri_recall':     round(report['RI']['recall'],    4),
        'ri_precision':  round(report['RI']['precision'], 4),
        'ri_f1':         round(report['RI']['f1-score'],  4),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────────────────────────
    print("── Step 8: Loading data & building quantum circuit ──────────────────")
    df = pd.read_csv(NORMALIZED_CSV)
    df_labeled = df[df['ri_label'].notna()].copy()
    df_labeled['ri_label'] = df_labeled['ri_label'].astype(int)

    norm_features = [c for c in df_labeled.columns if c.endswith('_norm')]
    n_qubits = len(norm_features)
    X_all    = df_labeled[norm_features].values
    y_all    = df_labeled['ri_label'].values
    groups   = df_labeled['storm_id'].values

    print(f"  Labelable observations: {len(df_labeled):,}")
    print(f"  Features (= n_qubits):  {norm_features}  →  {n_qubits} qubit(s)")

    # ── Load Phase 3 train/test split ─────────────────────────────────────────
    train_storms = np.loadtxt(TRAIN_IDS_FILE, dtype=str)
    test_storms  = np.loadtxt(TEST_IDS_FILE,  dtype=str)
    train_mask   = np.isin(groups, train_storms)
    test_mask    = np.isin(groups, test_storms)

    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]

    print(f"  Train: {len(X_train):,} obs  |  Test: {len(X_test):,} obs")
    assert len(set(groups[train_mask]) & set(groups[test_mask])) == 0

    # ── Build circuit & observables ───────────────────────────────────────────
    observables, shadow_labels = generate_observables(n_qubits, MAX_OBSERVABLES)
    circuit = build_shadow_circuit(n_qubits, observables)
    print(f"\n  Quantum circuit: {n_qubits} qubit(s), {len(observables)} Pauli observables")
    print(f"  Observable labels: {shadow_labels}")

    # Save observable labels
    with open(OUT_SHADOW_LBLS, 'w') as f:
        f.write('\n'.join(shadow_labels))

    # ── Extract shadow features ───────────────────────────────────────────────
    print(f"\n── Step 9: Extracting shadow features ───────────────────────────────")
    print(f"  Processing {len(X_train):,} training observations...")
    shadow_train = extract_shadow_features(X_train, circuit, shadow_labels)

    print(f"  Processing {len(X_test):,} test observations...")
    shadow_test  = extract_shadow_features(X_test, circuit, shadow_labels)
    n_shadow = len(shadow_labels) 
    # ── PCA on shadow features ────────────────────────────────────────────────
    # Fit only on training shadow features to prevent leakage, then apply to test.
    # n_components=0.95 keeps however many PCs explain 95% of variance (typically 8-12).
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    shadow_train_pca = pca.fit_transform(shadow_train)
    shadow_test_pca  = pca.transform(shadow_test)
    n_components     = shadow_train_pca.shape[1]
    pca_labels       = [f'shadow_pc{i+1}' for i in range(n_components)]
    joblib.dump(pca, 'data/shadow_pca.joblib')
    print(f"\n  PCA on shadow features: {n_shadow} → {n_components} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")

    # ── Build augmented feature matrices ──────────────────────────────────────
    # x_aug = [original normalized features | PCA-compressed shadow features]
    X_train_aug      = np.hstack([X_train, shadow_train_pca])
    X_test_aug       = np.hstack([X_test,  shadow_test_pca])
    aug_feature_names = norm_features + pca_labels

    # Save augmented dataset
    aug_df_train = pd.DataFrame(X_train_aug, columns=aug_feature_names)
    aug_df_test  = pd.DataFrame(X_test_aug,  columns=aug_feature_names)
    aug_df_train['ri_label'] = y_train
    aug_df_test['ri_label']  = y_test
    aug_df_train['split']    = 'train'
    aug_df_test['split']     = 'test'
    pd.concat([aug_df_train, aug_df_test], ignore_index=True).to_csv(OUT_AUGMENTED, index=False)
    print(f"  Augmented feature vector: {len(norm_features)} original + "
          f"{n_components} PCA shadow = {len(aug_feature_names)} total")
    print(f"  Augmented dataset saved → {OUT_AUGMENTED}")

    # ── Train hybrid models ───────────────────────────────────────────────────
    print(f"\n── Step 10: Training hybrid quantum-classical models ────────────────")
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos
    sample_weights   = np.where(y_train == 1, scale_pos_weight, 1.0)

    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1,
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            class_weight='balanced', random_state=RANDOM_STATE, verbose=0
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE, learning_rate_init=0.001
        ),
    }

    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=' ', flush=True)
        if name == 'Neural Network':
            model.fit(X_train_aug, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_aug, y_train)
        trained_models[name] = model
        print("done.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n── Step 11: Evaluating quantum-augmented models ─────────────────────")
    q_results  = []
    q_roc_data = {}
    for name, model in trained_models.items():
        metrics, y_prob = evaluate_model(name, model, X_test_aug, y_test)
        q_results.append(metrics)
        q_roc_data[name] = y_prob
        path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ','_')}_quantum.joblib")
        joblib.dump(model, path)

    q_results_df = pd.DataFrame(q_results)
    q_results_df.to_csv(OUT_RESULTS, index=False)

    # ── Compare against Phase 3 baselines ────────────────────────────────────
    print(f"\n── Step 12: Comparison — Baseline vs Quantum-Augmented ─────────────")
    baseline_df = pd.read_csv(BASELINE_CSV)
    compare_metrics = ['auc_roc', 'avg_precision', 'ri_recall', 'ri_precision', 'ri_f1']

    rows = []
    for _, brow in baseline_df.iterrows():
        mname = brow['model']
        qrow  = q_results_df[q_results_df['model'] == mname]
        if qrow.empty:
            continue
        qrow = qrow.iloc[0]
        row = {'model': mname}
        for m in compare_metrics:
            b_val = brow[m]
            q_val = qrow[m]
            delta = q_val - b_val
            row[f'baseline_{m}']  = b_val
            row[f'quantum_{m}']   = q_val
            row[f'delta_{m}']     = round(delta, 4)
        rows.append(row)

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(OUT_COMPARISON, index=False)

    print(f"\n  {'Model':<22} {'Metric':<18} {'Baseline':>10} {'Quantum':>10} {'Delta':>8}")
    print(f"  {'-'*70}")
    for _, r in compare_df.iterrows():
        for m in ['auc_roc', 'ri_recall', 'avg_precision']:
            b = r[f'baseline_{m}']
            q = r[f'quantum_{m}']
            d = r[f'delta_{m}']
            symbol = '+' if d >= 0 else ''
            print(f"  {r['model']:<22} {m:<18} {b:>10.4f} {q:>10.4f} {symbol}{d:>7.4f}")
    print()

    # ── Plot: ROC + delta bar chart ───────────────────────────────────────────
    print(f"── Step 13: Generating comparison plots ─────────────────────────────")
    colors_base    = ['#4878CF', '#D65F5F', '#6ACC65']
    colors_quantum = ['#1A3A8F', '#8B0000', '#1A6617']
    model_names    = list(trained_models.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: ROC curves (quantum models)
    ax = axes[0]
    for (name, y_prob), color in zip(q_roc_data.items(), colors_quantum):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC — Quantum-Augmented Models')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: AUC-ROC delta (quantum - baseline)
    ax = axes[1]
    auc_deltas = [compare_df.loc[compare_df['model']==m, 'delta_auc_roc'].values[0]
                  for m in model_names if m in compare_df['model'].values]
    bar_colors = [colors_quantum[i] for i in range(len(auc_deltas))]
    bars = ax.bar(model_names[:len(auc_deltas)], auc_deltas, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    for bar, val in zip(bars, auc_deltas):
        sym = '+' if val >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{sym}{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('ΔAUC-ROC (Quantum − Baseline)')
    ax.set_title('AUC-ROC Improvement\nfrom Quantum Features')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: RI Recall delta
    ax = axes[2]
    recall_deltas = [compare_df.loc[compare_df['model']==m, 'delta_ri_recall'].values[0]
                     for m in model_names if m in compare_df['model'].values]
    bars = ax.bar(model_names[:len(recall_deltas)], recall_deltas, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    for bar, val in zip(bars, recall_deltas):
        sym = '+' if val >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{sym}{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('ΔRI Recall (Quantum − Baseline)')
    ax.set_title('RI Recall Improvement\nfrom Quantum Features')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f'Quantum Feature Engineering: {n_qubits} qubits, {len(observables)} Pauli observables\n'
        f'Augmented features: {len(norm_features)} classical + {n_components} PCA shadow = {len(norm_features)+n_components} total',
        fontsize=10
    )
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'quantum_vs_baseline.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {plot_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n── Phase 4 Summary ───────────────────────────────────────────────────")
    print(f"  Quantum circuit:   {n_qubits} qubit(s)")
    print(f"  Shadow features:   {n_shadow} Pauli string expectation values")
    print(f"  Augmented vector:  {len(norm_features) + n_components} total features per observation")
    print(f"  Outputs:")
    print(f"    {OUT_AUGMENTED}")
    print(f"    {OUT_RESULTS}")
    print(f"    {OUT_COMPARISON}")
    print(f"    {plot_path}")
    print(f"\n  Full comparison table saved → {OUT_COMPARISON}")
    print(f"\n  Note: Results above use {n_qubits} qubits (HURDAT2 features only).")
    print(f"  Re-run Phases 2-4 after adding SHIPS data to get the 6-qubit results")
    print(f"  that will form the core of the workshop abstract.")
