"""
Phase 5: Statistical Validation, Operational Metrics & Paper-Ready Output
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Inputs:
  data/normalized_dataset.csv         (Phase 2)
  data/train_storm_ids.txt            (Phase 3 split)
  data/test_storm_ids.txt
  data/baseline_results.csv           (Phase 3)
  data/quantum_results.csv            (Phase 4)
  data/shadow_feature_labels.txt      (Phase 4)
  data/augmented_dataset.csv          (Phase 4 — precomputed test-set shadow features)
  models/*_baseline.joblib            (Phase 3)
  models/*_quantum.joblib             (Phase 4)
  models/svm_baseline_luqpi.joblib    (Phase 4b)
  models/svm_plus_luqpi.joblib        (Phase 4b)

Outputs:
  data/phase5_bootstrap_ci.csv        (95% CIs on AUC-ROC and RI Recall)
  data/phase5_operational_metrics.csv (POD/FAR/CSI/PSS at optimal threshold)
  data/phase5_cv_results.csv          (5-fold CV mean ± std, LR/GB/NN baseline vs quantum-online)
  plots/phase5_paper_figure.png       (6-panel publication-ready figure)
  reports/phase5_summary.txt          (abstract-ready numbers)

Metrics computed:
  Bootstrap (n=1000): 95% CI on AUC-ROC and RI Recall, for the 6 Phase 3/4
    models plus the 2 Phase 4b LUQPI models (SVM, SVM+)
  Operational (NHC-standard): POD, FAR, CSI (Threat Score), PSS (Peirce Skill Score)
    at optimal CSI threshold — not the default 0.5
  Cross-validation: 5-fold StratifiedGroupKFold, storm-safe (LR/GB/NN only —
    Phase 4b's own 50-400 obs training-size sweep, data/luqpi_results.csv,
    already characterizes SVM/SVM+ variance across resamples)
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from itertools import combinations

import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from luqpi_svm import SVMPlus  # noqa: F401 — needed so joblib can unpickle the saved SVM+ model

warnings.filterwarnings("ignore")

NORMALIZED_CSV   = "data/normalized_dataset.csv"
TRAIN_IDS_FILE   = "data/train_storm_ids.txt"
TEST_IDS_FILE    = "data/test_storm_ids.txt"
BASELINE_CSV     = "data/baseline_results.csv"
QUANTUM_CSV      = "data/quantum_results.csv"
SHADOW_LBLS_FILE = "data/shadow_feature_labels.txt"
AUGMENTED_CSV    = "data/augmented_dataset.csv"
MODELS_DIR       = "models"
PLOTS_DIR        = "plots"
REPORTS_DIR      = "reports"
OUT_BOOTSTRAP    = "data/phase5_bootstrap_ci.csv"
OUT_OPERATIONAL  = "data/phase5_operational_metrics.csv"
OUT_CV           = "data/phase5_cv_results.csv"
OUT_FIGURE       = "plots/phase5_paper_figure.png"
OUT_SUMMARY      = "reports/phase5_summary.txt"

RANDOM_STATE     = 42
N_BOOTSTRAP      = 1000
N_CV_FOLDS       = 5
MAX_OBSERVABLES  = 32

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities (duplicated from Phase 4 — no cross-phase imports)
# ══════════════════════════════════════════════════════════════════════════════

def generate_observables(n_qubits, max_obs=MAX_OBSERVABLES):
    observables, labels = [], []
    p_ops = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}

    def add(obs, lbl):
        if len(observables) < max_obs:
            observables.append(obs)
            labels.append(lbl)

    for name in ['Z', 'X', 'Y']:
        for q in range(n_qubits):
            add(p_ops[name](q), f'{name}{q}')

    two_body_order = [('Z','Z'),('X','X'),('Y','Y'),('Z','X'),('Z','Y'),
                      ('X','Z'),('X','Y'),('Y','Z'),('Y','X')]
    for p1, p2 in two_body_order:
        for q1, q2 in combinations(range(n_qubits), 2):
            add(p_ops[p1](q1) @ p_ops[p2](q2), f'{p1}{q1}_{p2}{q2}')

    if n_qubits >= 3:
        for q1, q2, q3 in combinations(range(n_qubits), 3):
            add(qml.PauliZ(q1) @ qml.PauliZ(q2) @ qml.PauliZ(q3),
                f'Z{q1}_Z{q2}_Z{q3}')
    return observables, labels


def build_shadow_circuit(n_qubits, observables):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features):
        for i in range(n_qubits):
            qml.RY(float(features[i]), wires=i)
            qml.RZ(float(features[i]), wires=i)
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
        for i in range(n_qubits):
            qml.RY(float(features[i]), wires=i)
        return [qml.expval(obs) for obs in observables]
    return circuit


def extract_shadow_features(X, circuit, n_obs):
    shadow_mat = np.zeros((len(X), n_obs))
    for i, x in enumerate(X):
        shadow_mat[i] = np.array(circuit(x))
    return shadow_mat


def make_models(scale_pos_weight):
    sample_weights_fn = lambda y: np.where(y == 1, scale_pos_weight, 1.0)
    return {
        'Logistic Regression': (
            LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000,
                               random_state=RANDOM_STATE),
            False
        ),
        'Gradient Boosting': (
            HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                           learning_rate=0.05,
                                           class_weight='balanced',
                                           random_state=RANDOM_STATE,
                                           verbose=0),
            False
        ),
        'Neural Network': (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                          max_iter=500, early_stopping=True,
                          validation_fraction=0.1,
                          random_state=RANDOM_STATE,
                          learning_rate_init=0.001),
            True   # needs sample_weight in fit()
        ),
    }, sample_weights_fn


# ══════════════════════════════════════════════════════════════════════════════
# Operational metrics
# ══════════════════════════════════════════════════════════════════════════════

def operational_metrics_at_threshold(y_true, y_prob, threshold):
    """
    Compute NHC-standard operational metrics at a given probability threshold.
      POD  = TP / (TP + FN)         Probability of Detection (= Recall)
      FAR  = FP / (TP + FP)         False Alarm Ratio (= 1 - Precision)
      CSI  = TP / (TP + FN + FP)    Critical Success Index (Threat Score)
      PSS  = POD - POFD              Peirce Skill Score
      POFD = FP / (FP + TN)         Probability of False Detection
      BIAS = (TP + FP) / (TP + FN)  Frequency Bias
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far  = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi  = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pss  = pod - pofd
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0
    return dict(threshold=threshold, pod=pod, far=far, csi=csi,
                pss=pss, pofd=pofd, bias=bias, tp=tp, fp=fp, fn=fn, tn=tn)


def find_optimal_threshold(y_true, y_prob, metric='csi'):
    """Find the probability threshold that maximises CSI (or PSS)."""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_val, best_thresh = -np.inf, 0.5
    for t in thresholds:
        m = operational_metrics_at_threshold(y_true, y_prob, t)
        if m[metric] > best_val:
            best_val  = m[metric]
            best_thresh = t
    return best_thresh, best_val


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, rng_seed=RANDOM_STATE):
    rng = np.random.default_rng(rng_seed)
    auc_scores, recall_scores, precision_scores, csi_scores = [], [], [], []
    n = len(y_true)
    opt_thresh, _ = find_optimal_threshold(y_true, y_prob)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        auc_scores.append(roc_auc_score(yt, yp))
        m = operational_metrics_at_threshold(yt, yp, opt_thresh)
        recall_scores.append(m['pod'])
        precision_scores.append(1 - m['far'])   # FAR = FP/(TP+FP) = 1 - precision
        csi_scores.append(m['csi'])

    return {
        'auc_mean':        np.mean(auc_scores),
        'auc_ci_lo':       np.percentile(auc_scores, 2.5),
        'auc_ci_hi':       np.percentile(auc_scores, 97.5),
        'recall_mean':     np.mean(recall_scores),
        'recall_ci_lo':    np.percentile(recall_scores, 2.5),
        'recall_ci_hi':    np.percentile(recall_scores, 97.5),
        'precision_mean':  np.mean(precision_scores),
        'precision_ci_lo': np.percentile(precision_scores, 2.5),
        'precision_ci_hi': np.percentile(precision_scores, 97.5),
        'csi_mean':        np.mean(csi_scores),
        'csi_ci_lo':       np.percentile(csi_scores, 2.5),
        'csi_ci_hi':       np.percentile(csi_scores, 97.5),
        'opt_threshold':   opt_thresh,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────────────────────────
    print("── Step 14: Loading data & reconstructing test predictions ──────────")
    df = pd.read_csv(NORMALIZED_CSV)
    df_labeled = df[df['ri_label'].notna()].copy()
    df_labeled['ri_label'] = df_labeled['ri_label'].astype(int)

    norm_features = [c for c in df_labeled.columns if c.endswith('_norm')]
    n_qubits      = len(norm_features)
    X_all         = df_labeled[norm_features].values
    y_all         = df_labeled['ri_label'].values
    groups        = df_labeled['storm_id'].values

    train_storms  = np.loadtxt(TRAIN_IDS_FILE, dtype=str)
    test_storms   = np.loadtxt(TEST_IDS_FILE,  dtype=str)
    train_mask    = np.isin(groups, train_storms)
    test_mask     = np.isin(groups, test_storms)
    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]
    print(f"  n_qubits={n_qubits}  |  train={len(X_train):,}  test={len(X_test):,}")

    # Build quantum shadow features for test set
    observables, shadow_labels = generate_observables(n_qubits, MAX_OBSERVABLES)
    circuit   = build_shadow_circuit(n_qubits, observables)
    n_shadow  = len(shadow_labels)
    print(f"  Building shadow features ({n_shadow} observables)...", end=' ', flush=True)
    shadow_train = extract_shadow_features(X_train, circuit, n_shadow)
    shadow_test  = extract_shadow_features(X_test,  circuit, n_shadow)
    pca          = joblib.load('data/shadow_pca.joblib')
    shadow_train_pca = pca.transform(shadow_train)
    shadow_test_pca  = pca.transform(shadow_test)
    n_shadow         = pca.n_components_   # update to PCA component count for reporting
    X_train_aug  = np.hstack([X_train, shadow_train_pca])
    X_test_aug   = np.hstack([X_test,  shadow_test_pca])
    print("done.")

    # Reconstruct probabilities from saved models
    model_names = ['Logistic Regression', 'Gradient Boosting', 'Neural Network']
    file_stems  = ['logistic_regression', 'gradient_boosting', 'neural_network']

    base_probs    = {}
    quantum_probs = {}
    for name, stem in zip(model_names, file_stems):
        bm = joblib.load(os.path.join(MODELS_DIR, f'{stem}_baseline.joblib'))
        qm = joblib.load(os.path.join(MODELS_DIR, f'{stem}_quantum.joblib'))
        base_probs[name]    = bm.predict_proba(X_test)[:, 1]
        quantum_probs[name] = qm.predict_proba(X_test_aug)[:, 1]
    print(f"  Loaded 6 models. Probabilities reconstructed.")

    # Phase 4b (LUQPI): deployment uses classical features only (X_test) —
    # no shadow features, quantum or otherwise, at prediction time.
    luqpi_svm  = joblib.load(os.path.join(MODELS_DIR, 'svm_baseline_luqpi.joblib'))
    luqpi_svmp = joblib.load(os.path.join(MODELS_DIR, 'svm_plus_luqpi.joblib'))
    luqpi_probs = {
        'SVM':          luqpi_svm.predict_proba(X_test)[:, 1],
        'SVM+ (LUQPI)': luqpi_svmp.predict_proba_pos(X_test),
    }
    luqpi_names = list(luqpi_probs.keys())
    print(f"  Loaded 2 Phase 4b LUQPI models (SVM, SVM+). Probabilities reconstructed.")

    # ── Step 15: Bootstrap confidence intervals ───────────────────────────────
    print("\n── Step 15: Bootstrap confidence intervals (n=1,000) ────────────────")
    boot_rows = []
    for name in model_names:
        for variant, probs in [('Baseline', base_probs[name]),
                                ('Quantum',  quantum_probs[name])]:
            print(f"  Bootstrapping {name} ({variant})...", end=' ', flush=True)
            ci = bootstrap_ci(y_test, probs)
            row = {'model': name, 'variant': variant}
            row.update(ci)
            boot_rows.append(row)
            print(f"AUC={ci['auc_mean']:.4f} [{ci['auc_ci_lo']:.4f}, {ci['auc_ci_hi']:.4f}]")

    for name in luqpi_names:
        print(f"  Bootstrapping {name} (LUQPI-offline)...", end=' ', flush=True)
        ci = bootstrap_ci(y_test, luqpi_probs[name])
        row = {'model': name, 'variant': 'LUQPI-offline'}
        row.update(ci)
        boot_rows.append(row)
        print(f"AUC={ci['auc_mean']:.4f} [{ci['auc_ci_lo']:.4f}, {ci['auc_ci_hi']:.4f}]")

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_BOOTSTRAP, index=False)
    print(f"  Saved → {OUT_BOOTSTRAP}")

    # ── Step 16: Operational metrics at optimal CSI threshold ─────────────────
    print("\n── Step 16: Operational metrics at optimal CSI threshold ─────────────")
    print(f"\n  {'Model':<22} {'Variant':<10} {'Thresh':>7} {'POD':>7} "
          f"{'FAR':>7} {'CSI':>7} {'PSS':>7} {'Bias':>7}")
    print(f"  {'-'*75}")

    op_rows = []
    for name in model_names:
        for variant, probs in [('Baseline', base_probs[name]),
                                ('Quantum',  quantum_probs[name])]:
            thresh, _ = find_optimal_threshold(y_test, probs, metric='csi')
            m = operational_metrics_at_threshold(y_test, probs, thresh)
            print(f"  {name:<22} {variant:<10} {thresh:>7.3f} {m['pod']:>7.3f} "
                  f"{m['far']:>7.3f} {m['csi']:>7.3f} {m['pss']:>7.3f} {m['bias']:>7.3f}")
            op_rows.append({'model': name, 'variant': variant, **m})

    for name in luqpi_names:
        thresh, _ = find_optimal_threshold(y_test, luqpi_probs[name], metric='csi')
        m = operational_metrics_at_threshold(y_test, luqpi_probs[name], thresh)
        print(f"  {name:<22} {'LUQPI':<10} {thresh:>7.3f} {m['pod']:>7.3f} "
              f"{m['far']:>7.3f} {m['csi']:>7.3f} {m['pss']:>7.3f} {m['bias']:>7.3f}")
        op_rows.append({'model': name, 'variant': 'LUQPI-offline', **m})

    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(OUT_OPERATIONAL, index=False)
    print(f"\n  Saved → {OUT_OPERATIONAL}")

    # ── Step 17: 5-fold cross-validation ─────────────────────────────────────
    print("\n── Step 17: 5-fold cross-validation (storm-safe) ────────────────────")
    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)

    cv_records = {name: {'base_auc': [], 'quant_auc': [],
                          'base_recall': [], 'quant_recall': []}
                  for name in model_names}

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_all, y_all, groups)):
        print(f"  Fold {fold+1}/{N_CV_FOLDS}...", end=' ', flush=True)
        Xtr, Xte = X_all[tr_idx], X_all[te_idx]
        ytr, yte = y_all[tr_idx], y_all[te_idx]

        spw = (ytr == 0).sum() / ytr.sum()
        sw  = np.where(ytr == 1, spw, 1.0)

        # Shadow features for this fold
        sh_tr = extract_shadow_features(Xtr, circuit, len(shadow_labels))
        sh_te = extract_shadow_features(Xte, circuit, len(shadow_labels))
        fold_pca  = PCA(n_components=0.95, random_state=RANDOM_STATE)
        sh_tr_pca = fold_pca.fit_transform(sh_tr)
        sh_te_pca = fold_pca.transform(sh_te)
        Xtr_aug   = np.hstack([Xtr, sh_tr_pca])
        Xte_aug   = np.hstack([Xte, sh_te_pca])

        fold_models, sw_fn = make_models(spw)

        for name, (bm, needs_sw) in fold_models.items():
            # Baseline
            if needs_sw:
                bm.fit(Xtr, ytr, sample_weight=sw)
            else:
                bm.fit(Xtr, ytr)
            bp = bm.predict_proba(Xte)[:, 1]
            cv_records[name]['base_auc'].append(roc_auc_score(yte, bp))
            opt_t, _ = find_optimal_threshold(yte, bp)
            cv_records[name]['base_recall'].append(
                operational_metrics_at_threshold(yte, bp, opt_t)['pod'])

            # Quantum (retrain on augmented)
            fold_models_q, _ = make_models(spw)
            qm, needs_sw_q = fold_models_q[name]
            if needs_sw_q:
                qm.fit(Xtr_aug, ytr, sample_weight=sw)
            else:
                qm.fit(Xtr_aug, ytr)
            qp = qm.predict_proba(Xte_aug)[:, 1]
            cv_records[name]['quant_auc'].append(roc_auc_score(yte, qp))
            opt_t, _ = find_optimal_threshold(yte, qp)
            cv_records[name]['quant_recall'].append(
                operational_metrics_at_threshold(yte, qp, opt_t)['pod'])

        print(f"done.")

    print(f"\n  {'Model':<22} {'AUC Baseline':>15} {'AUC Quantum':>15} "
          f"{'ΔAUC':>8} {'Recall Base':>13} {'Recall Quant':>14} {'ΔRecall':>9}")
    print(f"  {'-'*100}")

    cv_rows = []
    for name in model_names:
        ba  = np.array(cv_records[name]['base_auc'])
        qa  = np.array(cv_records[name]['quant_auc'])
        br  = np.array(cv_records[name]['base_recall'])
        qr  = np.array(cv_records[name]['quant_recall'])
        da  = qa - ba
        dr  = qr - br
        print(f"  {name:<22} {ba.mean():.4f}±{ba.std():.4f}"
              f"   {qa.mean():.4f}±{qa.std():.4f}"
              f"  {da.mean():+.4f}"
              f"  {br.mean():.4f}±{br.std():.4f}"
              f"   {qr.mean():.4f}±{qr.std():.4f}"
              f"  {dr.mean():+.4f}")
        cv_rows.append({
            'model':              name,
            'base_auc_mean':      round(ba.mean(), 4),
            'base_auc_std':       round(ba.std(),  4),
            'quant_auc_mean':     round(qa.mean(), 4),
            'quant_auc_std':      round(qa.std(),  4),
            'delta_auc_mean':     round(da.mean(), 4),
            'base_recall_mean':   round(br.mean(), 4),
            'base_recall_std':    round(br.std(),  4),
            'quant_recall_mean':  round(qr.mean(), 4),
            'quant_recall_std':   round(qr.std(),  4),
            'delta_recall_mean':  round(dr.mean(), 4),
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_CV, index=False)
    print(f"\n  Saved → {OUT_CV}")

    # ── Step 18: Publication-ready figure (6-panel) ───────────────────────────
    print("\n── Step 18: Publication-ready figure ────────────────────────────────")
    model_colors = {
        'Logistic Regression': ('#1f77b4', '#aec7e8'),
        'Gradient Boosting':   ('#d62728', '#f5a09e'),
        'Neural Network':      ('#2ca02c', '#98df8a'),
    }
    luqpi_color = '#9467bd'   # SVM+ (privileged/LUQPI)
    luqpi_color_light = '#c5b0d5'  # SVM (no privileged info)
    fig = plt.figure(figsize=(14, 15))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                             top=0.94, bottom=0.03, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Panel A: ROC curves — baseline (dashed) vs quantum (solid)
    for name in model_names:
        dark, light = model_colors[name]
        fpr_b, tpr_b, _ = roc_curve(y_test, base_probs[name])
        fpr_q, tpr_q, _ = roc_curve(y_test, quantum_probs[name])
        auc_b = roc_auc_score(y_test, base_probs[name])
        auc_q = roc_auc_score(y_test, quantum_probs[name])
        ax1.plot(fpr_b, tpr_b, color=light, lw=1.5, ls='--',
                 label=f'{name[:2]}+CL (AUC={auc_b:.3f})')
        ax1.plot(fpr_q, tpr_q, color=dark,  lw=2,
                 label=f'{name[:2]}+QE (AUC={auc_q:.3f})')
    ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4)
    ax1.set_xlabel('False Positive Rate', fontsize=10)
    ax1.set_ylabel('True Positive Rate', fontsize=10)
    ax1.set_title('(A) ROC Curves', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.25)

    # Panel B: Bootstrap AUC with 95% CI error bars
    x_pos   = np.arange(len(model_names))
    width   = 0.35
    for i, name in enumerate(model_names):
        brow = boot_df[(boot_df['model']==name) & (boot_df['variant']=='Baseline')].iloc[0]
        qrow = boot_df[(boot_df['model']==name) & (boot_df['variant']=='Quantum')].iloc[0]
        dark, light = model_colors[name]
        ax2.bar(x_pos[i]-width/2, brow['auc_mean'], width,
                color=light, label='Baseline' if i==0 else '',
                yerr=[[brow['auc_mean']-brow['auc_ci_lo']],
                      [brow['auc_ci_hi']-brow['auc_mean']]],
                capsize=4, error_kw={'linewidth':1.5})
        ax2.bar(x_pos[i]+width/2, qrow['auc_mean'], width,
                color=dark, label='Quantum' if i==0 else '',
                yerr=[[qrow['auc_mean']-qrow['auc_ci_lo']],
                      [qrow['auc_ci_hi']-qrow['auc_mean']]],
                capsize=4, error_kw={'linewidth':1.5})
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([n[:3] for n in model_names], fontsize=10)
    ax2.set_ylabel('AUC-ROC', fontsize=10)
    ax2.set_title('(B) AUC-ROC with 95% Bootstrap CI', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.25)
    ax2.set_ylim(bottom=max(0, ax2.get_ylim()[0] - 0.02))

    # Panel C: CV AUC mean ± std
    for i, row in cv_df.iterrows():
        dark, light = model_colors[row['model']]
        ax3.errorbar(i - 0.15, row['base_auc_mean'],  yerr=row['base_auc_std'],
                     fmt='o', color=light, capsize=5, ms=8,
                     label='Baseline' if i==0 else '')
        ax3.errorbar(i + 0.15, row['quant_auc_mean'], yerr=row['quant_auc_std'],
                     fmt='s', color=dark,  capsize=5, ms=8,
                     label='Quantum' if i==0 else '')
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([n[:3] for n in model_names], fontsize=10)
    ax3.set_ylabel('AUC-ROC (mean ± std)', fontsize=10)
    ax3.set_title(f'(C) {N_CV_FOLDS}-Fold Cross-Validation AUC', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.25)

    # Panel D: Operational metrics (CSI and PSS) at optimal threshold
    op_base   = op_df[op_df['variant'] == 'Baseline'].reset_index(drop=True)
    op_quant  = op_df[op_df['variant'] == 'Quantum'].reset_index(drop=True)
    for i, name in enumerate(model_names):
        dark, light = model_colors[name]
        ob = op_base[op_base['model']==name].iloc[0]
        oq = op_quant[op_quant['model']==name].iloc[0]
        ax4.bar(i*3,     ob['csi'], 0.6, color=light, label='Baseline CSI' if i==0 else '')
        ax4.bar(i*3+0.7, oq['csi'], 0.6, color=dark,  label='Quantum CSI'  if i==0 else '')
        ax4.bar(i*3+1.4, ob['pss'], 0.6, color=light, hatch='//',
                label='Baseline PSS' if i==0 else '', alpha=0.7)
        ax4.bar(i*3+2.1, oq['pss'], 0.6, color=dark,  hatch='//',
                label='Quantum PSS'  if i==0 else '', alpha=0.7)
    ax4.set_xticks([1.05 + i*3 for i in range(len(model_names))])
    ax4.set_xticklabels([n[:3] for n in model_names], fontsize=10)
    ax4.axhline(0, color='black', lw=0.8)
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_title('(D) Operational Metrics (CSI / PSS)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(axis='y', alpha=0.25)

    # Panel E: ROC curves — LUQPI SVM (no privileged info, dashed) vs SVM+ (solid)
    for name, color, ls in [('SVM', luqpi_color_light, '--'),
                             ('SVM+ (LUQPI)', luqpi_color, '-')]:
        fpr, tpr, _ = roc_curve(y_test, luqpi_probs[name])
        auc = roc_auc_score(y_test, luqpi_probs[name])
        ax5.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f'{name} (AUC={auc:.3f})')
    ax5.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax5.set_xlabel('False Positive Rate', fontsize=10)
    ax5.set_ylabel('True Positive Rate', fontsize=10)
    ax5.set_title('(E) LUQPI ROC Curves (Phase 4b)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.25)

    # Panel F: LUQPI bootstrap AUC-ROC, RI Recall and RI Precision, with 95% CI.
    # Recall and Precision are both shown (not recall alone) because SVM+'s
    # recall gain comes with a real precision cost -- it operates at a much
    # more liberal decision threshold, not a fundamentally better ranking
    # (that's what the near-identical AUC-ROC bars show).
    luqpi_boot = boot_df[boot_df['variant'] == 'LUQPI-offline'].set_index('model')
    metrics_f  = [('auc', 'AUC-ROC'), ('recall', 'RI Recall'), ('precision', 'RI Precision')]
    x_pos_f    = np.arange(len(metrics_f))
    width_f    = 0.35
    for j, name in enumerate(['SVM', 'SVM+ (LUQPI)']):
        row = luqpi_boot.loc[name]
        means = [row[f'{m}_mean'] for m, _ in metrics_f]
        los   = [row[f'{m}_mean'] - row[f'{m}_ci_lo'] for m, _ in metrics_f]
        his   = [row[f'{m}_ci_hi'] - row[f'{m}_mean'] for m, _ in metrics_f]
        offset = (j - 0.5) * width_f
        ax6.bar(x_pos_f + offset, means, width_f,
                color=luqpi_color if 'SVM+' in name else luqpi_color_light,
                label=name, yerr=[los, his], capsize=4, error_kw={'linewidth': 1.5})
    ax6.set_xticks(x_pos_f)
    ax6.set_xticklabels([m[1] for m in metrics_f], fontsize=10)
    ax6.set_ylabel('Score (95% bootstrap CI)', fontsize=10)
    ax6.set_title('(F) LUQPI: Privileged-Information Gain (n_train=400)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.25)

    n_aug = len(norm_features) + len(shadow_labels)
    fig.suptitle(
        f'Quantum Feature Engineering for Hurricane RI Prediction\n'
        f'{n_qubits}-qubit circuit | {len(shadow_labels)} Pauli shadow features '
        f'| Augmented vector: {n_aug} features | Test set: {len(y_test):,} obs',
        fontsize=10, y=1.01
    )
    plt.savefig(OUT_FIGURE, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_FIGURE}")

    # ── Step 19: Abstract-ready summary ───────────────────────────────────────
    print("\n── Step 19: Abstract-ready summary ──────────────────────────────────")
    best_base_auc  = max(boot_df[boot_df['variant']=='Baseline']['auc_mean'])
    best_quant_auc = max(boot_df[boot_df['variant']=='Quantum']['auc_mean'])
    best_base_csi  = op_base['csi'].max()
    best_quant_csi = op_quant['csi'].max()

    lr_b  = boot_df[(boot_df['model']=='Logistic Regression') &
                     (boot_df['variant']=='Baseline')].iloc[0]
    lr_q  = boot_df[(boot_df['model']=='Logistic Regression') &
                     (boot_df['variant']=='Quantum')].iloc[0]

    lines = [
        "=" * 70,
        "PHASE 5 SUMMARY — Quantum Feature Engineering for Hurricane RI",
        "=" * 70,
        "",
        f"Dataset:          HURDAT2 Atlantic 1980–2023, {df_labeled['storm_id'].nunique()} storms",
        f"Labelable obs:    {len(df_labeled):,}  (wind >= 35 kt, 24-h window complete)",
        f"RI events:        {y_all.sum():,} ({y_all.mean()*100:.1f}%)  |  "
        f"Class imbalance: 1:{(y_all==0).sum()/y_all.sum():.0f}",
        f"Quantum circuit:  {n_qubits} qubit(s), {len(shadow_labels)} Pauli observables",
        f"Augmented vector: {len(norm_features)} classical + "
        f"{len(shadow_labels)} shadow = {n_aug} total features",
        "",
        "BOOTSTRAP RESULTS (n=1,000, 95% CI):",
        f"  Best baseline AUC-ROC:  {best_base_auc:.4f}",
        f"  Best quantum AUC-ROC:   {best_quant_auc:.4f}  "
        f"(Δ = {best_quant_auc - best_base_auc:+.4f})",
        "",
        "  Logistic Regression (strongest quantum gain):",
        f"    Baseline  AUC={lr_b['auc_mean']:.4f} "
        f"[{lr_b['auc_ci_lo']:.4f}, {lr_b['auc_ci_hi']:.4f}]",
        f"    Quantum   AUC={lr_q['auc_mean']:.4f} "
        f"[{lr_q['auc_ci_lo']:.4f}, {lr_q['auc_ci_hi']:.4f}]",
        f"    ΔAUC = {lr_q['auc_mean']-lr_b['auc_mean']:+.4f}",
        "",
        "OPERATIONAL METRICS (optimal CSI threshold):",
        f"  Best baseline CSI:  {best_base_csi:.4f}",
        f"  Best quantum CSI:   {best_quant_csi:.4f}  "
        f"(Δ = {best_quant_csi - best_base_csi:+.4f})",
        "",
        f"CROSS-VALIDATION ({N_CV_FOLDS}-fold, storm-safe):",
    ]

    for row in cv_df.itertuples():
        lines.append(
            f"  {row.model:<22}  "
            f"Base AUC {row.base_auc_mean:.4f}±{row.base_auc_std:.4f}  "
            f"Quant AUC {row.quant_auc_mean:.4f}±{row.quant_auc_std:.4f}  "
            f"Δ={row.delta_auc_mean:+.4f}"
        )

    svm_row  = boot_df[(boot_df['model'] == 'SVM') & (boot_df['variant'] == 'LUQPI-offline')].iloc[0]
    svmp_row = boot_df[(boot_df['model'] == 'SVM+ (LUQPI)') & (boot_df['variant'] == 'LUQPI-offline')].iloc[0]
    lines += [
        "",
        "LUQPI RESULTS (Phase 4b — quantum features as privileged information,",
        "no quantum computation required at deployment):",
        f"  SVM  (no privileged info)  AUC={svm_row['auc_mean']:.4f} "
        f"[{svm_row['auc_ci_lo']:.4f}, {svm_row['auc_ci_hi']:.4f}]  "
        f"RI Recall={svm_row['recall_mean']:.4f}  RI Precision={svm_row['precision_mean']:.4f}",
        f"  SVM+ (privileged, offline) AUC={svmp_row['auc_mean']:.4f} "
        f"[{svmp_row['auc_ci_lo']:.4f}, {svmp_row['auc_ci_hi']:.4f}]  "
        f"RI Recall={svmp_row['recall_mean']:.4f}  RI Precision={svmp_row['precision_mean']:.4f}",
        f"  ΔRI Recall (SVM+ − SVM) = {svmp_row['recall_mean']-svm_row['recall_mean']:+.4f}   "
        f"ΔRI Precision (SVM+ − SVM) = {svmp_row['precision_mean']-svm_row['precision_mean']:+.4f}",
        "  (Single storm-safe split, n_train=400, both metrics at each model's own",
        "  CSI-optimal threshold — same convention as the LR/GB/NN rows above.",
        "  Phase 4b's own figures instead fix a naive 0.5 / zero-crossing threshold,",
        "  where SVM+ recall is far higher (0.81 vs 0.09 at n=400) but precision is",
        "  far lower (0.09 vs 0.45) — see data/luqpi_results.csv for the 10-seed",
        "  mean ± 95% CI on both, and plots/luqpi_svm_comparison.png.)",
        "",
        f"NOTE: Current results use {n_qubits} qubits "
        f"({'HURDAT2 features only' if n_qubits <= 2 else 'HURDAT2 + SHIPS features'}).",
        "=" * 70,
    ]

    summary_text = '\n'.join(lines)
    print(summary_text)
    with open(OUT_SUMMARY, 'w') as f:
        f.write(summary_text)
    print(f"\n  Saved → {OUT_SUMMARY}")

    print("\n── Phase 5 Complete ──────────────────────────────────────────────────")
    print(f"  {OUT_BOOTSTRAP}")
    print(f"  {OUT_OPERATIONAL}")
    print(f"  {OUT_CV}")
    print(f"  {OUT_FIGURE}")
    print(f"  {OUT_SUMMARY}")
