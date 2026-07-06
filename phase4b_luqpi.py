"""
Phase 4b: Learning Under Quantum Privileged Information (LUQPI)
Hurricane RI Prediction — Quantum Feature Engineering Pipeline

Based on: Bokov, Kohl, Schmitt & Dunjko, "Machine learning with minimal use
of quantum computers: Provable advantages in Learning Under Quantum
Privileged Information (LUQPI)" (arXiv:2601.22006).

Phase 4 uses the quantum shadow features "online": they are computed and
fed to the classifier both at training AND at test time, so a quantum
device (or its precomputed shadow) is required at deployment. This phase
instead treats the Phase 4 shadow-PCA features as *privileged information*
in the LUQPI/LUPI sense (Vapnik & Vashist, 2009): available only during
training, via the SVM+ algorithm, and never touched at deployment. The
trained model predicts RI using only the ordinary classical features
(HURDAT2 + SHIPS), exactly like the Phase 3 baseline — no quantum
computation is needed to run it.

Because SVM+ solves a dense quadratic program with no shortcuts for scale,
and because the LUQPI paper's own experiments show privileged-information
gains concentrated in the low-data regime, this phase sweeps training-set
size (50-400 storm-safe-sampled observations, same test set as Phase 3/4)
rather than fitting on the full ~2,000-observation training pool.

Inputs:
  data/normalized_dataset.csv   (Phase 2 — classical HURDAT2 + SHIPS features)
  data/augmented_dataset.csv    (Phase 4 — adds PCA-compressed shadow features,
                                  used here ONLY as privileged info during fit)
  data/train_storm_ids.txt      (Phase 3 split — reused exactly)
  data/test_storm_ids.txt
  data/baseline_results.csv     (Phase 3, for comparison)
  data/quantum_results.csv      (Phase 4, for comparison)

Outputs:
  data/luqpi_raw_results.csv    (per-seed, per-size, per-method metrics)
  data/luqpi_results.csv        (mean +/- 95% CI, aggregated over seeds)
  data/phase4b_comparison.csv   (SVM vs SVM+ vs Phase 3 baseline vs Phase 4
                                  quantum-online, with a deployment_requires_
                                  quantum flag for each)
  models/svm_baseline_luqpi.joblib
  models/svm_plus_luqpi.joblib
  plots/luqpi_svm_comparison.png

Algorithm: SVM+ (Vapnik & Vashist, 2009), see luqpi_svm.py.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from luqpi_svm import SVMPlus, median_heuristic_gamma

warnings.filterwarnings("ignore")

NORMALIZED_CSV = "data/normalized_dataset.csv"
AUGMENTED_CSV  = "data/augmented_dataset.csv"
TRAIN_IDS_FILE = "data/train_storm_ids.txt"
TEST_IDS_FILE  = "data/test_storm_ids.txt"
BASELINE_CSV   = "data/baseline_results.csv"
QUANTUM_CSV    = "data/quantum_results.csv"
OUT_RAW        = "data/luqpi_raw_results.csv"
OUT_RESULTS    = "data/luqpi_results.csv"
OUT_COMPARISON = "data/phase4b_comparison.csv"
MODELS_DIR     = "models"
PLOTS_DIR      = "plots"
RANDOM_STATE   = 42

TRAIN_SIZES  = [50, 100, 200, 400]
N_SEEDS      = 10
HPARAM_POOL_SIZE = 240   # pool used once for hyperparameter selection
C_GRID       = [1, 10, 100]
GAMMA_MULT   = [0.25, 1, 4]   # multipliers on the median-heuristic gamma

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate(y_true01, y_score, y_pred01):
    auc_roc  = roc_auc_score(y_true01, y_score)
    avg_prec = average_precision_score(y_true01, y_score)
    report   = classification_report(y_true01, y_pred01,
                                      target_names=['No RI', 'RI'],
                                      output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true01, y_pred01).ravel()
    return {
        'auc_roc':       round(auc_roc, 4),
        'avg_precision': round(avg_prec, 4),
        'ri_recall':     round(report['RI']['recall'], 4),
        'ri_precision':  round(report['RI']['precision'], 4),
        'ri_f1':         round(report['RI']['f1-score'], 4),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, 1.96 * se


if __name__ == "__main__":

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("── Step 1: Loading Phase 2/4 data & Phase 3 split ────────────────────")
    df = pd.read_csv(NORMALIZED_CSV)
    df_labeled = df[df['ri_label'].notna()].copy()
    df_labeled['ri_label'] = df_labeled['ri_label'].astype(int)
    norm_features = [c for c in df_labeled.columns if c.endswith('_norm')]

    aug = pd.read_csv(AUGMENTED_CSV)
    shadow_features = [c for c in aug.columns if c.startswith('shadow_pc')]

    train_storms = np.loadtxt(TRAIN_IDS_FILE, dtype=str)
    test_storms  = np.loadtxt(TEST_IDS_FILE, dtype=str)
    train_mask   = np.isin(df_labeled['storm_id'].values, train_storms)
    test_mask    = np.isin(df_labeled['storm_id'].values, test_storms)

    X_pool = df_labeled.loc[train_mask, norm_features].values
    y_pool = df_labeled.loc[train_mask, 'ri_label'].values
    X_test = df_labeled.loc[test_mask, norm_features].values
    y_test = df_labeled.loc[test_mask, 'ri_label'].values

    # Privileged (shadow-PCA) features for the training pool only — the
    # augmented dataset was built from the exact same Phase 3 split, so row
    # order for split=='train' aligns with X_pool/y_pool by construction
    # (both are filtered from the same labeled/split dataframe in order).
    aug_train = aug[aug['split'] == 'train'].reset_index(drop=True)
    assert len(aug_train) == len(X_pool), "Phase 4 train split does not match Phase 3 split"
    assert np.array_equal(aug_train['ri_label'].values, y_pool), \
        "Row order mismatch between normalized_dataset train rows and augmented_dataset train rows"
    Xstar_pool = aug_train[shadow_features].values

    # Test-set rows with precomputed shadow-PCA features (from Phase 4) — used
    # only below to reproduce Phase 4's quantum-online predictions for the
    # comparison figure, never as privileged info (which is train-only).
    aug_test = aug[aug['split'] == 'test'].reset_index(drop=True)
    assert len(aug_test) == len(X_test), "Phase 4 test split does not match Phase 3 split"
    assert np.array_equal(aug_test['ri_label'].values, y_test), \
        "Row order mismatch between normalized_dataset test rows and augmented_dataset test rows"
    X_test_aug = aug_test[norm_features + shadow_features].values

    print(f"  Classical features ({len(norm_features)}): {norm_features}")
    print(f"  Privileged features ({len(shadow_features)}): shadow PCA components from Phase 4")
    print(f"  Training pool: {len(X_pool):,} obs ({y_pool.mean()*100:.1f}% RI)")
    print(f"  Test set (fixed, classical-only at deployment): {len(X_test):,} obs "
          f"({y_test.mean()*100:.1f}% RI)")

    # ── Step 2: Hyperparameter selection ────────────────────────────────
    # Mirrors the LUQPI paper's own procedure (Sec. VI C): pick (C, gamma)
    # for the decision-space kernel once via plain SVM CV, then fix it and
    # separately pick (C*, gamma*) for the privileged-space kernel via
    # SVM+ CV. Both are reused for every training size and seed below.
    print(f"\n── Step 2: Hyperparameter selection (pool={HPARAM_POOL_SIZE}) ────────")
    rng_hp = np.random.default_rng(RANDOM_STATE)
    hp_idx, _ = train_test_split(
        np.arange(len(X_pool)), train_size=HPARAM_POOL_SIZE,
        stratify=y_pool, random_state=RANDOM_STATE
    )
    X_hp, y_hp, Xstar_hp = X_pool[hp_idx], y_pool[hp_idx], Xstar_pool[hp_idx]

    gamma0  = median_heuristic_gamma(X_hp, random_state=RANDOM_STATE)
    gamma0s = median_heuristic_gamma(Xstar_hp, random_state=RANDOM_STATE)
    print(f"  Median-heuristic gamma (decision space):   {gamma0:.4f}")
    print(f"  Median-heuristic gamma (privileged space):  {gamma0s:.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_auc, best_C, best_gamma = -1, C_GRID[0], gamma0
    for C in C_GRID:
        for gm in GAMMA_MULT:
            gamma = gamma0 * gm
            fold_aucs = []
            for tr_idx, va_idx in skf.split(X_hp, y_hp):
                clf = SVC(C=C, gamma=gamma, kernel='rbf').fit(X_hp[tr_idx], y_hp[tr_idx])
                fold_aucs.append(roc_auc_score(y_hp[va_idx], clf.decision_function(X_hp[va_idx])))
            auc = np.mean(fold_aucs)
            if auc > best_auc:
                best_auc, best_C, best_gamma = auc, C, gamma
    print(f"  Selected decision-space (C, gamma) = ({best_C}, {best_gamma:.4f})  "
          f"[CV AUC-ROC={best_auc:.4f}]")

    best_auc_s, best_Cs, best_gammas = -1, C_GRID[0], gamma0s
    for Cs in C_GRID:
        for gm in GAMMA_MULT:
            gammas = gamma0s * gm
            fold_aucs = []
            for tr_idx, va_idx in skf.split(X_hp, y_hp):
                y_pm = np.where(y_hp[tr_idx] == 1, 1.0, -1.0)
                m = SVMPlus(C=best_C, gamma=best_gamma, C_star=Cs, gamma_star=gammas)
                m.fit(X_hp[tr_idx], Xstar_hp[tr_idx], y_pm)
                fold_aucs.append(roc_auc_score(y_hp[va_idx], m.decision_function(X_hp[va_idx])))
            auc = np.mean(fold_aucs)
            if auc > best_auc_s:
                best_auc_s, best_Cs, best_gammas = auc, Cs, gammas
    print(f"  Selected privileged-space (C*, gamma*) = ({best_Cs}, {best_gammas:.4f})  "
          f"[CV AUC-ROC={best_auc_s:.4f}]")

    # ── Step 3: Training-size sweep ─────────────────────────────────────
    print(f"\n── Step 3: Training-size sweep {TRAIN_SIZES}, {N_SEEDS} seeds ──────────")
    raw_rows = []
    for size in TRAIN_SIZES:
        for seed in range(N_SEEDS):
            idx, _ = train_test_split(
                np.arange(len(X_pool)), train_size=size,
                stratify=y_pool, random_state=1000 * size + seed
            )
            Xs, ys, Xstars = X_pool[idx], y_pool[idx], Xstar_pool[idx]
            y_pm = np.where(ys == 1, 1.0, -1.0)

            # Plain SVM — no privileged information (classical features only)
            svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf').fit(Xs, ys)
            score = svm.decision_function(X_test)
            pred  = svm.predict(X_test)
            m = evaluate(y_test, score, pred)
            m.update(method='SVM', train_size=size, seed=seed, n_pos_train=int(ys.sum()))
            raw_rows.append(m)

            # SVM+ (LUQPI) — privileged shadow features used only for fit()
            svmp = SVMPlus(C=best_C, gamma=best_gamma, C_star=best_Cs, gamma_star=best_gammas)
            svmp.fit(Xs, Xstars, y_pm)
            score = svmp.decision_function(X_test)
            pred01 = (svmp.predict(X_test) == 1).astype(int)
            m = evaluate(y_test, score, pred01)
            m.update(method='SVM+ (LUQPI)', train_size=size, seed=seed, n_pos_train=int(ys.sum()))
            raw_rows.append(m)

        print(f"  train_size={size:4d}  done ({N_SEEDS} seeds x 2 methods)")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(OUT_RAW, index=False)
    print(f"\n  Raw per-seed results saved -> {OUT_RAW}")

    # ── Step 4: Aggregate ────────────────────────────────────────────────
    print(f"\n── Step 4: Aggregating across seeds ───────────────────────────────")
    agg_rows = []
    for (method, size), g in raw_df.groupby(['method', 'train_size']):
        row = {'method': method, 'train_size': size, 'n_seeds': len(g)}
        for metric in ['auc_roc', 'avg_precision', 'ri_recall', 'ri_precision', 'ri_f1']:
            m, ci = mean_ci95(g[metric].values)
            row[f'{metric}_mean'] = round(m, 4)
            row[f'{metric}_ci95'] = round(ci, 4)
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values(['train_size', 'method'])
    agg_df.to_csv(OUT_RESULTS, index=False)
    print(agg_df[['method', 'train_size', 'auc_roc_mean', 'auc_roc_ci95',
                   'ri_recall_mean', 'ri_recall_ci95']].to_string(index=False))
    print(f"\n  Aggregated results saved -> {OUT_RESULTS}")

    # ── Step 5: Comparison vs Phase 3 baseline & Phase 4 quantum-online ──
    print(f"\n── Step 5: Comparison across all pipeline stages ────────────────────")
    baseline_df = pd.read_csv(BASELINE_CSV)
    quantum_df  = pd.read_csv(QUANTUM_CSV)

    max_size = max(TRAIN_SIZES)
    luqpi_at_max = agg_df[agg_df['train_size'] == max_size]

    comparison_rows = []
    for _, r in baseline_df.iterrows():
        comparison_rows.append({
            'stage': 'Phase 3 (classical baseline)', 'model': r['model'],
            'n_train': r['n_train'], 'auc_roc': r['auc_roc'], 'ri_recall': r['ri_recall'],
            'deployment_requires_quantum': False,
        })
    for _, r in quantum_df.iterrows():
        comparison_rows.append({
            'stage': 'Phase 4 (quantum-online)', 'model': r['model'],
            'n_train': int(baseline_df['n_train'].iloc[0]), 'auc_roc': r['auc_roc'],
            'ri_recall': r['ri_recall'], 'deployment_requires_quantum': True,
        })
    for _, r in luqpi_at_max.iterrows():
        comparison_rows.append({
            'stage': 'Phase 4b (LUQPI, offline)', 'model': r['method'],
            'n_train': int(r['train_size']), 'auc_roc': r['auc_roc_mean'],
            'ri_recall': r['ri_recall_mean'], 'deployment_requires_quantum': False,
        })

    compare_df = pd.DataFrame(comparison_rows)
    compare_df.to_csv(OUT_COMPARISON, index=False)
    print(compare_df.to_string(index=False))
    print(f"\n  Comparison table saved -> {OUT_COMPARISON}")
    print(f"  Note: Phase 4b matches Phase 4's 'no quantum at deployment' property")
    print(f"  of Phase 3, while (unlike Phase 3) still benefiting from quantum shadow")
    print(f"  features — used only as privileged information during training.")

    # ── Step 6: Save final models (largest sweep size, RANDOM_STATE seed) ─
    print(f"\n── Step 6: Saving final models (train_size={max_size}) ─────────────")
    idx, _ = train_test_split(
        np.arange(len(X_pool)), train_size=max_size,
        stratify=y_pool, random_state=1000 * max_size + RANDOM_STATE
    )
    Xs, ys, Xstars = X_pool[idx], y_pool[idx], Xstar_pool[idx]
    y_pm = np.where(ys == 1, 1.0, -1.0)

    final_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf', probability=True).fit(Xs, ys)
    joblib.dump(final_svm, os.path.join(MODELS_DIR, 'svm_baseline_luqpi.joblib'))

    final_svmp = SVMPlus(C=best_C, gamma=best_gamma, C_star=best_Cs, gamma_star=best_gammas)
    final_svmp.fit(Xs, Xstars, y_pm)
    joblib.dump(final_svmp, os.path.join(MODELS_DIR, 'svm_plus_luqpi.joblib'))
    print(f"  Saved -> models/svm_baseline_luqpi.joblib, models/svm_plus_luqpi.joblib")

    # ── Step 7: Plot ──────────────────────────────────────────────────────
    # 4 panels: AUC-ROC, Recall and Precision vs. training size (same n for
    # both methods at every point, so this comparison was already fair), plus
    # a Delta-AUC bar. Recall and Precision are shown side by side because
    # SVM+'s recall gain comes with a real precision cost (it operates at a
    # far more liberal decision threshold) -- showing recall alone overstates
    # the win.
    print(f"\n── Step 7: Generating comparison plot ────────────────────────────")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    colors = {'SVM': 'steelblue', 'SVM+ (LUQPI)': 'darkorchid'}
    for metric, ax, ylabel in [
        ('auc_roc', axes[0, 0], 'AUC-ROC'),
        ('ri_recall', axes[0, 1], 'RI Recall'),
        ('ri_precision', axes[1, 0], 'RI Precision'),
    ]:
        for method in ['SVM', 'SVM+ (LUQPI)']:
            sub = agg_df[agg_df['method'] == method].sort_values('train_size')
            ax.errorbar(sub['train_size'], sub[f'{metric}_mean'], yerr=sub[f'{metric}_ci95'],
                        label=method, color=colors[method], marker='o', capsize=3, lw=2)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. Training Size')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    svm_auc  = agg_df[agg_df['method'] == 'SVM'].sort_values('train_size')['auc_roc_mean'].values
    plus_auc = agg_df[agg_df['method'] == 'SVM+ (LUQPI)'].sort_values('train_size')['auc_roc_mean'].values
    sizes    = sorted(agg_df['train_size'].unique())
    delta = plus_auc - svm_auc
    bar_colors = ['forestgreen' if d >= 0 else 'firebrick' for d in delta]
    ax.bar([str(s) for s in sizes], delta, color=bar_colors, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Delta AUC-ROC (SVM+ minus SVM)')
    ax.set_title('LUQPI Privileged-Information Gain (ranking quality)')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Phase 4b: LUQPI (SVM+) vs. Classical SVM — quantum shadow features used\n'
        'as privileged information at training only; deployment is fully classical.\n'
        'Both methods trained/evaluated on identical data at every point (this comparison is apples-to-apples).',
        fontsize=10
    )
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'luqpi_svm_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved -> {plot_path}")

    # ── Step 8: LUQPI vs Quantum-Online vs Baseline figure (apples-to-apples) ─
    # All four models below are fit on the IDENTICAL n=max_size subsample
    # (the same Xs/ys/Xstars used for the final SVM/SVM+ models in Step 6)
    # and scored on the identical fixed test set. This isolates "how is
    # quantum information used" (never / online / offline-privileged) from
    # "how much training data is available" -- comparing against Phase 3/4's
    # full n_train=2004 models would conflate the two and make Phase 4b look
    # worse than it is purely because of the (deliberate, QP-tractability)
    # smaller training set.
    print(f"\n── Step 8: LUQPI vs Quantum vs Baseline comparison plot (n_train={max_size} for all four) ──")
    Xs_aug = np.hstack([Xs, Xstars])

    lr_classical = LogisticRegression(class_weight='balanced', max_iter=1000,
                                       random_state=RANDOM_STATE).fit(Xs, ys)
    lr_quantum   = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000,
                                       random_state=RANDOM_STATE).fit(Xs_aug, ys)
    base_score  = lr_classical.predict_proba(X_test)[:, 1]
    quant_score = lr_quantum.predict_proba(X_test_aug)[:, 1]
    svm_score   = final_svm.predict_proba(X_test)[:, 1]
    svmp_score  = final_svmp.decision_function(X_test)

    series = [
        ('Classical LR\n(no quantum)',          base_score,  'steelblue',  '--'),
        ('Quantum-online LR',                   quant_score, 'darkorange', '--'),
        ('LUQPI: SVM\n(no privileged info)',    svm_score,   'gray',       '-'),
        ('LUQPI: SVM+\n(privileged, offline)',  svmp_score,  'darkorchid', '-'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    for label, score, color, ls in series:
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = roc_auc_score(y_test, score)
        ax.plot(fpr, tpr, label=f'{label.splitlines()[0]} (AUC={auc:.3f})', color=color, lw=2, ls=ls)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    labels_metric = [s[0] for s in series]
    colors_metric = [s[2] for s in series]

    ax = axes[1]
    aucs = [roc_auc_score(y_test, s[1]) for s in series]
    ax.bar(range(len(series)), aucs, color=colors_metric, alpha=0.85)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(labels_metric, fontsize=9)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC (ranking quality)')
    ax.grid(axis='y', alpha=0.3)

    # Recall AND precision, grouped, so the trade-off is visible in the
    # figure itself -- SVM+'s recall gain comes from operating at a much
    # more liberal threshold, not from better ranking (see the AUC panel).
    ax = axes[2]
    recalls, precisions = [], []
    for label, score, color, ls in series:
        # svmp_score is a raw decision_function (centered at 0); the other
        # three are predict_proba outputs in [0, 1] (natural 0.5 threshold).
        thresh = 0.0 if 'SVM+' in label else 0.5
        pred = (score >= thresh).astype(int)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        recalls.append(report['1']['recall'])
        precisions.append(report['1']['precision'])
    x = np.arange(len(series))
    width = 0.35
    ax.bar(x - width / 2, recalls, width, color=colors_metric, alpha=0.9, label='Recall')
    ax.bar(x + width / 2, precisions, width, color=colors_metric, alpha=0.5, hatch='//', label='Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_metric, fontsize=9)
    for i, (r, p) in enumerate(zip(recalls, precisions)):
        ax.text(i - width / 2, r + 0.01, f'{r:.2f}', ha='center', fontsize=8)
        ax.text(i + width / 2, p + 0.01, f'{p:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('RI Recall vs. Precision (@ default threshold)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f'Baseline vs. Quantum-Online vs. LUQPI-Offline — all four models trained on the\n'
        f'identical n_train={max_size} subsample and evaluated on the identical test set (n={len(y_test)})',
        fontsize=10
    )
    for_reference = (f"For reference, Phase 3/4's full-data (n_train={int(baseline_df['n_train'].iloc[0])}) models reach "
                      f"AUC-ROC up to {baseline_df['auc_roc'].max():.3f} (baseline) / "
                      f"{quantum_df['auc_roc'].max():.3f} (quantum-online) -- see data/baseline_results.csv, "
                      f"data/quantum_results.csv. Only 'Quantum-online' needs quantum computation at deployment.")
    fig.text(0.5, -0.04, for_reference, ha='center', fontsize=8, style='italic', color='dimgray')
    plt.tight_layout()
    three_way_path = os.path.join(PLOTS_DIR, 'luqpi_vs_quantum_vs_baseline.png')
    plt.savefig(three_way_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved -> {three_way_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n── Phase 4b Summary ─────────────────────────────────────────────────")
    print(f"  Privileged features: {len(shadow_features)} shadow PCA components (Phase 4)")
    print(f"  Training sizes swept: {TRAIN_SIZES}  ({N_SEEDS} seeds each)")
    best_row = agg_df.loc[agg_df[agg_df['method'] == 'SVM+ (LUQPI)']['auc_roc_mean'].idxmax()] \
        if (agg_df['method'] == 'SVM+ (LUQPI)').any() else None
    if best_row is not None:
        print(f"  Best SVM+ (LUQPI): train_size={int(best_row['train_size'])}, "
              f"AUC-ROC={best_row['auc_roc_mean']:.4f} +/- {best_row['auc_roc_ci95']:.4f}")
    print(f"  Outputs: {OUT_RAW}, {OUT_RESULTS}, {OUT_COMPARISON}, {plot_path}, {three_way_path}")