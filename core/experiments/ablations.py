import copy
import time

import numpy as np

from core.fl.utils import set_seed

ABLATION_VARIANTS = {
    'Full KIQFL': {},
    'No Quantum Walk': {'disable_quantum_walk': True},
    'No KAN': {'disable_kan': True},
    'No Sporadic': {'disable_sporadic': True},
    'No Indirect Aggregation': {'use_indirect_aggregation': False},
    'No QAOA': {'disable_qaoa': True},
    'No NISQ Noise': {'disable_noise': True},
}

def create_ablation_configs(base_config, variants=None):
    """
    Generate a dict of {variant_name: config} for ablation study.
    Each variant overrides specific flags in base_config.
    """
    if variants is None:
        variants = ABLATION_VARIANTS
    configs = {}
    for name, overrides in variants.items():
        cfg = copy.deepcopy(base_config)
        cfg.update(overrides)
        configs[name] = cfg
    return configs

def run_ablation_study(run_kiqfl_fn, trainset, testset, base_config,
                       num_seeds=10, base_seed=42, variants=None):
    """
    Run a full ablation study: every variant × num_seeds independent runs.

    Args:
        run_kiqfl_fn: The run_kiqfl generator function from core.py.
        trainset, testset: Datasets.
        base_config: Base experiment config dict.
        num_seeds: Number of independent seeds.
        base_seed: Starting seed value.
        variants: Dict of {name: override_dict}. Defaults to ABLATION_VARIANTS.

    Returns:
        dict: {variant_name: {'accs': [...], 'losses': [...],
               'mean_acc': float, 'std_acc': float, ...}}
    """
    import time
    if variants is None:
        variants = ABLATION_VARIANTS
    seeds = [base_seed + i * 111 for i in range(num_seeds)]
    configs = create_ablation_configs(base_config, variants)
    results = {}

    for variant_name, cfg in configs.items():
        print(f"\n--- Ablation: {variant_name} ---")
        accs, losses = [], []
        for si, s in enumerate(seeds):
            cfg_run = copy.deepcopy(cfg)
            cfg_run['seed'] = s
            set_seed(s)
            t0 = time.time()
            run_results = list(run_kiqfl_fn(trainset, testset, cfg_run, yield_results=True))
            final = run_results[-1]
            accs.append(final['accuracy'])
            losses.append(final['loss'])
            elapsed = time.time() - t0
            print(f"  Seed {si+1}/{num_seeds} (s={s}): "
                  f"acc={final['accuracy']:.4f}, loss={final['loss']:.4f} [{elapsed:.1f}s]")

        results[variant_name] = {
            'accs': accs,
            'losses': losses,
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'best_acc': float(np.max(accs)),
            'worst_acc': float(np.min(accs)),
        }
        print(f"  => {variant_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    return results


# ========================================================================================
# 14. MULTI-RUN STATISTICAL ENGINE
# ========================================================================================

def run_multi_seed(run_fn, trainset, testset, base_config,
                   num_seeds=10, base_seed=42):
    """
    Run an algorithm num_seeds times with different seeds.

    Args:
        run_fn: Generator function (run_kiqfl, run_fedavg, etc.).
        trainset, testset: Datasets.
        base_config: Config dict.
        num_seeds: Number of independent runs.
        base_seed: Starting seed.

    Returns:
        metrics: dict with 'accuracy', 'loss', 'fidelity' arrays (num_seeds × num_epochs).
        summary: dict with 'mean_acc', 'std_acc', 'mean_loss', etc.
        all_runs: list of per-seed result lists.
    """
    import time
    seeds = [base_seed + i * 111 for i in range(num_seeds)]
    all_runs = []

    for si, s in enumerate(seeds):
        cfg = copy.deepcopy(base_config)
        cfg['seed'] = s
        set_seed(s)
        t0 = time.time()
        run_results = list(run_fn(trainset, testset, cfg, yield_results=True))
        elapsed = time.time() - t0
        all_runs.append(run_results)
        final = run_results[-1]
        method = final.get('method', '?')
        print(f"  [{method}] Seed {si+1}/{num_seeds} (s={s}): "
              f"acc={final['accuracy']:.4f}, loss={final['loss']:.4f} [{elapsed:.1f}s]")

    num_epochs = min(len(r) for r in all_runs)
    metric_keys = ['accuracy', 'loss', 'fidelity', 'f1_macro', 'jain_fairness',
                   'comm_efficiency', 'balanced_accuracy']
    metrics = {}
    for k in metric_keys:
        default = 0.0 if k in ('fidelity',) else np.nan
        metrics[k] = np.array([
            [run[e].get(k, default) for e in range(num_epochs)]
            for run in all_runs
        ], dtype=np.float64)

    final_accs = metrics['accuracy'][:, -1]
    final_losses = metrics['loss'][:, -1]
    final_f1 = metrics['f1_macro'][:, -1]
    from evaluation_metrics import bootstrap_ci_95
    mean_acc, ci_lo, ci_hi = bootstrap_ci_95(final_accs)
    summary = {
        'mean_acc': float(np.mean(final_accs)),
        'std_acc': float(np.std(final_accs)),
        'mean_loss': float(np.mean(final_losses)),
        'std_loss': float(np.std(final_losses)),
        'mean_f1_macro': float(np.nanmean(final_f1)),
        'best_acc': float(np.max(final_accs)),
        'worst_acc': float(np.min(final_accs)),
        'acc_ci_low': ci_lo,
        'acc_ci_high': ci_hi,
    }
    return metrics, summary, all_runs


# ========================================================================================
# 15. STATISTICAL SIGNIFICANCE TESTS
# ========================================================================================

def cohens_d(x, y):
    """Compute Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 +
                          (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)

def compute_statistical_significance(kiqfl_accs, baseline_accs_dict,
                                     alpha=0.05, paired=False):
    """
    Pairwise significance: t-test + Wilcoxon (paired) + Cohen's d + bootstrap CI.
    Requires ≥2 samples per group; otherwise returns [].
    """
    from scipy.stats import ttest_ind, ttest_rel, wilcoxon
    from evaluation_metrics import bootstrap_ci_95

    kiqfl_accs = np.asarray(kiqfl_accs, dtype=np.float64)
    if len(kiqfl_accs) < 2:
        return []

    k = len(baseline_accs_dict)
    corrected_alpha = alpha / max(1, k)
    results = []
    for name, bl_accs in baseline_accs_dict.items():
        bl_accs = np.asarray(bl_accs, dtype=np.float64)
        if len(bl_accs) < 2:
            continue
        if paired and len(kiqfl_accs) != len(bl_accs):
            continue
        if paired:
            t_stat, p_val = ttest_rel(kiqfl_accs, bl_accs)
            try:
                w_stat, p_wilcoxon = wilcoxon(kiqfl_accs, bl_accs)
            except Exception:
                w_stat, p_wilcoxon = 0.0, 1.0
        else:
            t_stat, p_val = ttest_ind(kiqfl_accs, bl_accs)
            w_stat, p_wilcoxon = 0.0, 1.0
        d = cohens_d(kiqfl_accs, bl_accs)
        mean_k, ci_lo, ci_hi = bootstrap_ci_95(kiqfl_accs)
        results.append({
            'comparison': f'KIQFL vs {name}',
            'test': 'paired_t' if paired else 'independent_t',
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'wilcoxon_stat': float(w_stat),
            'wilcoxon_p': float(p_wilcoxon),
            'cohens_d': d,
            'kiqfl_mean': mean_k,
            'kiqfl_ci_low': ci_lo,
            'kiqfl_ci_high': ci_hi,
            'significant': 'YES' if p_val < corrected_alpha else 'NO',
            'wilcoxon_significant': 'YES' if p_wilcoxon < corrected_alpha else 'NO',
            'corrected_alpha': corrected_alpha,
        })
    return results

def print_statistical_table(stat_results):
    """Pretty-print a table of statistical test results."""
    print(f"{'Comparison':35s} {'t-stat':>8s} {'p-value':>10s} "
          f"{'Cohen d':>8s} {'Sig?':>6s}")
    print("-" * 70)
    for r in stat_results:
        print(f"  {r['comparison']:33s} {r['t_statistic']:8.3f} "
              f"{r['p_value']:10.6f} {r['cohens_d']:8.3f}   {r['significant']}")
    print(f"\n  Bonferroni-corrected α = {stat_results[0]['corrected_alpha']:.4f}")

