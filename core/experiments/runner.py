"""Unified experiment runner for KIQFL suites."""

import argparse
import copy
import csv
import os
import shutil
import time
from datetime import datetime

import numpy as np

from core.fl.data import load_dataset
from core.fl.algorithms import run_kiqfl, run_fedavg, run_fqngd, run_centralized
from core.fl.utils import set_seed
from core.fl.config import default_config, apply_comparison_defaults
from core.experiments.ablations import (
    ABLATION_VARIANTS, create_ablation_configs, compute_statistical_significance,
)

RESULT_DIR = 'result'
CSV_PATH = os.path.join(RESULT_DIR, 'experiments_log.csv')

# Thesis-quality defaults — defense / ~24h profile (gaming GPU).
# User --epochs overrides all. Locked: fisher parameter_shift + layer_block, 8 qubits.
SUITE_EPOCH_DEFAULTS = {
    'main': 60,
    'ablation': 35,
    'noise': 35,
    'fashion': 60,
    'scalability': 35,
    'noniid': 35,
}

ALGO_MAP = {
    'KIQFL': run_kiqfl,
    'FedAvg': run_fedavg,
    'FQNGD': run_fqngd,
    'Centralized': run_centralized,
}

CSV_COLUMNS = [
    'timestamp', 'suite', 'algorithm', 'seed', 'dataset', 'num_clients',
    'global_epochs', 'final_accuracy', 'final_loss', 'final_fidelity',
    'f1_macro', 'balanced_accuracy', 'jain_fairness', 'comm_efficiency',
    'convergence_epoch', 'wall_time_s', 'comm_full_KB', 'comm_state_KB',
    'comm_quantum_subnet_KB', 'comm_effective_KB', 'cumulative_comm_KB',
    'gate_count', 'sim_time_per_round_s', 'hilbert_dim', 'mean_acc', 'std_acc',
    'acc_ci_low', 'acc_ci_high', 'p_value_vs_fedavg', 'aggregation_mode',
    'comparison_mode', 'fisher_method', 'nisq_error_rate', 'data_fraction',
    'non_iid_alpha', 'variant',
]


def ensure_result_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def _csv_header_matches(path, expected=CSV_COLUMNS):
    """Return True if existing CSV header matches the current schema."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return True
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return True
    return header == list(expected)


def rotate_csv_if_schema_mismatch(path=CSV_PATH):
    """
    If experiments_log.csv has an outdated/mismatched header, rename it so new
    runs write a clean file. Prevents column shift when appending.
    """
    if not os.path.exists(path) or _csv_header_matches(path):
        return None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = os.path.join(
        RESULT_DIR, f'experiments_log_schema_mismatch_{stamp}.csv')
    os.rename(path, backup)
    print(f"[warn] CSV schema mismatch — archived old log to {backup}")
    return backup


def snapshot_csv(tag):
    """Copy current CSV after a suite finishes (recover if later suite fails)."""
    if not os.path.exists(CSV_PATH):
        return
    ensure_result_dir()
    dest = os.path.join(RESULT_DIR, f'experiments_log_after_{tag}.csv')
    shutil.copy2(CSV_PATH, dest)
    print(f"[snapshot] {dest}")


def flatten_run_result(algo, seed, suite, cfg, results, wall_time, extra=None):
    if not results:
        return None
    final = results[-1]
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'suite': suite,
        'algorithm': algo,
        'seed': seed,
        'dataset': cfg.get('dataset_name', 'mnist'),
        'num_clients': cfg.get('num_clients', 10),
        'global_epochs': cfg.get('global_epochs', 50),
        'final_accuracy': final.get('accuracy', 0.0),
        'final_loss': final.get('loss', 0.0),
        'final_fidelity': final.get('fidelity', 0.0),
        'f1_macro': final.get('f1_macro', 0.0),
        'balanced_accuracy': final.get('balanced_accuracy', 0.0),
        'jain_fairness': final.get('jain_fairness', ''),
        'comm_efficiency': final.get('comm_efficiency', 0.0),
        'cumulative_comm_KB': final.get('cumulative_comm_KB', 0.0),
        'convergence_epoch': final.get('convergence_epoch'),
        'wall_time_s': wall_time,
        'comm_full_KB': final.get('comm_full_KB', 0.0),
        'comm_state_KB': final.get('comm_state_KB', 0.0),
        'comm_quantum_subnet_KB': final.get('comm_quantum_subnet_KB', 0.0),
        'comm_effective_KB': final.get('comm_effective_KB', 0.0),
        'gate_count': final.get('gate_count', 0),
        'sim_time_per_round_s': final.get('sim_time_per_round_s', 0.0),
        'hilbert_dim': final.get('hilbert_dim', 0),
        'aggregation_mode': cfg.get('aggregation_mode', ''),
        'comparison_mode': cfg.get('comparison_mode', 'fair'),
        'fisher_method': cfg.get('fisher_method', 'parameter_shift'),
        'nisq_error_rate': cfg.get('nisq_error_rate', 0.01),
        'data_fraction': cfg.get('data_fraction', 0.5),
        'non_iid_alpha': cfg.get('non_iid_alpha', 0.5),
        'variant': (extra or {}).get('variant', ''),
        'mean_acc': '',
        'std_acc': '',
        'acc_ci_low': '',
        'acc_ci_high': '',
        'p_value_vs_fedavg': '',
    }
    return row


def append_rows(rows, path=CSV_PATH):
    ensure_result_dir()
    rotate_csv_if_schema_mismatch(path)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def completed_run_keys(path=CSV_PATH):
    """
    Keys already finished in CSV so long runs can resume after Ctrl+C.
    Key: (suite, algorithm, seed, dataset, variant)
    """
    keys = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return keys
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((
                    str(row.get('suite', '')),
                    str(row.get('algorithm', '')),
                    str(row.get('seed', '')),
                    str(row.get('dataset', '')),
                    str(row.get('variant', '') or ''),
                ))
    except Exception as exc:
        print(f"[warn] could not read resume keys from CSV: {exc}")
    return keys


def is_run_done(done, suite, algo, seed, dataset, variant=''):
    return (
        str(suite), str(algo), str(seed), str(dataset), str(variant or '')
    ) in done


def run_algorithm_once(algo, trainset, testset, cfg):
    func = ALGO_MAP[algo]
    t0 = time.time()
    results = list(func(trainset, testset, cfg, yield_results=True))
    return results, time.time() - t0


def build_base_config(args):
    cfg = default_config(
        num_clients=args.num_clients,
        global_epochs=args.epochs,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        nisq_error_rate=args.nisq_error,
        aggregation_mode=args.aggregation_mode,
        comparison_mode=args.comparison_mode,
        data_fraction=args.data_fraction,
        non_iid_alpha=args.non_iid_alpha,
        batch_size=args.batch_size,
        fisher_period=args.fisher_period,
    )
    cfg['dataset_name'] = args.dataset
    cfg['data_fraction'] = args.data_fraction
    cfg['non_iid_alpha'] = args.non_iid_alpha
    cfg['resume_from_checkpoint'] = True
    # Locked thesis constraints (never overridden by CLI soft defaults)
    cfg['fisher_method'] = 'parameter_shift'
    cfg['fisher_structure'] = 'layer_block'
    cfg['fqngd_aggregation'] = 'natural_grad'
    cfg['use_hybrid_qnn'] = True
    return apply_comparison_defaults(cfg)


def run_main_comparison(args):
    rows = []
    acc_by_algo = {a: [] for a in (
        ['KIQFL', 'FedAvg'] if args.quick else ['KIQFL', 'FedAvg', 'FQNGD', 'Centralized']
    )}
    cfg_base = build_base_config(args)
    trainset, testset, ic, isz, nc = load_dataset(args.dataset, args.data_fraction)
    dataset_name = args.dataset
    done = completed_run_keys()

    algos = list(acc_by_algo.keys())
    seeds = [42] if args.quick else [42 + i * 111 for i in range(args.num_seeds)]
    seed_set = {str(s) for s in seeds}

    for algo in algos:
        for seed in seeds:
            if is_run_done(done, 'main', algo, seed, dataset_name):
                print(f"[main] SKIP {algo} seed={seed} (already in CSV)")
                continue
            cfg = copy.deepcopy(cfg_base)
            cfg.update({'seed': seed, 'input_channels': ic, 'image_size': isz, 'num_classes': nc})
            cfg['checkpoint_suite'] = 'main'
            cfg['checkpoint_variant'] = ''
            set_seed(seed)
            results, wall = run_algorithm_once(algo, trainset, testset, cfg)
            row = flatten_run_result(algo, seed, 'main', cfg, results, wall)
            if row:
                rows.append(row)
                append_rows([row])  # checkpoint after each seed (Ctrl+C safe)
                done.add(('main', str(algo), str(seed), str(dataset_name), ''))
            print(f"[main] {algo} seed={seed} acc={row['final_accuracy']:.4f} time={wall:.1f}s")

    # Rebuild accuracies from CSV for stats (includes resumed/skipped runs)
    rebuilt = {a: {} for a in algos}  # seed -> acc
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', newline='') as f:
            for r in csv.DictReader(f):
                if r.get('suite') != 'main' or r.get('dataset') != str(dataset_name):
                    continue
                algo = r.get('algorithm')
                seed = str(r.get('seed', ''))
                if algo not in rebuilt or seed not in seed_set:
                    continue
                try:
                    rebuilt[algo][seed] = float(r['final_accuracy'])
                except (TypeError, ValueError):
                    continue
    for algo in algos:
        acc_by_algo[algo] = [rebuilt[algo][str(s)] for s in seeds if str(s) in rebuilt[algo]]

    n_kiqfl = len(acc_by_algo.get('KIQFL', []))
    n_fed = len(acc_by_algo.get('FedAvg', []))
    if n_kiqfl >= 2 and n_fed >= 2:
        stats = compute_statistical_significance(
            acc_by_algo['KIQFL'], {'FedAvg': acc_by_algo['FedAvg']}, paired=True)
        if stats:
            pval = stats[0]['p_value']
            print(f"[main] paired t-test KIQFL vs FedAvg p={pval:.4g} "
                  f"(n_kiqfl={n_kiqfl}, n_fed={n_fed})")
    elif n_kiqfl >= 1 or n_fed >= 1:
        print("[main] Skipping paired t-test / CI (need ≥2 seeds per algorithm)")

    return rows


def run_ablation_suite(args):
    """Ablation uses 1 seed by default (defense). Pass --num-seeds N to repeat."""
    rows = []
    cfg_base = build_base_config(args)
    trainset, testset, ic, isz, nc = load_dataset(args.dataset, args.data_fraction)
    variants = create_ablation_configs(cfg_base, ABLATION_VARIANTS)
    n_seeds = getattr(args, 'ablation_seeds', None)
    if n_seeds is None:
        n_seeds = 1
    seeds = [42 + i * 111 for i in range(n_seeds)]
    done = completed_run_keys()
    for name, cfg_v in variants.items():
        for seed in seeds:
            if is_run_done(done, 'ablation', 'KIQFL', seed, args.dataset, variant=name):
                print(f"[ablation] SKIP {name} seed={seed}")
                continue
            cfg = copy.deepcopy(cfg_v)
            cfg.update({'seed': seed, 'input_channels': ic, 'image_size': isz, 'num_classes': nc})
            cfg['checkpoint_suite'] = 'ablation'
            cfg['checkpoint_variant'] = name
            set_seed(seed)
            results, wall = run_algorithm_once('KIQFL', trainset, testset, cfg)
            row = flatten_run_result('KIQFL', seed, 'ablation', cfg, results, wall,
                                    extra={'variant': name})
            if row:
                rows.append(row)
                append_rows([row])
                done.add(('ablation', 'KIQFL', str(seed), str(args.dataset), str(name)))
            print(f"[ablation] {name} seed={seed} acc={row['final_accuracy']:.4f}")
    return rows


def run_noise_sweep(args):
    """Compare KIQFL vs FedAvg across NISQ noise rates (1 seed; defense)."""
    rows = []
    noises = [0.0, 0.01, 0.03, 0.05, 0.10]
    algos = ['KIQFL', 'FedAvg'] if not args.quick else ['KIQFL']
    trainset, testset, ic, isz, nc = load_dataset(args.dataset, args.data_fraction)
    done = completed_run_keys()
    for noise in noises:
        for algo in algos:
            suite = f'noise_{noise}'
            if is_run_done(done, suite, algo, 42, args.dataset):
                print(f"[noise] SKIP {algo} ε={noise}")
                continue
            cfg = build_base_config(args)
            cfg['nisq_error_rate'] = noise
            cfg.update({'seed': 42, 'input_channels': ic, 'image_size': isz, 'num_classes': nc})
            cfg['checkpoint_suite'] = suite
            cfg['checkpoint_variant'] = ''
            cfg['checkpoint_tag'] = f'eps{noise}'
            set_seed(42)
            results, wall = run_algorithm_once(algo, trainset, testset, cfg)
            row = flatten_run_result(algo, 42, suite, cfg, results, wall)
            if row:
                rows.append(row)
                append_rows([row])
                done.add((suite, str(algo), '42', str(args.dataset), ''))
            print(f"[noise] {algo} ε={noise} acc={row['final_accuracy']:.4f}")
    return rows


def run_scalability_suite(args):
    """Client-count sweep: KIQFL vs FedAvg (fair)."""
    rows = []
    client_counts = [5, 10] if args.quick else [5, 10, 15]
    algos = ['KIQFL', 'FedAvg']
    done = completed_run_keys()
    for n_clients in client_counts:
        cfg_base = build_base_config(args)
        cfg_base['num_clients'] = n_clients
        trainset, testset, ic, isz, nc = load_dataset(args.dataset, args.data_fraction)
        for algo in algos:
            suite = f'scale_{n_clients}'
            if is_run_done(done, suite, algo, 42, args.dataset):
                print(f"[scale] SKIP {algo} clients={n_clients}")
                continue
            cfg = copy.deepcopy(cfg_base)
            cfg.update({'seed': 42, 'input_channels': ic, 'image_size': isz, 'num_classes': nc})
            cfg['checkpoint_suite'] = suite
            cfg['checkpoint_variant'] = ''
            cfg['checkpoint_tag'] = f'c{n_clients}'
            set_seed(42)
            results, wall = run_algorithm_once(algo, trainset, testset, cfg)
            row = flatten_run_result(algo, 42, suite, cfg, results, wall)
            if row:
                rows.append(row)
                append_rows([row])
                done.add((suite, str(algo), '42', str(args.dataset), ''))
            print(f"[scale] {algo} clients={n_clients} acc={row['final_accuracy']:.4f}")
    return rows


def run_fashion_generalization(args):
    if args.quick:
        return []
    rows = []
    for ds in ['fashion_mnist']:
        args_copy = copy.deepcopy(args)
        args_copy.dataset = ds
        rows.extend(run_main_comparison(args_copy))
    return rows


def run_noniid_suite(args):
    """Non-IID Dirichlet stress: KIQFL vs FedAvg over alpha grid."""
    rows = []
    alphas = [0.1, 0.5, 1.0, 5.0]
    algos = ['KIQFL', 'FedAvg']
    trainset, testset, ic, isz, nc = load_dataset(args.dataset, args.data_fraction)
    n_seeds = 1 if args.quick else max(1, min(args.num_seeds, 3))
    seeds = [42 + i * 111 for i in range(n_seeds)]
    done = completed_run_keys()
    for alpha in alphas:
        for algo in algos:
            for seed in seeds:
                suite = f'noniid_{alpha}'
                if is_run_done(done, suite, algo, seed, args.dataset):
                    print(f"[noniid] SKIP {algo} α={alpha} seed={seed}")
                    continue
                cfg = build_base_config(args)
                cfg['non_iid_alpha'] = alpha
                cfg.update({
                    'seed': seed, 'input_channels': ic,
                    'image_size': isz, 'num_classes': nc,
                })
                cfg['checkpoint_suite'] = suite
                cfg['checkpoint_variant'] = ''
                cfg['checkpoint_tag'] = f'a{alpha}'
                set_seed(seed)
                results, wall = run_algorithm_once(algo, trainset, testset, cfg)
                row = flatten_run_result(
                    algo, seed, suite, cfg, results, wall)
                if row:
                    rows.append(row)
                    append_rows([row])
                    done.add((suite, str(algo), str(seed), str(args.dataset), ''))
                print(f"[noniid] {algo} α={alpha} seed={seed} "
                      f"acc={row['final_accuracy']:.4f}")
    return rows


def resolve_epochs(args, suite):
    """Per-suite thesis epochs unless user passed --epochs explicitly."""
    if getattr(args, 'epochs_user', None) is not None:
        return args.epochs_user
    if args.quick:
        return 10
    return SUITE_EPOCH_DEFAULTS.get(suite, 100)


def parse_args():
    p = argparse.ArgumentParser(
        description='KIQFL experiment suites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Defense / ~18–25h experiment pipeline:
  bash kiqfl pipeline.sh

Or stepwise (defaults: data_fraction=0.50, lr=0.01, local_epochs=2,
  batch_size=128, main=60 epochs / 3 seeds, ablation/noise=35):
  kiqfl experiments --quick --suite main
  kiqfl experiments --suite main
  kiqfl experiments --suite ablation
  kiqfl experiments --suite noise
  kiqfl analyze
""")
    p.add_argument('--quick', action='store_true', help='1 seed, fewer epochs/algorithms')
    p.add_argument('--suite', default='all',
                   choices=['all', 'main', 'ablation', 'noise', 'scalability',
                            'fashion', 'noniid'])
    p.add_argument('--dataset', default='mnist')
    p.add_argument('--data-fraction', type=float, default=0.50,
                   help='Fraction of training data (defense default: 0.50)')
    p.add_argument('--num-seeds', type=int, default=3)
    p.add_argument('--ablation-seeds', type=int, default=1,
                   help='Seeds for ablation suite (default 1 for defense speed)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override per-suite defaults (main=60, ablation/noise=35, …)')
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--num-qubits', type=int, default=8)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--nisq-error', type=float, default=0.01)
    p.add_argument('--non-iid-alpha', type=float, default=1.0,
                   help='Dirichlet α (lower = more non-IID)')
    p.add_argument('--batch-size', type=int, default=128,
                   help='Mini-batch size (128 preferred; fall back to 64/32 if OOM)')
    p.add_argument('--fisher-period', default='per_local_epoch',
                   choices=['every_batch', 'per_local_epoch', 'per_round'],
                   help='FQNGD parameter-shift Fisher frequency (method/structure unchanged)')
    p.add_argument('--aggregation-mode', default='quantum_guided',
                   choices=['full', 'quantum_guided', 'quantum_subnet'])
    p.add_argument('--comparison-mode', default='fair', choices=['fair', 'classical'])
    args = p.parse_args()
    args.epochs_user = args.epochs  # None => use SUITE_EPOCH_DEFAULTS
    if args.quick and args.epochs_user is None:
        args.epochs = 10
    elif args.epochs_user is not None:
        args.epochs = args.epochs_user
    else:
        # Placeholder; main() sets per suite via resolve_epochs
        args.epochs = SUITE_EPOCH_DEFAULTS.get(args.suite, 60)
    return args


def main():
    args = parse_args()
    ensure_result_dir()
    rotate_csv_if_schema_mismatch(CSV_PATH)
    print(f"KIQFL experiments | suite={args.suite} quick={args.quick}")
    print(f"Defaults: data_fraction={args.data_fraction}, lr={args.lr}, "
          f"local_epochs={args.local_epochs}, batch_size={args.batch_size}, "
          f"comparison={args.comparison_mode}, fisher_period={args.fisher_period}")
    print("Locked: fisher_method=parameter_shift | fisher_structure=layer_block | "
          f"num_qubits={args.num_qubits}")
    # Defense default suite set: main + ablation + noise (no fashion/scale unless asked)
    suites = [args.suite] if args.suite != 'all' else [
        'main', 'ablation', 'noise']
    for s in suites:
        args.epochs = resolve_epochs(args, s)
        print(f"\n>>> Running suite={s} | global_epochs={args.epochs} | "
              f"local_epochs={args.local_epochs} | num_seeds={args.num_seeds}")
        if s == 'main':
            run_main_comparison(args)
            snapshot_csv('suite_main')
        elif s == 'ablation':
            run_ablation_suite(args)
            snapshot_csv('suite_ablation')
        elif s == 'noise':
            run_noise_sweep(args)
            snapshot_csv('suite_noise')
        elif s == 'scalability':
            run_scalability_suite(args)
            snapshot_csv('suite_scalability')
        elif s == 'fashion':
            run_fashion_generalization(args)
            snapshot_csv('suite_fashion')
        elif s == 'noniid':
            run_noniid_suite(args)
            snapshot_csv('suite_noniid')
    print(f"Results appended to {CSV_PATH}")


if __name__ == '__main__':
    main()
