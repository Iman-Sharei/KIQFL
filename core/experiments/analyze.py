"""Generate an English results summary from experiments_log.csv."""

import argparse
import os
from datetime import datetime

import pandas as pd

RESULT_DIR = 'result'
CSV_PATH = os.path.join(RESULT_DIR, 'experiments_log.csv')
OUTPUT_PATH = os.path.join(RESULT_DIR, 'results_summary.txt')


def load_experiments(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No experiment log at {path}. Run: kiqfl experiments")
    return pd.read_csv(path)


def summarize_main(df):
    main = df[df['suite'] == 'main'].copy()
    if main.empty:
        return "Main algorithm comparison\n  No main comparison runs found.\n"
    lines = [
        "Main algorithm comparison",
        "  algorithm | mean_acc | std | f1_macro | jain | comm_eff | conv",
    ]
    for algo, grp in main.groupby('algorithm'):
        jain = grp['jain_fairness'].dropna().mean() if 'jain_fairness' in grp.columns else 0
        f1 = grp['f1_macro'].mean() if 'f1_macro' in grp.columns else 0
        comm_eff = grp['comm_efficiency'].mean() if 'comm_efficiency' in grp.columns else 0
        conv = grp['convergence_epoch'].dropna().mean() if grp['convergence_epoch'].notna().any() else 'n/a'
        lines.append(
            f"  {algo} | {grp['final_accuracy'].mean():.4f} | {grp['final_accuracy'].std():.4f} | "
            f"{f1:.4f} | {jain:.3f} | {comm_eff:.6f} | {conv}"
        )
    kiqfl = main[main['algorithm'] == 'KIQFL']
    if not kiqfl.empty and 'p_value_vs_fedavg' in kiqfl.columns:
        pvals = kiqfl['p_value_vs_fedavg'].dropna().unique()
        if len(pvals):
            lines.append(f"  KIQFL vs FedAvg paired t-test p = {pvals[0]:.4f}")
    return '\n'.join(lines) + '\n'


def summarize_ablation(df):
    abl = df[df['suite'] == 'ablation'].copy()
    if abl.empty:
        return ""
    lines = ["Ablation study", "  variant | final_accuracy | wall_time_s"]
    for _, row in abl.iterrows():
        lines.append(
            f"  {row.get('variant', 'n/a')} | {row['final_accuracy']:.4f} | {row['wall_time_s']:.1f}"
        )
    return '\n'.join(lines) + '\n'


def summarize_comm(df):
    if 'comm_full_KB' not in df.columns:
        return ""
    lines = ["Communication cost"]
    for algo in df['algorithm'].unique():
        sub = df[df['algorithm'] == algo]
        lines.append(
            f"  {algo}: full={sub['comm_full_KB'].mean():.2f} KB, "
            f"state={sub['comm_state_KB'].mean():.2f} KB, "
            f"effective={sub['comm_effective_KB'].mean():.2f} KB"
        )
    return '\n'.join(lines) + '\n'


def write_results_summary(df, output=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    body = [
        "KIQFL experiment results summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total runs: {len(df)}",
        "",
        summarize_main(df),
        summarize_ablation(df),
        summarize_comm(df),
        "NISQ cost transparency",
    ]
    if 'gate_count' in df.columns:
        body.append(f"  Mean gates per round: {df['gate_count'].mean():.0f}")
    if 'hilbert_dim' in df.columns and df['hilbert_dim'].notna().any():
        body.append(
            f"  Hilbert dimension (2^n): {int(df['hilbert_dim'].dropna().iloc[0])}"
        )
    body.append("")
    body.append(
        "Note: classical state-vector simulation — cost transparency, not hardware speedup."
    )
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(body) + '\n')
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment CSV to a text report.')
    parser.add_argument('--csv', default=CSV_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    args = parser.parse_args()
    df = load_experiments(args.csv)
    write_results_summary(df, args.output)


if __name__ == '__main__':
    main()
