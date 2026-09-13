"""
KIQFL full experiment pipeline (cross-platform).

Usage (from project root, with venv active):
  python kiqfl pipeline.py
  python kiqfl pipeline.py --skip-smoke
  python kiqfl pipeline.py --only main

Logging layers (do not confuse them):
  1) result/experiments_log.csv     — final seed summaries (result tables)
  2) result/epoch_logs/*.csv        — per-epoch Acc/Loss/F (scientific curves)
  3) result/checkpoints/*.pt        — resume state only (not a result table)
  4) result/terminal_logs/*.log     — full stdout/stderr transcript (audit trail)

Locked: fisher_method=parameter_shift, fisher_structure=layer_block, num_qubits=8
FQNGD: fisher_period=per_local_epoch (sampled parameter-shift Fisher)

Quality profile (est. ~21h GPU):
  data_fraction=0.50, local_epochs=2, main=60 ep / 3 seeds, ablation/noise=35
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
CSV = RESULT / "experiments_log.csv"
TERM_LOG_DIR = RESULT / "terminal_logs"

COMMON = [
    "--dataset", "mnist",
    "--data-fraction", "0.50",
    "--local-epochs", "2",
    "--lr", "0.01",
    "--batch-size", "128",
    "--num-qubits", "8",
    "--num-layers", "2",
    "--nisq-error", "0.01",
    "--non-iid-alpha", "1.0",
    "--comparison-mode", "fair",
    "--fisher-period", "per_local_epoch",
]


def open_terminal_log(stage: str | None = None) -> Path:
    TERM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"pipeline_{stamp}.log" if not stage else f"{stage}_{stamp}.log"
    return TERM_LOG_DIR / name


def run(cmd: list[str], log_fp=None) -> None:
    line = "\n>>> " + " ".join(cmd)
    print(line, flush=True)
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()
    with subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for out_line in proc.stdout:
            sys.stdout.write(out_line)
            sys.stdout.flush()
            if log_fp is not None:
                log_fp.write(out_line)
                log_fp.flush()
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def check_cuda(log_fp=None) -> None:
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA required for full GPU runs'; "
        "print(torch.cuda.get_device_name(0))"
    )
    run([sys.executable, "-c", code], log_fp=log_fp)


def archive_csv() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = RESULT / f"experiments_log_pre_pipeline_{stamp}.csv"
        shutil.move(str(CSV), str(dest))
        print(f"Archived previous CSV -> {dest.name}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="KIQFL defense (~18–25h) experiment pipeline")
    p.add_argument("--skip-smoke", action="store_true", help="Skip 5-epoch smoke test")
    p.add_argument(
        "--only",
        choices=["smoke", "main", "ablation", "noise", "analyze"],
        default=None,
        help="Run a single stage only",
    )
    p.add_argument("--skip-cuda-check", action="store_true")
    p.add_argument(
        "--archive-csv",
        action="store_true",
        help="Move existing experiments_log.csv aside before starting (fresh run)",
    )
    p.add_argument(
        "--no-terminal-log",
        action="store_true",
        help="Do not tee stdout into result/terminal_logs/",
    )
    args = p.parse_args()

    py = sys.executable
    stages = {
        "smoke": [
            py, "-m", "core", "experiments", "--quick", "--suite", "main",
            *COMMON, "--epochs", "5",
        ],
        "main": [
            py, "-m", "core", "experiments", "--suite", "main",
            *COMMON, "--num-seeds", "3", "--epochs", "60",
        ],
        "ablation": [
            py, "-m", "core", "experiments", "--suite", "ablation",
            *COMMON, "--epochs", "35", "--ablation-seeds", "1",
        ],
        "noise": [
            py, "-m", "core", "experiments", "--suite", "noise",
            *COMMON, "--epochs", "35",
        ],
        "analyze": [py, "-m", "core", "analyze"],
    }

    log_path = None if args.no_terminal_log else open_terminal_log(args.only or "full")
    log_fp = None
    if log_path is not None:
        log_fp = open(log_path, "a", encoding="utf-8")
        print(f"Terminal transcript -> {log_path}")
        log_fp.write(f"# KIQFL defense pipeline log started {datetime.now().isoformat()}\n")
        log_fp.write(
            "# Locked: fisher_method=parameter_shift, fisher_structure=layer_block, "
            "num_qubits=8, fisher_period=per_local_epoch\n"
            "# Quality: data_fraction=0.50, local_epochs=2, main=60, ablation/noise=35\n"
        )
        log_fp.flush()

    try:
        if args.only:
            if not args.skip_cuda_check and args.only != "analyze":
                check_cuda(log_fp=log_fp)
            if args.only != "analyze":
                if args.archive_csv:
                    archive_csv()
            run(stages[args.only], log_fp=log_fp)
        else:
            if not args.skip_cuda_check:
                check_cuda(log_fp=log_fp)
            if args.archive_csv:
                archive_csv()
            order = []
            if not args.skip_smoke:
                order.append("smoke")
            order.extend(["main", "ablation", "noise", "analyze"])
            for name in order:
                print(f"\n========== STAGE: {name} ==========", flush=True)
                if log_fp is not None:
                    log_fp.write(f"\n# ===== STAGE {name} =====\n")
                    log_fp.flush()
                run(stages[name], log_fp=log_fp)
        print("\nDONE. See result/results_summary.txt and result/")
        print(
            "Thesis note: FQNGD parameter-shift Fisher computed periodically "
            "(per_local_epoch / representative batch), not for every mini-batch."
        )
    finally:
        if log_fp is not None:
            log_fp.close()


if __name__ == "__main__":
    main()
