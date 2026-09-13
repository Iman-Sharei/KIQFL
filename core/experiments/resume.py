"""
Resume helper for KIQFL notebook / pipeline after power loss or crash.

Two layers (already in the project):
  1) result/experiments_log.csv  — finished (suite, algo, seed, variant) rows are skipped
  2) result/checkpoints/*.pt     — mid-seed epoch resume inside an unfinished run

This module adds a single entry point that:
  - reports what is already done
  - runs only remaining suites: main → ablation → noise → analyze → figures
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import core.experiments.runner as rex
from core.experiments.ablations import ABLATION_VARIANTS
from core.fl.utils import set_seed
from core.fl.config import default_config

RESULT_DIR = Path("result")
CSV_PATH = Path(rex.CSV_PATH)
CKPT_DIR = RESULT_DIR / "checkpoints"


def _defense_defaults():
    """Match defense / ~18–25h profile."""
    cfg = default_config()
    return {
        "dataset": "mnist",
        "data_fraction": float(cfg.get("data_fraction", 0.50)),
        "local_epochs": int(cfg.get("local_epochs", 2)),
        "batch_size": int(cfg.get("batch_size", 128)),
        "lr": float(cfg.get("learning_rate", 0.01)),
        "num_qubits": 8,
        "num_layers": 2,
        "nisq_error": float(cfg.get("nisq_error_rate", 0.01)),
        "non_iid_alpha": float(cfg.get("non_iid_alpha", 1.0)),
        "comparison_mode": "fair",
        "fisher_period": str(cfg.get("fisher_period", "per_local_epoch")),
        "aggregation_mode": "quantum_guided",
        "num_seeds": 3,
        "ablation_seeds": 1,
        "main_epochs": 60,
        "ablation_epochs": 35,
        "noise_epochs": 35,
        "main_algos": ["KIQFL", "FedAvg", "FQNGD", "Centralized"],
        "noise_rates": [0.0, 0.01, 0.03, 0.05, 0.10],
        "noise_algos": ["KIQFL", "FedAvg"],
    }


def make_args(defense=None, **overrides):
    d = _defense_defaults()
    if defense:
        # Accept notebook DEFENSE_CONFIG shape
        d["dataset"] = defense.get("dataset", d["dataset"])
        d["data_fraction"] = float(defense.get("data_fraction", d["data_fraction"]))
        d["local_epochs"] = int(defense.get("main", {}).get("local_epochs", d["local_epochs"]))
        d["batch_size"] = int(defense.get("batch_size", d["batch_size"]))
        d["lr"] = float(defense.get("learning_rate", d["lr"]))
        d["num_qubits"] = int(defense.get("num_qubits", d["num_qubits"]))
        d["num_layers"] = int(defense.get("num_layers", d["num_layers"]))
        d["nisq_error"] = float(defense.get("nisq_error_rate", d["nisq_error"]))
        d["non_iid_alpha"] = float(defense.get("non_iid_alpha", d["non_iid_alpha"]))
        d["comparison_mode"] = defense.get("comparison_mode", d["comparison_mode"])
        d["fisher_period"] = defense.get("fisher_period", d["fisher_period"])
        d["aggregation_mode"] = defense.get("aggregation_mode", d["aggregation_mode"])
        d["num_seeds"] = int(defense.get("main", {}).get("num_seeds", d["num_seeds"]))
        d["ablation_seeds"] = int(defense.get("ablation", {}).get("ablation_seeds", d["ablation_seeds"]))
        d["main_epochs"] = int(defense.get("main", {}).get("global_epochs", d["main_epochs"]))
        d["ablation_epochs"] = int(defense.get("ablation", {}).get("global_epochs", d["ablation_epochs"]))
        d["noise_epochs"] = int(defense.get("noise", {}).get("global_epochs", d["noise_epochs"]))
        if "main" in defense and "algorithms" in defense["main"]:
            d["main_algos"] = list(defense["main"]["algorithms"])
        if "noise" in defense:
            d["noise_rates"] = list(defense["noise"].get("noise_rates", d["noise_rates"]))
            d["noise_algos"] = list(defense["noise"].get("algorithms", d["noise_algos"]))
    d.update(overrides)
    return SimpleNamespace(
        quick=False,
        suite="main",
        dataset=d["dataset"],
        data_fraction=d["data_fraction"],
        num_seeds=d["num_seeds"],
        ablation_seeds=d["ablation_seeds"],
        epochs=d["main_epochs"],
        epochs_user=d["main_epochs"],
        local_epochs=d["local_epochs"],
        num_clients=10,
        num_qubits=d["num_qubits"],
        num_layers=d["num_layers"],
        lr=d["lr"],
        nisq_error=d["nisq_error"],
        non_iid_alpha=d["non_iid_alpha"],
        batch_size=d["batch_size"],
        fisher_period=d["fisher_period"],
        aggregation_mode=d["aggregation_mode"],
        comparison_mode=d["comparison_mode"],
        _defense=d,
    )


def _load_csv():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[resume] warn: could not read CSV: {e}")
        return pd.DataFrame()


def expected_main_jobs(defense=None):
    d = make_args(defense)._defense
    seeds = [42 + i * 111 for i in range(d["num_seeds"])]
    return [(algo, seed) for algo in d["main_algos"] for seed in seeds]


def expected_ablation_jobs(defense=None):
    d = make_args(defense)._defense
    seeds = [42 + i * 111 for i in range(d["ablation_seeds"])]
    return [(name, seed) for name in ABLATION_VARIANTS for seed in seeds]


def expected_noise_jobs(defense=None):
    d = make_args(defense)._defense
    return [(algo, eps) for eps in d["noise_rates"] for algo in d["noise_algos"]]


def status_report(defense=None, out_dir: Path | None = None):
    """Print / return what is finished vs remaining."""
    df = _load_csv()
    done = rex.completed_run_keys()
    dataset = make_args(defense).dataset

    main_jobs = expected_main_jobs(defense)
    main_done = [
        (a, s) for a, s in main_jobs
        if rex.is_run_done(done, "main", a, s, dataset)
    ]
    main_todo = [j for j in main_jobs if j not in main_done]

    abl_jobs = expected_ablation_jobs(defense)
    abl_done = [
        (v, s) for v, s in abl_jobs
        if rex.is_run_done(done, "ablation", "KIQFL", s, dataset, variant=v)
    ]
    abl_todo = [j for j in abl_jobs if j not in abl_done]

    noise_jobs = expected_noise_jobs(defense)
    noise_done = [
        (a, e) for a, e in noise_jobs
        if rex.is_run_done(done, f"noise_{e}", a, 42, dataset)
    ]
    noise_todo = [j for j in noise_jobs if j not in noise_done]

    ckpts = list(CKPT_DIR.glob("*.pt")) if CKPT_DIR.exists() else []

    report = {
        "csv": str(CSV_PATH),
        "csv_rows": int(len(df)),
        "main": {"done": len(main_done), "total": len(main_jobs), "todo": main_todo},
        "ablation": {"done": len(abl_done), "total": len(abl_jobs), "todo": abl_todo},
        "noise": {"done": len(noise_done), "total": len(noise_jobs), "todo": noise_todo},
        "open_checkpoints": [c.name for c in ckpts],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print("=" * 64)
    print("  KIQFL RESUME STATUS")
    print("=" * 64)
    print(f"CSV: {CSV_PATH}  ({report['csv_rows']} rows)")
    print(f"Main:     {report['main']['done']}/{report['main']['total']} done"
          f"  | remaining: {len(main_todo)}")
    if main_todo[:5]:
        print("  next:", main_todo[:5], ("..." if len(main_todo) > 5 else ""))
    print(f"Ablation: {report['ablation']['done']}/{report['ablation']['total']} done"
          f"  | remaining: {len(abl_todo)}")
    print(f"Noise:    {report['noise']['done']}/{report['noise']['total']} done"
          f"  | remaining: {len(noise_todo)}")
    print(f"Open mid-run checkpoints: {len(ckpts)}")
    for name in report["open_checkpoints"][:8]:
        print(f"  - {name}")
    if not main_todo and not abl_todo and not noise_todo:
        print("\nAll training suites complete. Safe to run analyze + figures.")
    else:
        print("\nRe-run this resume cell (or suite cells). Finished jobs are SKIPPED.")
        print("Mid-seed interruptions resume from result/checkpoints/*.pt")
    print("=" * 64)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resume_status.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    return report


def run_remaining(
    defense=None,
    out_dir: Path | None = None,
    run_main=True,
    run_ablation=True,
    run_noise=True,
    run_analyze=True,
    run_figures=True,
):
    """
    Continue unfinished experiment suites safely after reboot / power loss.

    Safe to call repeatedly: completed CSV rows are skipped; unfinished seeds
    resume from mid-run checkpoints.
    """
    rex.ensure_result_dir()
    report0 = status_report(defense=defense, out_dir=out_dir)
    base = make_args(defense)
    t0 = time.time()
    summary = {"started": datetime.now().isoformat(timespec="seconds"), "stages": {}}

    if run_main and report0["main"]["todo"]:
        print("\n>>> RESUME MAIN")
        args = make_args(defense)
        args.suite = "main"
        args.epochs = args._defense["main_epochs"]
        args.epochs_user = args.epochs
        rows = rex.run_main_comparison(args)
        rex.snapshot_csv("suite_main")
        summary["stages"]["main"] = {"rows": len(rows), "ok": True}
    elif run_main:
        print("\n>>> MAIN already complete — skip")
        summary["stages"]["main"] = {"skipped": True}

    if run_ablation and report0["ablation"]["todo"]:
        # refresh status (main may have finished more)
        report0 = status_report(defense=defense, out_dir=out_dir)
        print("\n>>> RESUME ABLATION")
        args = make_args(defense)
        args.suite = "ablation"
        args.epochs = args._defense["ablation_epochs"]
        args.epochs_user = args.epochs
        args.local_epochs = args._defense["local_epochs"]
        rows = rex.run_ablation_suite(args)
        rex.snapshot_csv("suite_ablation")
        summary["stages"]["ablation"] = {"rows": len(rows), "ok": True}
    elif run_ablation:
        print("\n>>> ABLATION already complete — skip")
        summary["stages"]["ablation"] = {"skipped": True}

    if run_noise and status_report(defense=defense, out_dir=out_dir)["noise"]["todo"]:
        print("\n>>> RESUME NOISE")
        args = make_args(defense)
        args.suite = "noise"
        args.epochs = args._defense["noise_epochs"]
        args.epochs_user = args.epochs
        args.num_seeds = 1
        rows = rex.run_noise_sweep(args)
        rex.snapshot_csv("suite_noise")
        summary["stages"]["noise"] = {"rows": len(rows), "ok": True}
    elif run_noise:
        print("\n>>> NOISE already complete — skip")
        summary["stages"]["noise"] = {"skipped": True}

    if run_analyze:
        print("\n>>> ANALYZE")
        try:
            import core.experiments.analyze as ar
            import sys
            old = sys.argv
            sys.argv = ["kiqfl-analyze"]
            try:
                ar.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old
            summary["stages"]["analyze"] = {"ok": True}
            if out_dir is not None and Path("result/results_summary.txt").exists():
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy2("result/results_summary.txt", Path(out_dir) / "results_summary.txt")
            if out_dir is not None and CSV_PATH.exists():
                shutil.copy2(CSV_PATH, Path(out_dir) / "experiments_log.csv")
        except Exception as e:
            print(f"[resume] analyze warning: {e}")
            summary["stages"]["analyze"] = {"ok": False, "error": str(e)}

    if run_figures and CSV_PATH.exists():
        print("\n>>> FIGURES")
        try:
            _write_basic_figures(out_dir)
            summary["stages"]["figures"] = {"ok": True}
        except Exception as e:
            print(f"[resume] figures warning: {e}")
            summary["stages"]["figures"] = {"ok": False, "error": str(e)}

    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    final = status_report(defense=defense, out_dir=out_dir)
    summary["final_status"] = {
        "main_done": f"{final['main']['done']}/{final['main']['total']}",
        "ablation_done": f"{final['ablation']['done']}/{final['ablation']['total']}",
        "noise_done": f"{final['noise']['done']}/{final['noise']['total']}",
    }
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "resume_run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    print(f"\n[resume] finished in {summary['elapsed_s']}s")
    return summary


def _write_basic_figures(out_dir: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _load_csv()
    if df.empty:
        print("[resume] no CSV rows for figures")
        return
    fig_dir = Path(out_dir) / "figures" if out_dir else RESULT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    main = df[df["suite"] == "main"] if "suite" in df.columns else df.iloc[0:0]
    if not main.empty and "final_accuracy" in main.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        stats = main.groupby("algorithm")["final_accuracy"].agg(["mean", "std"])
        order = [a for a in ["KIQFL", "FedAvg", "FQNGD", "Centralized"] if a in stats.index]
        stats = stats.reindex(order)
        ax.bar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), capsize=4)
        ax.set_ylim(0, 1.05)
        ax.set_title("Main comparison (resume-generated)")
        fig.tight_layout()
        fig.savefig(fig_dir / "main_accuracy.png", dpi=160)
        plt.close(fig)
        print("Saved", fig_dir / "main_accuracy.png")

    abl = df[df["suite"] == "ablation"] if "suite" in df.columns else df.iloc[0:0]
    if not abl.empty and "variant" in abl.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        abl_s = abl.sort_values("final_accuracy", ascending=False)
        ax.barh(abl_s["variant"].astype(str), abl_s["final_accuracy"])
        ax.set_title("Ablation (resume-generated)")
        fig.tight_layout()
        fig.savefig(fig_dir / "ablation_study.png", dpi=160)
        plt.close(fig)

    noise = df[df["suite"].astype(str).str.startswith("noise_")] if "suite" in df.columns else df.iloc[0:0]
    if not noise.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for algo, g in noise.groupby("algorithm"):
            g = g.sort_values("nisq_error_rate")
            ax.plot(g["nisq_error_rate"], g["final_accuracy"], marker="o", label=algo)
        ax.legend()
        ax.set_title("Noise sweep (resume-generated)")
        fig.tight_layout()
        fig.savefig(fig_dir / "noise_sweep.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    status_report()
    run_remaining()
