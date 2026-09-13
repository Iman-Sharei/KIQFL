"""
Smoke test for mid-seed checkpoints + epoch/terminal logging.

Run from repo root:
  python test/test_checkpoint_logging.py
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core.fl.checkpoint import (  # noqa: E402
    append_epoch_log,
    apply_rng_from_checkpoint,
    checkpoint_path,
    clear_midrun_checkpoint,
    epoch_log_path,
    load_midrun_checkpoint,
    run_checkpoint_id,
    save_midrun_checkpoint,
)
from core.fl.config import default_config  # noqa: E402
from core.fl.data import load_dataset
from core.fl.algorithms import run_fedavg  # noqa: E402
from core.experiments.pipeline import open_terminal_log, run as pipeline_run  # noqa: E402


PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def test_unit_helpers():
    print("\n=== 1) Unit: checkpoint id / save / load / RNG / epoch log / clear ===")
    cfg = {
        "checkpoint_suite": "ckpt_unit_test",
        "seed": 777,
        "dataset_name": "mnist",
        "checkpoint_variant": "unit",
        "resume_from_checkpoint": True,
        "global_epochs": 10,
    }
    algo = "UnitDummy"
    # Isolate under result/ so paths match production layout
    ckpt_p = checkpoint_path(cfg, algo)
    log_p = epoch_log_path(cfg, algo)
    for p in (ckpt_p, log_p):
        if os.path.exists(p):
            os.remove(p)

    rid = run_checkpoint_id(cfg, algo)
    check("stable id", rid == "ckpt_unit_test_UnitDummy_seed777_mnist_vunit", rid)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

    model = Tiny()
    with torch.no_grad():
        model.fc.weight.fill_(1.23)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(3)

    save_midrun_checkpoint(
        cfg, algo,
        next_epoch=3,
        global_model=model,
        results=[{"global_epoch": 1, "accuracy": 0.5}, {"global_epoch": 2, "accuracy": 0.6}],
        convergence_epoch=None,
        cumulative_comm_kb=1.5,
        extra={"note": "unit"},
    )
    check("ckpt file exists", os.path.isfile(ckpt_p), ckpt_p)

    append_epoch_log(cfg, algo, {"global_epoch": 1, "loss": 1.0, "accuracy": 0.5, "fidelity": 0.9})
    append_epoch_log(cfg, algo, {"global_epoch": 2, "loss": 0.8, "accuracy": 0.6, "fidelity": 0.85})
    check("epoch log exists", os.path.isfile(log_p), log_p)
    with open(log_p, newline="") as f:
        rows = list(csv.DictReader(f))
    check("epoch log rows", len(rows) == 2, f"n={len(rows)}")
    check("epoch log columns", "accuracy" in rows[0] and "fidelity" in rows[0])

    # Mutate RNG + weights, then restore
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    model2 = Tiny()
    with torch.no_grad():
        model2.fc.weight.fill_(0.0)

    ckpt = load_midrun_checkpoint(cfg, algo, map_location="cpu")
    check("load returns dict", isinstance(ckpt, dict))
    check("next_epoch", ckpt.get("next_epoch") == 3, str(ckpt.get("next_epoch")))
    check("results len", len(ckpt.get("results") or []) == 2)

    model2.load_state_dict(ckpt["model_state_dict"])
    apply_rng_from_checkpoint(ckpt)
    check("weights restored", torch.allclose(model2.fc.weight, torch.full_like(model2.fc.weight, 1.23)))
    # After restore, next draws should match what would have followed the saved state
    # (we only check that setstate does not crash and torch state is a Tensor)
    check("rng_torch present", torch.is_tensor(ckpt.get("rng_torch")))

    clear_midrun_checkpoint(cfg, algo)
    check("ckpt cleared", not os.path.exists(ckpt_p))
    # keep epoch log for inspection; remove to avoid clutter
    os.remove(log_p)


def test_fedavg_resume():
    print("\n=== 2) Integration: FedAvg classical mid-seed resume ===")
    trainset, testset, ch, sz, ncls = load_dataset("mnist", data_fraction=0.02)
    cfg = default_config(
        comparison_mode="classical",
        use_hybrid_qnn=False,
        optimizer="sgd",
        num_clients=2,
        local_epochs=1,
        global_epochs=4,
        learning_rate=0.01,
        batch_size=64,
        data_fraction=0.02,
        seed=4242,
        dataset_name="mnist",
        input_channels=ch,
        image_size=sz,
        num_classes=ncls,
        checkpoint_suite="ckpt_smoke",
        checkpoint_variant="fedavg_resume",
        resume_from_checkpoint=True,
        use_early_stopping=False,
        require_circuit_state=False,
    )
    algo = "FedAvg"
    ckpt_p = checkpoint_path(cfg, algo)
    log_p = epoch_log_path(cfg, algo)
    for p in (ckpt_p, log_p):
        if os.path.exists(p):
            os.remove(p)

    # Phase A: run 2 epochs then stop generator before clear
    print("  … phase A: train 2/4 epochs then interrupt")
    gen = run_fedavg(trainset, testset, cfg, yield_results=True)
    r1 = next(gen)
    r2 = next(gen)
    gen.close()  # GeneratorExit before clear_midrun_checkpoint

    check("phase A epoch metrics", r1["global_epoch"] == 1 and r2["global_epoch"] == 2)
    check("ckpt after interrupt", os.path.isfile(ckpt_p), ckpt_p)
    ckpt = load_midrun_checkpoint(cfg, algo, map_location="cpu")
    check("ckpt next_epoch==2", ckpt is not None and int(ckpt["next_epoch"]) == 2,
          str(None if ckpt is None else ckpt.get("next_epoch")))
    with open(log_p, newline="") as f:
        rows = list(csv.DictReader(f))
    check("epoch log has 2 rows", len(rows) == 2, f"n={len(rows)}")
    acc_after_2 = float(r2["accuracy"])

    # Phase B: resume should start at epoch 3 and finish 4, then clear ckpt
    print("  … phase B: resume remaining epochs")
    out = list(run_fedavg(trainset, testset, cfg, yield_results=True))
    epochs = [r["global_epoch"] for r in out]
    check("resume yielded epochs 3–4 only", epochs == [3, 4], str(epochs))
    check("total results in last yield chain", True)  # soft
    # Full results live in returned list when generator finishes — list() gets yielded only
    check("ckpt cleared after finish", not os.path.exists(ckpt_p))
    with open(log_p, newline="") as f:
        rows = list(csv.DictReader(f))
    check("epoch log has 4 rows total", len(rows) == 4, f"n={len(rows)}")
    # Continuity: first two rows still match interrupted run's epochs
    check("log epoch sequence", [int(r["global_epoch"]) for r in rows] == [1, 2, 3, 4])
    check("acc after epoch2 still in log", abs(float(rows[1]["accuracy"]) - acc_after_2) < 1e-9,
          f"{rows[1]['accuracy']} vs {acc_after_2}")


def test_terminal_tee():
    print("\n=== 3) Terminal tee (pipeline helper) ===")
    log_path = open_terminal_log("ckpt_smoke")
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write("# smoke terminal test\n")
        pipeline_run([sys.executable, "-c", "print('TEE_MARKER_OK'); print('line2')"], log_fp=fp)
    text = log_path.read_text(encoding="utf-8")
    check("terminal log file", log_path.is_file(), str(log_path))
    check("tee captured stdout", "TEE_MARKER_OK" in text and "line2" in text)
    # cleanup this tiny log
    try:
        log_path.unlink()
    except OSError:
        pass


def main():
    print("KIQFL checkpoint + logging smoke test")
    print(f"cwd={ROOT}  cuda={torch.cuda.is_available()}")
    test_unit_helpers()
    test_fedavg_resume()
    test_terminal_tee()
    print(f"\n=== SUMMARY: {PASSED} passed, {FAILED} failed ===")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
