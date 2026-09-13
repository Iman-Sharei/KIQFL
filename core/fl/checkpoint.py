"""Mid-seed epoch checkpoints so Ctrl+C / crash can resume inside a seed."""
from __future__ import annotations

# moved from training_checkpoint.py
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch

CKPT_DIR = os.path.join("result", "checkpoints")
EPOCH_LOG_DIR = os.path.join("result", "epoch_logs")


def ensure_dirs():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(EPOCH_LOG_DIR, exist_ok=True)


def run_checkpoint_id(config, algorithm):
    """Stable id for one (suite, algo, seed, dataset, variant) training job."""
    suite = str(config.get("checkpoint_suite", config.get("suite", "main")))
    seed = config.get("seed", 42)
    dataset = str(config.get("dataset_name", "mnist"))
    variant = str(config.get("checkpoint_variant", config.get("variant", "")) or "none")
    noise = config.get("nisq_error_rate", "")
    clients = config.get("num_clients", "")
    # Include noise/clients so noise/scale sweeps don't collide
    extra = []
    if suite.startswith("noise") or noise != "":
        extra.append(f"eps{noise}")
    if str(suite).startswith("scale") or config.get("checkpoint_tag"):
        extra.append(f"c{clients}")
    tag = config.get("checkpoint_tag", "")
    if tag:
        extra.append(str(tag))
    extras = ("_" + "_".join(extra)) if extra else ""
    safe_algo = str(algorithm).replace("+", "_").replace(" ", "_")
    return f"{suite}_{safe_algo}_seed{seed}_{dataset}_v{variant}{extras}"


def checkpoint_path(config, algorithm):
    ensure_dirs()
    return os.path.join(CKPT_DIR, run_checkpoint_id(config, algorithm) + ".pt")


def epoch_log_path(config, algorithm):
    ensure_dirs()
    return os.path.join(EPOCH_LOG_DIR, run_checkpoint_id(config, algorithm) + ".csv")


_EPOCH_CSV_COLUMNS = [
    "timestamp", "algorithm", "seed", "global_epoch", "loss", "accuracy", "fidelity",
    "convergence_epoch", "wall_note",
]


def append_epoch_log(config, algorithm, result_row):
    """Append one finished epoch to a human-readable CSV (survives Ctrl+C)."""
    path = epoch_log_path(config, algorithm)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm,
        "seed": config.get("seed", ""),
        "global_epoch": result_row.get("global_epoch", ""),
        "loss": result_row.get("loss", ""),
        "accuracy": result_row.get("accuracy", ""),
        "fidelity": result_row.get("fidelity", ""),
        "convergence_epoch": result_row.get("convergence_epoch", ""),
        "wall_note": "epoch_checkpoint",
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_EPOCH_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _cpu_tensor(t):
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach().cpu()
    return t


def _list_cpu(states):
    if states is None:
        return None
    out = []
    for s in states:
        out.append(_cpu_tensor(s) if s is not None else None)
    return out


def save_midrun_checkpoint(
    config,
    algorithm,
    *,
    next_epoch,
    global_model,
    results,
    convergence_epoch=None,
    cumulative_comm_kb=0.0,
    extra=None,
):
    """
    Save state after finishing `next_epoch - 1` (0-based next index to run).
    """
    if not config.get("resume_from_checkpoint", True):
        return None
    path = checkpoint_path(config, algorithm)
    ensure_dirs()
    payload = {
        "algorithm": algorithm,
        "next_epoch": int(next_epoch),
        "global_epochs": int(config.get("global_epochs", 100)),
        "seed": config.get("seed", 42),
        "model_state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
        "results": results,
        "convergence_epoch": convergence_epoch,
        "cumulative_comm_kb": float(cumulative_comm_kb),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        try:
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    # atomic-ish write
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_midrun_checkpoint(config, algorithm, map_location=None):
    """Return dict or None. Caller applies model/kan/etc."""
    if not config.get("resume_from_checkpoint", True):
        return None
    path = checkpoint_path(config, algorithm)
    if not os.path.exists(path):
        return None
    if map_location is None:
        map_location = "cpu"
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        print(f"[checkpoint] failed to load {path}: {exc}")
        return None
    # Sanity: seed / algo
    if str(ckpt.get("algorithm", "")) != str(algorithm):
        print(f"[checkpoint] algo mismatch, ignoring {path}")
        return None
    if int(ckpt.get("seed", -1)) != int(config.get("seed", 42)):
        print(f"[checkpoint] seed mismatch, ignoring {path}")
        return None
    return ckpt


def _as_byte_tensor(state):
    """torch.set_rng_state requires a CPU ByteTensor (load may alter dtype/device)."""
    if state is None:
        return None
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.uint8)
    else:
        state = state.detach().cpu()
        if state.dtype != torch.uint8:
            state = state.to(dtype=torch.uint8)
    return state.contiguous()


def apply_rng_from_checkpoint(ckpt):
    if not ckpt:
        return
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    if "rng_numpy" in ckpt:
        np.random.set_state(ckpt["rng_numpy"])
    if "rng_torch" in ckpt:
        torch.random.set_rng_state(_as_byte_tensor(ckpt["rng_torch"]))
    if "rng_cuda" in ckpt and torch.cuda.is_available():
        try:
            states = ckpt["rng_cuda"]
            if isinstance(states, (list, tuple)):
                states = [_as_byte_tensor(s) for s in states]
            torch.cuda.set_rng_state_all(states)
        except Exception:
            pass


def clear_midrun_checkpoint(config, algorithm):
    path = checkpoint_path(config, algorithm)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[checkpoint] cleared {path}")
        except OSError as exc:
            print(f"[checkpoint] could not remove {path}: {exc}")


def pack_kiqfl_extra(
    kan_model,
    kan_trainer,
    prev_circuit_states,
    prev_global_state,
    fidelity_history,
):
    return {
        "kan_state_dict": {k: v.detach().cpu() for k, v in kan_model.state_dict().items()},
        "kan_optimizer_state_dict": kan_trainer.optimizer.state_dict(),
        "kan_training_data": kan_trainer.training_data,
        "kan_train_step_count": getattr(kan_trainer, "train_step_count", 0),
        "prev_circuit_states": _list_cpu(prev_circuit_states),
        "prev_global_state": _cpu_tensor(prev_global_state),
        "fidelity_history": list(fidelity_history) if fidelity_history else [],
    }


def restore_kiqfl_extra(extra, kan_model, kan_trainer, device):
    if not extra:
        return None, None, []
    if "kan_state_dict" in extra:
        kan_model.load_state_dict(extra["kan_state_dict"])
    if "kan_optimizer_state_dict" in extra:
        try:
            kan_trainer.optimizer.load_state_dict(extra["kan_optimizer_state_dict"])
        except Exception:
            pass
    if "kan_training_data" in extra:
        kan_trainer.training_data = extra["kan_training_data"]
    if "kan_train_step_count" in extra:
        kan_trainer.train_step_count = extra["kan_train_step_count"]
    prev_states = extra.get("prev_circuit_states")
    if prev_states is not None:
        prev_states = [
            (s.to(device) if torch.is_tensor(s) else s) for s in prev_states
        ]
    prev_global = extra.get("prev_global_state")
    if torch.is_tensor(prev_global):
        prev_global = prev_global.to(device)
    fidelity_history = list(extra.get("fidelity_history") or [])
    return prev_states, prev_global, fidelity_history


def pack_simple_extra(prev_circuit_states=None, prev_state=None, optimizer=None, scheduler=None):
    extra = {}
    if prev_circuit_states is not None:
        extra["prev_circuit_states"] = _list_cpu(prev_circuit_states)
    if prev_state is not None:
        extra["prev_state"] = _cpu_tensor(prev_state)
    if optimizer is not None:
        extra["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        extra["scheduler_state_dict"] = scheduler.state_dict()
    return extra
