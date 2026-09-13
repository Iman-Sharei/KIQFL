import copy
import random

import numpy as np
import torch

from core.quantum.fidelity import compute_state_fidelity
from core.quantum.fidelity import apply_adaptive_weight_to_state  # noqa: F401 used by callers via grid

QUANTUM_PARAM_KEYS = ('angles', 'gammas', 'betas', 'output_layer')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indirect_quantum_aggregation(local_state_dicts, circuit_states, prev_global_state,
                                 num_qubits, device_arg=None):
    """
    Central-server aggregation with fidelity-guided weights (KIQFL).

    Architecture: clients upload parameters (+ circuit states) to a **central
    aggregator**; this function builds the next global model. DTQW/KAN only
    reshape the guidance states used for weights — they do not replace the
    server.

    Weight rule (vs previous global circuit state |ψ_g⟩):

        w_i = F(|ψ_i⟩, |ψ_g⟩) / Σ_j F(|ψ_j⟩, |ψ_g⟩)
        F(|a⟩,|b⟩) = |⟨a|b⟩|²

    Then classical parameters are averaged with those weights (not sample-size
    FedAvg). "Indirect" means weights come from quantum fidelity, not that
    parameters skip the server or avoid averaging.

    Falls back to uniform 1/K weights if guidance states are unavailable.
    """
    if device_arg is None:
        device_arg = device
    num_clients = len(local_state_dicts)

    # Fidelity to previous *global* circuit state (not coherent sum of clients)
    fidelity_weights = []
    for i in range(num_clients):
        if (i < len(circuit_states) and circuit_states[i] is not None
                and prev_global_state is not None):
            fid = compute_state_fidelity(circuit_states[i], prev_global_state)
            fid_val = fid.item() if torch.is_tensor(fid) else float(fid)
            fidelity_weights.append(max(fid_val, 1e-6))
        else:
            fidelity_weights.append(1.0 / num_clients)

    total_w = sum(fidelity_weights)
    if total_w < 1e-12:
        weights = [1.0 / num_clients] * num_clients
    else:
        weights = [w / total_w for w in fidelity_weights]

    aggregated = copy.deepcopy(local_state_dicts[0])
    for key in aggregated.keys():
        aggregated[key] = weights[0] * local_state_dicts[0][key].float()
        for i in range(1, num_clients):
            aggregated[key] = aggregated[key] + weights[i] * local_state_dicts[i][key].float()
        aggregated[key] = aggregated[key].to(local_state_dicts[0][key].dtype)

    return aggregated, weights


# ========================================================================================
# 2c. DIFFERENTIAL PRIVACY MECHANISM
# ========================================================================================


def _average_weights(state_dicts):
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict

def is_quantum_param_key(key):
    return any(q in key for q in QUANTUM_PARAM_KEYS)

def compute_coherent_global_state(states):
    """Normalized coherent superposition of client circuit states."""
    valid = [s for s in states if s is not None]
    if not valid:
        return None
    acc = torch.zeros_like(valid[0].to(dtype=torch.cfloat))
    for s in valid:
        acc = acc + s.to(dtype=torch.cfloat, device=acc.device)
    norm = torch.norm(acc)
    if norm < 1e-12:
        return valid[0].clone()
    return (acc / norm).reshape(-1, 1)

def build_fidelity_reference_inputs(input_channels, image_size, num_refs, seed=42):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    refs = []
    for _ in range(num_refs):
        x = torch.randn(1, input_channels, image_size, image_size, generator=g, device=device)
        refs.append(x / torch.norm(x))
    return refs

def extract_circuit_state(model, reference_inputs, num_qubits, prev_state=None,
                          require_circuit_state=True, debug_mode=False):
    """Run model on reference inputs and return mean circuit state."""
    if not hasattr(model, 'get_circuit_state'):
        if require_circuit_state and not debug_mode:
            raise RuntimeError("Model lacks get_circuit_state()")
        return prev_state
    states = []
    with torch.no_grad():
        model.eval()
        for ref in reference_inputs:
            _ = model(ref)
            st = model.get_circuit_state()
            if st is not None:
                states.append(st)
    if states:
        return compute_coherent_global_state(states)
    if prev_state is not None:
        return prev_state.clone()
    if require_circuit_state and not debug_mode:
        raise RuntimeError("Failed to extract circuit state")
    dim = 2 ** num_qubits
    st = torch.randn(dim, dtype=torch.cfloat, device=device)
    return (st / torch.norm(st)).reshape(-1, 1)

def compute_multi_fidelity(prev_states, new_states, reference_states_fn=None):
    """Pairwise fidelity averaged over clients."""
    if not prev_states or not new_states:
        return 0.0
    total, count = 0.0, 0
    for old, new in zip(prev_states, new_states):
        if old is None or new is None:
            continue
        fid = compute_state_fidelity(old, new)
        fv = fid.item() if torch.is_tensor(fid) else float(fid)
        total += max(0.0, min(1.0, fv))
        count += 1
    return total / max(1, count)

def _full_local_state_dicts(local_state_dicts, global_state_dict):
    """Merge partial uploads with global weights."""
    return [{**global_state_dict, **ld} for ld in local_state_dicts]

def freeze_cnn_for_subnet(model, freeze=True):
    """Freeze CNN backbone params for quantum_subnet local training."""
    for name, param in model.named_parameters():
        if 'output_layer' in name or 'angles' in name or 'gammas' in name or 'betas' in name:
            continue
        if 'cnn' in name or 'conv' in name or '.fc' in name:
            param.requires_grad = not freeze

def aggregate_with_mode(local_state_dicts, circuit_states, prev_global_state,
                        global_state_dict, num_qubits, aggregation_mode='quantum_guided',
                        epoch=0, cnn_warmup_epochs=3, num_clients=None):
    """
    Central-server federated aggregation with optional quantum-subnet mode.

    Returns (aggregated_state_dict, weights, global_qstate) where global_qstate
    is a coherent summary of client circuit states for the *next* round's
    previous-global reference (logging / continuity), while fidelity weights
    use ``prev_global_state`` from the prior round when available.
    """
    # Snapshot for next round; weighting uses previous global when present
    client_summary = compute_coherent_global_state(circuit_states)
    fidelity_ref = prev_global_state if prev_global_state is not None else client_summary
    global_qstate = client_summary if client_summary is not None else prev_global_state

    full_locals = _full_local_state_dicts(local_state_dicts, global_state_dict)

    if aggregation_mode == 'full':
        return _average_weights(full_locals), [1.0 / len(full_locals)] * len(full_locals), global_qstate

    agg_sd, weights = indirect_quantum_aggregation(
        full_locals, circuit_states, fidelity_ref, num_qubits, device)

    if aggregation_mode != 'quantum_subnet':
        return agg_sd, weights, global_qstate

    # quantum_subnet: fidelity-weighted quantum params; CNN frozen until warmup then FedAvg
    merged = copy.deepcopy(global_state_dict)
    n = len(local_state_dicts)
    if weights is None or len(weights) != n:
        weights = [1.0 / n] * n

    for key in merged.keys():
        if is_quantum_param_key(key):
            merged[key] = agg_sd[key]
        elif epoch >= cnn_warmup_epochs and ('cnn' in key or 'conv' in key or 'fc' in key):
            merged[key] = agg_sd[key]

    return merged, weights, global_qstate

