import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.models import Hybrid_QNN, SimpleCNN
from core.quantum import (
    compute_state_fidelity, apply_adaptive_weight_to_state,
    quantum_walk_transfer_state_list, compute_adaptive_alpha, KANEdge,
)
from core.fl.data import (
    partition_dataset, weighted_average_weights, _client_data_sizes,
    build_epoch_evaluation, get_client_state_vectors, compute_real_fidelity,
    compute_noise_levels, subsample_clients,
)
from core.fl.aggregation import (
    build_fidelity_reference_inputs, extract_circuit_state, compute_multi_fidelity,
    aggregate_with_mode, freeze_cnn_for_subnet,
)
from core.fl.fisher import (
    apply_quantum_natural_gradient, extract_grad_state_dict, zeros_like_grad_dict,
    accumulate_grad_dict, average_grad_dict, weighted_average_grad_dicts,
    apply_grad_dict_to_model,
)
from core.fl.config import apply_comparison_defaults, create_fl_optimizer
from core.fl.utils import (
    EarlyStopping, KANTrainer, set_seed, create_scheduler,
    compute_comm_breakdown, build_nisq_report, grid_search_alpha_target,
    make_client_val_loader, add_state_dp_noise, enrich_epoch_result,
    track_convergence, sporadic_skip_probability, estimate_grad_noise,
)
from core.fl.checkpoint import (
    save_midrun_checkpoint, load_midrun_checkpoint, clear_midrun_checkpoint,
    append_epoch_log, apply_rng_from_checkpoint, pack_kiqfl_extra, restore_kiqfl_extra,
    pack_simple_extra, checkpoint_path,
)
from core.fl.metrics import evaluate_comprehensive, make_client_holdout_loaders, extract_labels_from_dataset
from core.fl.logging_utils import Colors, log_kiqfl, log_fedavg, log_fqngd, log_centralized

try:
    from core.fl.privacy import ClientPrivacySession, OPACUS_AVAILABLE
except ImportError:
    ClientPrivacySession = None
    OPACUS_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_fedavg(trainset, testset, config, yield_results=True):
    """
    FedAvg: Classical Federated Averaging with fair comparison mode.
    comparison_mode='fair' (default): Hybrid_QNN + AdamW, same architecture as KIQFL.
    comparison_mode='classical': SimpleCNN + SGD for ablation baseline.

    When use_hybrid_qnn=True, reports round-to-round circuit-state fidelity
    (same definition as KIQFL: F = |⟨ψ^{t-1}|ψ^t⟩|² averaged over clients).
    High F ≈ little state change; low F ≈ active learning — not a quality score.
    """
    config = apply_comparison_defaults(dict(config))
    seed = config.get('seed', 42)
    set_seed(seed)
    num_clients = config['num_clients']
    local_epochs = config['local_epochs']
    global_epochs = config['global_epochs']
    learning_rate = config['learning_rate']
    input_channels = config['input_channels']
    image_size = config['image_size']
    num_classes = config['num_classes']
    batch_size = config.get('batch_size', 32)
    non_iid_alpha = config.get('non_iid_alpha', None)

    use_hybrid_qnn = config.get('use_hybrid_qnn', False)
    num_qubits = config.get('num_qubits', 8)
    num_layers = config.get('num_layers', 2)
    nisq_error_rate = config.get('nisq_error_rate', 0.01)
    comparison_mode = config.get('comparison_mode', 'fair')
    require_circuit_state = config.get('require_circuit_state', True)
    num_fidelity_refs = config.get('num_fidelity_refs', 5)
    # Fair Hybrid must match KIQFL/FQNGD (same QAOA block unless ablated).
    use_qaoa = bool(use_hybrid_qnn) and (not config.get('disable_qaoa', False))
    
    client_indices = partition_dataset(trainset, num_clients, topology='ring',
                                       non_iid_alpha=non_iid_alpha, seed=seed)
    client_datasets = [torch.utils.data.Subset(trainset, idx) for idx in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    client_holdout_loaders = make_client_holdout_loaders(client_datasets, batch_size=batch_size, seed=seed)
    all_labels = extract_labels_from_dataset(trainset)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    if use_hybrid_qnn:
        cnn_backbone = SimpleCNN(input_channels, 2**num_qubits, image_size)
        global_model = Hybrid_QNN(
            cnn_model=cnn_backbone,
            num_qubits=num_qubits,
            num_layers=num_layers,
            num_classes=num_classes,
            use_qaoa=use_qaoa,
            qaoa_p=1,
            nisq_error_rate=nisq_error_rate
        ).to(device)
        method_label = 'FedAvg+HybridQNN' if comparison_mode == 'fair' else 'FedAvg-Hybrid'
        fidelity_refs = build_fidelity_reference_inputs(
            input_channels, image_size, num_fidelity_refs, seed=seed)
    else:
        global_model = SimpleCNN(input_channels, num_classes, image_size).to(device)
        method_label = 'FedAvg-Classical'
        fidelity_refs = None

    criterion = nn.CrossEntropyLoss()
    results = []
    conv_threshold = config.get('convergence_acc_threshold', 0.90)
    convergence_epoch = None
    cumulative_comm_kb = 0.0
    prev_circuit_states = None
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}═══════════════════════════════════════════════════════════")
    print(f"  Starting {method_label} ({comparison_mode}) - {num_clients} clients, {global_epochs} rounds")
    print(f"  Optimizer: {config.get('optimizer', 'adamw')}")
    print(f"═══════════════════════════════════════════════════════════{Colors.END}\n")

    algo_name = 'FedAvg'
    start_epoch = 0
    ckpt = load_midrun_checkpoint(config, algo_name, map_location=device)
    if ckpt is not None and int(ckpt.get('next_epoch', 0)) < global_epochs:
        start_epoch = int(ckpt['next_epoch'])
        global_model.load_state_dict(ckpt['model_state_dict'])
        results = list(ckpt.get('results') or [])
        convergence_epoch = ckpt.get('convergence_epoch')
        cumulative_comm_kb = float(ckpt.get('cumulative_comm_kb') or 0.0)
        extra = ckpt.get('extra') or {}
        pcs = extra.get('prev_circuit_states')
        if pcs is not None:
            prev_circuit_states = [
                (s.to(device) if torch.is_tensor(s) else s) for s in pcs
            ]
        apply_rng_from_checkpoint(ckpt)
        print(f"{Colors.YELLOW}[checkpoint] RESUME FedAvg seed={seed} from epoch "
              f"{start_epoch + 1}/{global_epochs}{Colors.END}")
    
    for epoch in range(start_epoch, global_epochs):
        t_round = time.time()
        local_state_dicts = []
        current_circuit_states = []
        
        for client_idx, client_loader in enumerate(client_loaders):
            if use_hybrid_qnn:
                local_backbone = SimpleCNN(input_channels, 2**num_qubits, image_size)
                local_model = Hybrid_QNN(
                    cnn_model=local_backbone,
                    num_qubits=num_qubits,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    use_qaoa=use_qaoa,
                    qaoa_p=1,
                    nisq_error_rate=nisq_error_rate
                ).to(device)
            else:
                local_model = SimpleCNN(input_channels, num_classes, image_size).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            optimizer = create_fl_optimizer(local_model, config)
            local_model.train()
            
            for local_ep in range(local_epochs):
                for X, y in client_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    # Fair Hybrid mode: same clip as KIQFL quantum training (1.0).
                    # Classical CNN FedAvg keeps a looser clip (5.0).
                    clip_norm = 1.0 if use_hybrid_qnn else 5.0
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=clip_norm)
                    optimizer.step()
            
            if use_hybrid_qnn:
                st = extract_circuit_state(
                    local_model, fidelity_refs, num_qubits,
                    prev_state=(prev_circuit_states[client_idx]
                                if prev_circuit_states is not None else None),
                    require_circuit_state=require_circuit_state)
                if st is not None:
                    current_circuit_states.append(st)
            
            local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
        
        global_model.load_state_dict(
            weighted_average_weights(local_state_dicts, _client_data_sizes(client_indices)))

        if use_hybrid_qnn and current_circuit_states and prev_circuit_states is not None:
            avg_fidelity = compute_multi_fidelity(prev_circuit_states, current_circuit_states)
        else:
            avg_fidelity = 0.0
        if current_circuit_states:
            prev_circuit_states = current_circuit_states

        comm = compute_comm_breakdown(global_model, num_clients, num_qubits,
                                      aggregation_mode='full')
        cumulative_comm_kb += comm.get('comm_effective_KB', 0.0)
        gates = build_nisq_report(num_qubits, num_layers, use_qaoa=use_qaoa,
                                  round_time_s=time.time() - t_round)
        gates['hilbert_dim'] = comm['hilbert_dim']

        eval_m = build_epoch_evaluation(
            global_model, test_loader, criterion, config, epoch, global_epochs,
            comm, avg_fidelity, cumulative_comm_kb, client_holdout_loaders,
            client_indices, all_labels)
        test_loss, accuracy = eval_m['loss'], eval_m['accuracy']
        if convergence_epoch is None and accuracy >= conv_threshold:
            convergence_epoch = epoch + 1

        result = enrich_epoch_result({
            'method': method_label,
            'comparison_mode': comparison_mode,
            'global_epoch': epoch + 1,
            'loss': test_loss,
            'accuracy': accuracy,
            'fidelity': avg_fidelity,
            'convergence_epoch': convergence_epoch,
            **{k: eval_m[k] for k in eval_m if k not in ('loss', 'accuracy', 'confusion_matrix')},
        }, comm=comm, gates=gates)
        results.append(result)

        log_fedavg(epoch + 1, global_epochs, test_loss, accuracy,
                   fidelity=avg_fidelity if use_hybrid_qnn else None)
        append_epoch_log(config, algo_name, result)
        save_midrun_checkpoint(
            config, algo_name,
            next_epoch=epoch + 1,
            global_model=global_model,
            results=results,
            convergence_epoch=convergence_epoch,
            cumulative_comm_kb=cumulative_comm_kb,
            extra=pack_simple_extra(prev_circuit_states=prev_circuit_states),
        )
        # Yield only after durable epoch log + checkpoint (Ctrl+C-safe).
        if yield_results:
            yield result
    
    clear_midrun_checkpoint(config, algo_name)
    return results
