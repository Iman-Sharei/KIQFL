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


def run_fqngd(trainset, testset, config, yield_results=True):
    """
    FQNGD baseline aligned with Qi et al. (ICASSP 2023) Algorithm 1 / Eq. (13):
      θ̄ ← θ̄ − η Σ_k (n_k/N) g_k^{+}(θ) ∇L_k

    Locally: layer-block (or diagonal) Fisher preconditioning on quantum params
    (parameter-shift diagonal blended into blocks). Server aggregates *natural
    gradients* with McMahan weights — not parameter averaging.

    Still Hybrid_QNN for fair comparison with KIQFL; not bit-exact pure-VQC
    Fubini–Study generator expectations from the original paper.
    """
    config = apply_comparison_defaults(dict(config))
    # Paper uses metric-preconditioned GD, not Adam moments on top of NG.
    config['optimizer'] = config.get('fqngd_optimizer', 'sgd')
    seed = config.get('seed', 42)
    set_seed(seed)
    num_clients = config['num_clients']
    local_epochs = config['local_epochs']
    global_epochs = config['global_epochs']
    learning_rate = config['learning_rate']
    num_qubits = config['num_qubits']
    num_layers = config['num_layers']
    input_channels = config['input_channels']
    image_size = config['image_size']
    num_classes = config['num_classes']
    nisq_error_rate = config.get('nisq_error_rate', 0.01)
    batch_size = config.get('batch_size', 32)
    non_iid_alpha = config.get('non_iid_alpha', None)
    require_circuit_state = config.get('require_circuit_state', True)
    debug_mode = config.get('debug_mode', False)
    num_fidelity_refs = config.get('num_fidelity_refs', 5)
    conv_threshold = config.get('convergence_acc_threshold', 0.90)
    fisher_method = config.get('fisher_method', 'parameter_shift')
    fisher_structure = config.get('fisher_structure', 'layer_block')
    fqngd_aggregation = config.get('fqngd_aggregation', 'natural_grad')
    # Cost control only: reuse parameter-shift Fisher within a local epoch / round.
    # Never changes fisher_method or fisher_structure.
    fisher_period = config.get('fisher_period', 'per_local_epoch')
    
    client_indices = partition_dataset(trainset, num_clients, topology='ring',
                                       non_iid_alpha=non_iid_alpha, seed=seed)
    client_datasets = [torch.utils.data.Subset(trainset, idx) for idx in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    client_holdout_loaders = make_client_holdout_loaders(client_datasets, batch_size=batch_size, seed=seed)
    all_labels = extract_labels_from_dataset(trainset)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    data_sizes = _client_data_sizes(client_indices)
    
    cnn_backbone = SimpleCNN(input_channels, 2**num_qubits, image_size)
    global_model = Hybrid_QNN(
        cnn_model=cnn_backbone,
        num_qubits=num_qubits,
        num_layers=num_layers,
        num_classes=num_classes,
        use_qaoa=True,
        qaoa_p=1,
        nisq_error_rate=nisq_error_rate
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    results = []
    convergence_epoch = None
    fidelity_refs = build_fidelity_reference_inputs(
        input_channels, image_size, num_fidelity_refs, seed=seed)

    initial_state = extract_circuit_state(
        global_model, fidelity_refs, num_qubits,
        require_circuit_state=require_circuit_state, debug_mode=debug_mode)
    prev_circuit_states = [initial_state.clone() for _ in range(num_clients)]
    
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}═══════════════════════════════════════════════════════════")
    print(f"  Starting FQNGD - {num_clients} clients, {global_epochs} rounds")
    print(f"  Aggregation: {fqngd_aggregation} | Fisher: {fisher_method}/{fisher_structure}")
    print(f"  Fisher period: {fisher_period} (parameter-shift sampled, not every batch)")
    print(f"  Optimizer: {config['optimizer']} (Qi-style natural-grad federated update)")
    print(f"═══════════════════════════════════════════════════════════{Colors.END}\n")
    
    cumulative_comm_kb = 0.0
    algo_name = 'FQNGD'
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
        print(f"{Colors.YELLOW}[checkpoint] RESUME FQNGD seed={seed} from epoch "
              f"{start_epoch + 1}/{global_epochs}{Colors.END}")

    for epoch in range(start_epoch, global_epochs):
        t_round = time.time()
        local_state_dicts = []
        local_nat_grads = []
        current_circuit_states = []
        
        for client_idx, client_loader in enumerate(client_loaders):
            local_cnn = SimpleCNN(input_channels, 2**num_qubits, image_size)
            local_model = Hybrid_QNN(
                cnn_model=local_cnn,
                num_qubits=num_qubits,
                num_layers=num_layers,
                num_classes=num_classes,
                use_qaoa=True,
                qaoa_p=1,
                nisq_error_rate=nisq_error_rate
            ).to(device)
            
            local_model.load_state_dict(global_model.state_dict())
            # Local steps only accumulate / refine natural grads; server applies Eq. (13).
            # When aggregating parameters (legacy), still use local SGD after preconditioning.
            optimizer = create_fl_optimizer(local_model, config)
            local_model.train()
            ng_acc = None
            n_ng_steps = 0
            
            cached_fisher = None
            for local_ep in range(local_epochs):
                if fisher_period == 'per_local_epoch':
                    cached_fisher = None  # recompute on first batch of each local epoch
                for batch_idx, (X, y) in enumerate(client_loader):
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    # Representative-batch Fisher: compute parameter-shift once, reuse.
                    if fisher_method == 'parameter_shift':
                        if fisher_period == 'every_batch':
                            need_ps = True
                        elif fisher_period == 'per_round':
                            need_ps = (local_ep == 0 and batch_idx == 0)
                        else:  # per_local_epoch (default defense profile)
                            need_ps = (batch_idx == 0) or (cached_fisher is None)
                        sample_batch = (X, y) if need_ps else None
                        fisher_arg = None if need_ps else cached_fisher
                    else:
                        sample_batch = None
                        fisher_arg = None
                    cached_fisher = apply_quantum_natural_gradient(
                        local_model, sample_batch, criterion, device,
                        method=fisher_method, structure=fisher_structure,
                        fisher_diag=fisher_arg)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    grads = extract_grad_state_dict(local_model)
                    if ng_acc is None:
                        ng_acc = zeros_like_grad_dict(grads)
                    accumulate_grad_dict(ng_acc, grads)
                    n_ng_steps += 1
                    if fqngd_aggregation == 'param_avg':
                        # Legacy: local NG-preconditioned steps + n_k/N param average
                        optimizer.step()
                    # else: Algorithm 1 — only upload mean g^{+}∇L; server applies Eq. (13)
            
            if ng_acc is None:
                ng_acc = extract_grad_state_dict(local_model)
                n_ng_steps = 1
            local_nat_grads.append(average_grad_dict(ng_acc, n_ng_steps))
            local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
        
        if fqngd_aggregation == 'param_avg':
            global_model.load_state_dict(
                weighted_average_weights(local_state_dicts, data_sizes))
        else:
            # Eq. (13): θ̄ ← θ̄ − η Σ (n_k/N) g_k^{+} ∇L_k
            agg_ng = weighted_average_grad_dicts(local_nat_grads, data_sizes)
            apply_grad_dict_to_model(global_model, agg_ng, learning_rate)

        # Round-to-round fidelity from post-aggregation global circuit state
        current_circuit_states = []
        for client_idx in range(num_clients):
            circuit_state = extract_circuit_state(
                global_model, fidelity_refs, num_qubits,
                prev_state=prev_circuit_states[client_idx],
                require_circuit_state=require_circuit_state,
                debug_mode=debug_mode)
            current_circuit_states.append(circuit_state)

        avg_fidelity = compute_multi_fidelity(prev_circuit_states, current_circuit_states)

        comm = compute_comm_breakdown(global_model, num_clients, num_qubits, aggregation_mode='full')
        cumulative_comm_kb += comm.get('comm_effective_KB', 0.0)
        gates = build_nisq_report(num_qubits, num_layers, use_qaoa=True,
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
            'method': 'FQNGD',
            'global_epoch': epoch + 1,
            'loss': test_loss,
            'accuracy': accuracy,
            'fidelity': avg_fidelity,
            'fisher_method': fisher_method,
            'fisher_structure': fisher_structure,
            'fisher_period': fisher_period,
            'fqngd_aggregation': fqngd_aggregation,
            'convergence_epoch': convergence_epoch,
            **{k: eval_m[k] for k in eval_m if k not in ('loss', 'accuracy', 'confusion_matrix')},
        }, comm=comm, gates=gates)
        results.append(result)

        log_fqngd(epoch + 1, global_epochs, test_loss, accuracy, avg_fidelity)
        prev_circuit_states = [state.clone() for state in current_circuit_states]
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
        if yield_results:
            yield result

    clear_midrun_checkpoint(config, algo_name)
    return results
