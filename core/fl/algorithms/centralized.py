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


def run_centralized(trainset, testset, config, yield_results=True):
    """Centralized upper-bound baseline with Hybrid CNN+QNN."""
    config = apply_comparison_defaults(dict(config))
    seed = config.get('seed', 42)
    set_seed(seed)
    global_epochs = config['global_epochs']
    learning_rate = config['learning_rate']
    num_qubits = config['num_qubits']
    num_layers = config['num_layers']
    input_channels = config['input_channels']
    image_size = config['image_size']
    num_classes = config['num_classes']
    nisq_error_rate = config.get('nisq_error_rate', 0.01)
    batch_size = config.get('batch_size', 64)
    num_fidelity_refs = config.get('num_fidelity_refs', 5)
    conv_threshold = config.get('convergence_acc_threshold', 0.90)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    cnn_backbone = SimpleCNN(input_channels, 2**num_qubits, image_size)
    model = Hybrid_QNN(
        cnn_model=cnn_backbone,
        num_qubits=num_qubits,
        num_layers=num_layers,
        num_classes=num_classes,
        use_qaoa=True,
        qaoa_p=1,
        nisq_error_rate=nisq_error_rate
    ).to(device)
    
    optimizer = create_fl_optimizer(model, config)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    
    fidelity_refs = build_fidelity_reference_inputs(
        input_channels, image_size, num_fidelity_refs, seed=seed)
    prev_state = extract_circuit_state(
        model, fidelity_refs, num_qubits, require_circuit_state=False, debug_mode=True)
    
    results = []
    convergence_epoch = None
    cumulative_comm_kb = 0.0
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════════════")
    print(f"  Starting Centralized - {global_epochs} epochs (Upper Bound)")
    print(f"═══════════════════════════════════════════════════════════{Colors.END}\n")

    algo_name = 'Centralized'
    start_epoch = 0
    ckpt = load_midrun_checkpoint(config, algo_name, map_location=device)
    if ckpt is not None and int(ckpt.get('next_epoch', 0)) < global_epochs:
        start_epoch = int(ckpt['next_epoch'])
        model.load_state_dict(ckpt['model_state_dict'])
        results = list(ckpt.get('results') or [])
        convergence_epoch = ckpt.get('convergence_epoch')
        cumulative_comm_kb = float(ckpt.get('cumulative_comm_kb') or 0.0)
        extra = ckpt.get('extra') or {}
        if 'optimizer_state_dict' in extra:
            try:
                optimizer.load_state_dict(extra['optimizer_state_dict'])
            except Exception:
                pass
        if 'scheduler_state_dict' in extra:
            try:
                scheduler.load_state_dict(extra['scheduler_state_dict'])
            except Exception:
                pass
        ps = extra.get('prev_state')
        if torch.is_tensor(ps):
            prev_state = ps.to(device)
        apply_rng_from_checkpoint(ckpt)
        print(f"{Colors.YELLOW}[checkpoint] RESUME Centralized seed={seed} from epoch "
              f"{start_epoch + 1}/{global_epochs}{Colors.END}")
    
    for epoch in range(start_epoch, global_epochs):
        t_round = time.time()
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        current_state = extract_circuit_state(
            model, fidelity_refs, num_qubits, prev_state=prev_state,
            require_circuit_state=False, debug_mode=True)
        avg_fidelity = compute_multi_fidelity([prev_state], [current_state])
        prev_state = current_state.clone()
        
        comm = compute_comm_breakdown(model, 1, num_qubits, aggregation_mode='full', num_participating=1)
        cumulative_comm_kb = comm.get('comm_effective_KB', 0.0) * (epoch + 1)
        gates = build_nisq_report(num_qubits, num_layers, use_qaoa=True,
                                  round_time_s=time.time() - t_round)
        gates['hilbert_dim'] = comm['hilbert_dim']

        eval_m = build_epoch_evaluation(
            model, test_loader, criterion, config, epoch, global_epochs,
            comm, avg_fidelity, cumulative_comm_kb)
        test_loss, accuracy = eval_m['loss'], eval_m['accuracy']
        if convergence_epoch is None and accuracy >= conv_threshold:
            convergence_epoch = epoch + 1

        result = enrich_epoch_result({
            'method': 'Centralized',
            'global_epoch': epoch + 1,
            'loss': test_loss,
            'accuracy': accuracy,
            'fidelity': avg_fidelity,
            'convergence_epoch': convergence_epoch,
            **{k: eval_m[k] for k in eval_m if k not in ('loss', 'accuracy', 'confusion_matrix')},
        }, comm=comm, gates=gates)
        results.append(result)

        log_centralized(epoch + 1, global_epochs, test_loss, accuracy, avg_fidelity)
        append_epoch_log(config, algo_name, result)
        save_midrun_checkpoint(
            config, algo_name,
            next_epoch=epoch + 1,
            global_model=model,
            results=results,
            convergence_epoch=convergence_epoch,
            cumulative_comm_kb=cumulative_comm_kb,
            extra=pack_simple_extra(
                prev_state=prev_state, optimizer=optimizer, scheduler=scheduler),
        )
        if yield_results:
            yield result

    clear_midrun_checkpoint(config, algo_name)
    return results
