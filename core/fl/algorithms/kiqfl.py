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


def run_kiqfl(trainset, testset, config, yield_results=True):
    """KIQFL: KAN-Enhanced Indirect Quantum Federated Learning."""
    config = apply_comparison_defaults(dict(config))
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
    sporadic_p = config.get('sporadic_p', 0.1)
    sporadic_mode = config.get('sporadic_mode', 'noise_adaptive')
    theta_walk = config.get('theta_walk', math.pi / 6)
    phi_walk = config.get('phi_walk', 0.0)
    nisq_error_rate = config.get('nisq_error_rate', 0.01)
    batch_size = config.get('batch_size', 64)
    non_iid_alpha = config.get('non_iid_alpha', None)

    walk_steps = config.get('walk_steps', 3)
    use_teleportation = config.get('use_teleportation', False)
    use_indirect_aggregation = config.get('use_indirect_aggregation', True)
    aggregation_mode = config.get('aggregation_mode', 'quantum_guided')
    use_dp = config.get('use_dp', False)
    use_state_dp = config.get('use_state_dp', False)
    use_state_only_upload = config.get('use_state_only_upload', False)
    dp_noise_multiplier = config.get('dp_noise_multiplier', 1.0)
    dp_max_grad_norm = config.get('dp_max_grad_norm', 1.0)
    client_participation_rate = config.get('client_participation_rate', 1.0)
    require_circuit_state = config.get('require_circuit_state', True)
    debug_mode = config.get('debug_mode', False)
    num_fidelity_refs = config.get('num_fidelity_refs', 5)
    fidelity_threshold = config.get('fidelity_threshold', 0.90)
    patience = config.get('patience', 5)
    cnn_warmup_epochs = config.get('cnn_warmup_epochs', 3)

    disable_quantum_walk = config.get('disable_quantum_walk', False)
    disable_kan = config.get('disable_kan', False)
    disable_sporadic = config.get('disable_sporadic', False)
    disable_qaoa = config.get('disable_qaoa', False)
    disable_noise = config.get('disable_noise', False)

    if disable_sporadic:
        sporadic_p = 0.0
    if disable_quantum_walk:
        theta_walk = 0.0
        phi_walk = 0.0
    if disable_noise:
        nisq_error_rate = 0.0
    use_qaoa = not disable_qaoa
    
    client_indices = partition_dataset(trainset, num_clients, topology='ring',
                                       non_iid_alpha=non_iid_alpha, seed=seed)
    client_datasets = [torch.utils.data.Subset(trainset, idx) for idx in client_indices]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    client_holdout_loaders = make_client_holdout_loaders(client_datasets, batch_size=batch_size, seed=seed)
    all_labels = extract_labels_from_dataset(trainset)
    
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
    
    criterion = nn.CrossEntropyLoss()
    fidelity_refs = build_fidelity_reference_inputs(
        input_channels, image_size, num_fidelity_refs, seed=seed)

    initial_state = extract_circuit_state(
        global_model, fidelity_refs, num_qubits, require_circuit_state=require_circuit_state,
        debug_mode=debug_mode)
    prev_global_state = initial_state.clone() if initial_state is not None else None

    kan_model = KANEdge(input_dim=3, hidden_dim=16).to(device)
    kan_trainer = KANTrainer(kan_model, learning_rate=0.001)
    early_stopping = EarlyStopping(fidelity_threshold=fidelity_threshold, patience=patience)

    dp_delta = config.get('dp_delta', 1e-5)
    dp_epsilons = []
    
    prev_circuit_states = [initial_state.clone() for _ in range(num_clients)]
    
    results = []
    fidelity_history = []
    convergence_epoch = None
    conv_threshold = config.get('convergence_acc_threshold', 0.90)
    cumulative_comm_kb = 0.0
    freeze_cnn_subnet = config.get('freeze_cnn_subnet', True)

    ablation_flags = []
    if disable_quantum_walk: ablation_flags.append('no_walk')
    if disable_kan: ablation_flags.append('no_kan')
    if disable_sporadic: ablation_flags.append('no_sporadic')
    if disable_qaoa: ablation_flags.append('no_qaoa')
    if disable_noise: ablation_flags.append('no_noise')
    ablation_label = f" [Ablation: {', '.join(ablation_flags)}]" if ablation_flags else ""

    print(f"\n{Colors.CYAN}{Colors.BOLD}═══════════════════════════════════════════════════════════")
    print(f"  Starting KIQFL - {num_clients} clients, {global_epochs} rounds{ablation_label}")
    print(f"  Aggregation: {aggregation_mode}, refs={num_fidelity_refs}")
    print(f"  Sporadic: p={sporadic_p}, mode={sporadic_mode}, θ_walk: {theta_walk:.3f}")
    if use_indirect_aggregation:
        print(f"  Indirect aggregation: ON")
    if use_dp:
        print(f"  Gradient DP: ON (σ={dp_noise_multiplier})")
    print(f"═══════════════════════════════════════════════════════════{Colors.END}\n")

    algo_name = 'KIQFL'
    start_epoch = 0
    ckpt = load_midrun_checkpoint(config, algo_name, map_location=device)
    if ckpt is not None and int(ckpt.get('next_epoch', 0)) < global_epochs:
        start_epoch = int(ckpt['next_epoch'])
        global_model.load_state_dict(ckpt['model_state_dict'])
        results = list(ckpt.get('results') or [])
        convergence_epoch = ckpt.get('convergence_epoch')
        cumulative_comm_kb = float(ckpt.get('cumulative_comm_kb') or 0.0)
        prev_s, prev_g, fidelity_history = restore_kiqfl_extra(
            ckpt.get('extra') or {}, kan_model, kan_trainer, device)
        if prev_s is not None:
            prev_circuit_states = prev_s
        if prev_g is not None:
            prev_global_state = prev_g
        apply_rng_from_checkpoint(ckpt)
        print(f"{Colors.YELLOW}[checkpoint] RESUME KIQFL seed={seed} from epoch "
              f"{start_epoch + 1}/{global_epochs} ({checkpoint_path(config, algo_name)}){Colors.END}")
    elif ckpt is not None and int(ckpt.get('next_epoch', 0)) >= global_epochs:
        clear_midrun_checkpoint(config, algo_name)
    
    for epoch in range(start_epoch, global_epochs):
        t_round = time.time()
        local_state_dicts = []
        current_circuit_states = []
        skipped_clients_local = []
        
        if client_participation_rate < 1.0:
            selected_clients = subsample_clients(num_clients, client_participation_rate)
        else:
            selected_clients = list(range(num_clients))
        
        for client_idx in selected_clients:
            client_dataset = client_datasets[client_idx]
            val_loader = make_client_val_loader(client_dataset, batch_size=batch_size)
            local_model = copy.deepcopy(global_model).to(device)
            if aggregation_mode == 'quantum_subnet' and freeze_cnn_subnet and epoch < cnn_warmup_epochs:
                freeze_cnn_for_subnet(local_model, freeze=True)
            else:
                freeze_cnn_for_subnet(local_model, freeze=False)
            optimizer = create_fl_optimizer(local_model, config)
            dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            client_dp_eps = 0.0
            
            client_skipped_epochs = 0
            if use_dp and ClientPrivacySession is not None:
                try:
                    dp_session = ClientPrivacySession(
                        local_model, optimizer, dataloader,
                        noise_multiplier=dp_noise_multiplier,
                        max_grad_norm=dp_max_grad_norm,
                        delta=dp_delta)
                    local_model = dp_session.model
                    optimizer = dp_session.optimizer
                    dataloader = dp_session.dataloader
                except Exception as exc:
                    print(f"{Colors.YELLOW}[KIQFL] DP disabled for client {client_idx}: {exc}{Colors.END}")
                    use_dp_local = False
                else:
                    use_dp_local = True
            else:
                use_dp_local = False

            local_model.train()
            # Schedule + NISQ intensity for SpoQFL-inspired adaptive skip
            round_noise = compute_noise_levels(epoch, global_epochs)
            for local_ep in range(local_epochs):
                skip_p = sporadic_skip_probability(
                    sporadic_p, noise_level=round_noise,
                    nisq_error_rate=nisq_error_rate, mode=sporadic_mode)
                if random.random() < skip_p:
                    client_skipped_epochs += 1
                    continue
                
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    if not use_dp_local:
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    # Optional mid-batch adaptive skip using grad noise intensity
                    if sporadic_mode == 'noise_adaptive' and sporadic_p > 0:
                        gnoise = estimate_grad_noise(local_model)
                        batch_skip = sporadic_skip_probability(
                            sporadic_p * 0.25, noise_level=round_noise,
                            nisq_error_rate=nisq_error_rate, grad_noise=gnoise,
                            mode=sporadic_mode)
                        if random.random() < batch_skip:
                            optimizer.zero_grad(set_to_none=True)
                            continue
                    optimizer.step()
            
            if use_dp_local:
                client_dp_eps = dp_session.epsilon()
                dp_epsilons.append(client_dp_eps)
            
            if use_state_only_upload:
                sd = {k: v for k, v in local_model.state_dict().items()
                      if any(q in k for q in ('angles', 'gammas', 'betas', 'output_layer', 'linear'))}
                local_state_dicts.append(sd)
            else:
                local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            
            if client_skipped_epochs > 0:
                skipped_clients_local.append(f"C{client_idx}:{client_skipped_epochs}ep")
            
            circuit_state = extract_circuit_state(
                local_model, fidelity_refs, num_qubits,
                prev_state=prev_circuit_states[client_idx],
                require_circuit_state=require_circuit_state,
                debug_mode=debug_mode)
            current_circuit_states.append(circuit_state)
        
        if use_state_dp and config.get('state_dp_sigma', 0.0) > 0:
            current_circuit_states = add_state_dp_noise(
                current_circuit_states, config.get('state_dp_sigma', 0.01))

        noise_level = compute_noise_levels(epoch, global_epochs)
        
        if not disable_quantum_walk:
            walked_states = quantum_walk_transfer_state_list(
                current_circuit_states, theta_walk, phi_walk,
                walk_steps=walk_steps,
                use_teleportation=use_teleportation
            )
        else:
            walked_states = [s.clone() for s in current_circuit_states]
        
        local_states_before_mix = [s.clone() if s is not None else None for s in current_circuit_states]
        new_circuit_states = []
        skipped_clients_walk = []
        for i, (state, walked) in enumerate(zip(current_circuit_states, walked_states)):
            ci = selected_clients[i]
            walk_skip_p = sporadic_skip_probability(
                sporadic_p, noise_level=noise_level,
                nisq_error_rate=nisq_error_rate, mode=sporadic_mode)
            if random.random() < walk_skip_p:
                new_circuit_states.append(state.clone())
                skipped_clients_walk.append(f"C{ci}")
            elif disable_kan:
                new_state = apply_adaptive_weight_to_state(state, walked, 0.5)
                new_circuit_states.append(new_state)
            else:
                fid_i = compute_real_fidelity([prev_circuit_states[ci]], [state])
                t_norm = epoch / max(1, global_epochs)
                alpha = compute_adaptive_alpha(noise_level, fid_i, t_norm, kan_model, device)
                new_state = apply_adaptive_weight_to_state(state, walked, alpha)
                new_circuit_states.append(new_state)
        
        current_circuit_states = new_circuit_states
        
        prev_selected = [prev_circuit_states[selected_clients[i]] for i in range(len(selected_clients))]
        avg_fidelity = compute_multi_fidelity(prev_selected, current_circuit_states)
        
        for i in range(len(selected_clients)):
            ci = selected_clients[i]
            client_dataset = client_datasets[ci]
            val_loader = make_client_val_loader(client_dataset, batch_size=batch_size)
            local_tmp = copy.deepcopy(global_model).to(device)
            local_tmp.load_state_dict(local_state_dicts[i] if not use_state_only_upload
                                      else {**global_model.state_dict(), **local_state_dicts[i]})
            alpha_target = grid_search_alpha_target(
                local_states_before_mix[i], walked_states[i], prev_global_state,
                model=local_tmp, val_loader=val_loader, criterion=criterion, device_arg=device)
            if disable_kan:
                alpha_target = 0.5
            else:
                fidelity_factor = 1.0 - avg_fidelity
                noise_factor = 1.0 - noise_level
                progress_factor = 1.0 - (epoch / global_epochs)
                heuristic = 0.5 * fidelity_factor + 0.3 * noise_factor + 0.2 * progress_factor
                alpha_target = 0.7 * alpha_target + 0.3 * max(0.0, min(1.0, heuristic))
            kan_trainer.add_sample(noise_level, avg_fidelity, epoch / global_epochs, alpha_target)
        
        if not disable_kan and kan_trainer.get_samples_count() >= 16:
            kan_trainer.train_step(batch_size=min(16, kan_trainer.get_samples_count()))
        
        agg_weights = None
        if use_indirect_aggregation and len(current_circuit_states) > 0:
            mode = aggregation_mode if use_indirect_aggregation else 'full'
            agg_state_dict, agg_weights, global_qstate = aggregate_with_mode(
                local_state_dicts, current_circuit_states, prev_global_state,
                global_model.state_dict(), num_qubits, aggregation_mode=mode,
                epoch=epoch, cnn_warmup_epochs=cnn_warmup_epochs)
            if use_state_only_upload:
                merged = copy.deepcopy(global_model.state_dict())
                merged.update(agg_state_dict)
                global_model.load_state_dict(merged)
            else:
                global_model.load_state_dict(agg_state_dict)
            # Next-round fidelity reference = circuit state of the *aggregated* global model
            prev_global_state = extract_circuit_state(
                global_model, fidelity_refs, num_qubits,
                prev_state=global_qstate,
                require_circuit_state=require_circuit_state,
                debug_mode=debug_mode)
        else:
            # Fallback without indirect aggregation: McMahan n_k/N on participating clients.
            sizes = [len(client_indices[ci]) for ci in selected_clients]
            dicts = ([{**global_model.state_dict(), **sd} for sd in local_state_dicts]
                     if use_state_only_upload else local_state_dicts)
            global_model.load_state_dict(weighted_average_weights(dicts, sizes))
        
        round_epsilon = max(dp_epsilons) if dp_epsilons else 0.0
        cumulative_epsilon = round_epsilon

        comm = compute_comm_breakdown(
            global_model, num_clients, num_qubits,
            aggregation_mode=aggregation_mode if use_indirect_aggregation else 'full',
            num_participating=len(selected_clients))
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

        fidelity_history.append((avg_fidelity, accuracy))
        result = enrich_epoch_result({
            'method': 'KIQFL',
            'global_epoch': epoch + 1,
            'loss': test_loss,
            'accuracy': accuracy,
            'fidelity': avg_fidelity,
            'dp_epsilon': round_epsilon,
            'dp_cumulative_epsilon': cumulative_epsilon,
            'dp_delta': dp_delta,
            'aggregation_mode': aggregation_mode,
            'convergence_epoch': convergence_epoch,
            'state_to_param_ratio': comm.get('state_to_param_ratio', 0.0),
            **{k: eval_m[k] for k in eval_m if k not in ('loss', 'accuracy', 'confusion_matrix')},
        }, comm=comm, gates=gates, agg_weights=agg_weights,
           fidelity_history=fidelity_history)
        results.append(result)

        all_skipped = skipped_clients_local + [f"{c}(walk)" for c in skipped_clients_walk]
        log_kiqfl(epoch + 1, global_epochs, test_loss, accuracy, avg_fidelity, all_skipped)

        for idx_i, ci in enumerate(selected_clients):
            if idx_i < len(current_circuit_states):
                prev_circuit_states[ci] = current_circuit_states[idx_i].clone()

        append_epoch_log(config, algo_name, result)
        save_midrun_checkpoint(
            config, algo_name,
            next_epoch=epoch + 1,
            global_model=global_model,
            results=results,
            convergence_epoch=convergence_epoch,
            cumulative_comm_kb=cumulative_comm_kb,
            extra=pack_kiqfl_extra(
                kan_model, kan_trainer, prev_circuit_states,
                prev_global_state, fidelity_history),
        )
        if yield_results:
            yield result

        if config.get('use_early_stopping', False) and early_stopping.should_stop(
                avg_fidelity, loss=test_loss, accuracy=accuracy,
                model=global_model, epoch=epoch + 1):
            print(f"{Colors.CYAN}[KIQFL]{Colors.END} {Colors.YELLOW}Early stopping at epoch {epoch+1}{Colors.END}")
            break

    clear_midrun_checkpoint(config, algo_name)
    return results
