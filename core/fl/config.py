import copy
import math
import torch.optim as optim

def apply_comparison_defaults(config):
    """Normalize config for fair fair comparisons."""
    mode = config.get('comparison_mode', 'fair')
    if mode == 'fair':
        config.setdefault('use_hybrid_qnn', True)
        config.setdefault('optimizer', 'adamw')
        config.setdefault('nisq_error_rate', config.get('nisq_error_rate', 0.01))
    else:
        config.setdefault('use_hybrid_qnn', False)
        config.setdefault('optimizer', 'sgd')
    config.setdefault('aggregation_mode', 'quantum_guided')
    config.setdefault('walk_steps', 3)
    config.setdefault('require_circuit_state', True)
    config.setdefault('debug_mode', False)
    config.setdefault('num_fidelity_refs', 3)
    config.setdefault('fidelity_threshold', 0.90)
    config.setdefault('patience', 5)
    config.setdefault('use_early_stopping', False)
    config.setdefault('use_state_only_upload', False)
    config.setdefault('use_state_dp', False)
    config.setdefault('report_nisq_cost', True)
    config.setdefault('cnn_warmup_epochs', 3)
    config.setdefault('convergence_acc_threshold', 0.90)
    config.setdefault('fisher_method', 'parameter_shift')
    config.setdefault('fisher_structure', 'layer_block')
    config.setdefault('fqngd_aggregation', 'natural_grad')
    config.setdefault('fisher_period', 'per_local_epoch')
    config.setdefault('sporadic_mode', 'noise_adaptive')
    config.setdefault('freeze_cnn_subnet', True)
    config.setdefault('report_detailed_metrics', True)
    return config

def create_fl_optimizer(model, config):
    lr = config['learning_rate']
    opt_name = config.get('optimizer', 'adamw').lower()
    if opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

def default_config(**overrides):
    """
    Defense / ~18–25h quality default profile (fair Hybrid-QNN comparison).

    Locked: fisher_method=parameter_shift, fisher_structure=layer_block, num_qubits=8.
    FQNGD cost control: fisher_period='per_local_epoch' reuses a representative-batch
    parameter-shift Fisher for remaining mini-batches of that local epoch.

    Quality ladder vs the ~7h draft: data_fraction=0.50, main=60 ep, abl/noise=35,
    local_epochs=2 (est. ~21h GPU; still ≤30h ceiling).
    """
    cfg = {
        'comparison_mode': 'fair',
        'use_hybrid_qnn': True,
        'optimizer': 'adamw',
        'aggregation_mode': 'quantum_guided',
        'walk_steps': 3,
        'use_indirect_aggregation': True,
        'require_circuit_state': True,
        'num_fidelity_refs': 3,
        'fidelity_threshold': 0.90,
        'patience': 5,
        'use_early_stopping': False,  # full global_epochs for training curves / fair baselines
        'report_nisq_cost': True,
        'num_clients': 10,
        'local_epochs': 2,
        'global_epochs': 60,
        'learning_rate': 0.01,
        'num_qubits': 8,
        'num_layers': 2,
        'sporadic_p': 0.1,
        'sporadic_mode': 'noise_adaptive',
        'theta_walk': math.pi / 6,
        'phi_walk': 0.0,
        'nisq_error_rate': 0.01,
        'batch_size': 128,
        'non_iid_alpha': 1.0,
        'data_fraction': 0.50,
        'seed': 42,
        'fisher_method': 'parameter_shift',
        'fisher_structure': 'layer_block',
        'fqngd_aggregation': 'natural_grad',
        # every_batch | per_local_epoch | per_round — never changes fisher_method/structure
        'fisher_period': 'per_local_epoch',
        'use_dp': False,
        'use_state_dp': False,
        'use_teleportation': False,
    }
    cfg.update(overrides)
    return apply_comparison_defaults(cfg)
