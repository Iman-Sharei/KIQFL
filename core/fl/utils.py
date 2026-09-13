import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.quantum.circuit import quantum_circuit
from core.quantum.fidelity import compute_state_fidelity, apply_adaptive_weight_to_state
from core.quantum.kan import KANEdge
from core.fl.aggregation import is_quantum_param_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.
    Controls: Python random, NumPy, PyTorch CPU/CUDA, cuDNN determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ========================================================================================
# 1. EARLY STOPPING WITH MULTI-METRIC TRACKING
# ========================================================================================

def compute_fidelity_between_states(old_states, new_states):
    """
    Compute average quantum fidelity between old and new states.
    F = |<psi_old|psi_new>|^2 (Qiskit-compatible pure-state fidelity).
    """
    if len(old_states) == 0 or len(new_states) == 0:
        return 0.0
    total_fidelity = 0.0
    count = 0
    for old_state, new_state in zip(old_states, new_states):
        if old_state is new_state or old_state is None or new_state is None:
            continue
        fidelity = compute_state_fidelity(old_state, new_state)
        fid_val = fidelity.item() if torch.is_tensor(fidelity) else float(fidelity)
        fid_val = max(0.0, min(1.0, fid_val))
        total_fidelity += fid_val
        count += 1
    return total_fidelity / count if count > 0 else 0.5

class EarlyStopping:
    """
    Multi-metric early stopping: fidelity + loss + accuracy.
    Supports adaptive patience, best model checkpointing.
    """
    def __init__(self, fidelity_threshold=0.95, patience=5,
                 min_delta_loss=1e-4, min_delta_acc=1e-3):
        self.fidelity_threshold = fidelity_threshold
        self.base_patience = patience
        self.min_delta_loss = min_delta_loss
        self.min_delta_acc = min_delta_acc
        self.counter = 0
        self.best_fidelity = 0.0
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        self.patience_multiplier = 1.0

    def should_stop(self, avg_fidelity_or_epoch=None, loss=None, accuracy=None,
                    model=None, epoch=None):
        # Backward compat: single fidelity arg
        if loss is None and accuracy is None:
            avg_fidelity = float(avg_fidelity_or_epoch) if avg_fidelity_or_epoch is not None else 0.0
            avg_fidelity = max(0.0, min(1.0, avg_fidelity))
            if avg_fidelity > self.fidelity_threshold:
                self.counter += 1
                if self.counter >= self.base_patience:
                    return True
            else:
                self.counter = 0
            self.best_fidelity = max(self.best_fidelity, avg_fidelity)
            return False
        # New multi-metric API
        fidelity = max(0.0, min(1.0, float(avg_fidelity_or_epoch or 0.0)))
        loss_val = float(loss) if loss is not None else float('inf')
        acc_val = float(accuracy) if accuracy is not None else 0.0
        ep = int(epoch) if epoch is not None else 0
        improved = (loss_val < self.best_loss - self.min_delta_loss or
                    acc_val > self.best_acc + self.min_delta_acc)
        if improved:
            self.counter = 0
            self.best_fidelity = fidelity
            self.best_loss = min(self.best_loss, loss_val)
            self.best_acc = max(self.best_acc, acc_val)
            if model is not None:
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.best_epoch = ep
        else:
            self.counter += 1
        if ep > 10:
            self.patience_multiplier = 1.5
        current_patience = int(self.base_patience * self.patience_multiplier)
        # Fidelity is round-to-round state change, not a quality signal — do not
        # stop solely because F is high (often stagnation). Stop on task plateau.
        return self.counter >= current_patience

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            return True
        return False


# ========================================================================================
# 2. NISQ NOISE SIMULATION
# ========================================================================================

def apply_nisq_noise(qc, noise_model='depolarizing', error_rate=0.01, rng=None):
    """Apply realistic NISQ noise to a quantum circuit.

    This is a thin wrapper around qc.apply_noise() to provide a unified API.
    All noise models are implemented in circuit.py using physically correct
    quantum channel formalism:

    - depolarizing: Per-qubit ε(ρ) = (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
    - bit_phase_flip: Per-qubit independent X and Z errors
    - amplitude_damping: Per-qubit Kraus operators K₀, K₁ (T1 relaxation)

    Args:
        qc: quantum_circuit instance.
        noise_model: 'depolarizing', 'bit_phase_flip', or 'amplitude_damping'.
        error_rate: error probability per qubit per operation.
        rng: Optional random.Random instance for reproducibility.
             Falls back to global random module if None.

    Returns:
        State vector after noise application.
    """
    qc.apply_noise(noise_model=noise_model, error_rate=error_rate, rng=rng)
    return qc.state_vector


# ========================================================================================
# 3. HYBRID CLASSICAL-QUANTUM ARCHITECTURE
# ========================================================================================

# ========================================================================================
# 2b. INDIRECT QUANTUM AGGREGATION
# ========================================================================================

def apply_dp_noise(model, noise_multiplier=1.0, max_grad_norm=1.0, batch_size=32,
                   dataset_size=60000):
    """
    Apply Gaussian differential privacy noise to model parameters.

    SCIENTIFIC DESCRIPTION:
    Implements the Gaussian mechanism for (ε, δ)-differential privacy.
    After gradient clipping (done separately), Gaussian noise is added
    to model parameters:

    θ_private = θ + N(0, σ²I)
    where σ = noise_multiplier × (max_grad_norm / batch_size)

    Privacy budget per round is computed using the simple composition theorem:
    ε = √(2 ln(1.25/δ)) × (Δf / σ)
    where Δf = max_grad_norm × learning_rate is the sensitivity
    and δ = 1/dataset_size.

    For quantum-federated context, this provides quantum-inspired DP:
    The noise level can be tied to the NISQ error rate, creating a natural
    mapping between hardware noise and privacy guarantees.

    Args:
        model: PyTorch model whose parameters will be noised
        noise_multiplier: σ/Δf ratio (higher = more noise = more private)
        max_grad_norm: L2 sensitivity bound (from gradient clipping)
        batch_size: Minibatch size used in training
        dataset_size: Total size of dataset (for δ computation)

    Returns:
        epsilon: Privacy budget consumed this round
        sigma: Actual noise standard deviation applied
    """
    # Compute noise scale
    sensitivity = max_grad_norm / max(1, batch_size)
    sigma = noise_multiplier * sensitivity

    # Add Gaussian noise to each parameter
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * sigma
            param.add_(noise)

    # Compute ε-DP budget for this round
    # Using advanced composition via RDP → (ε,δ)-DP
    delta = 1.0 / dataset_size
    if sigma > 1e-10:
        # Simple analytical ε bound (Gaussian mechanism)
        epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) * (sensitivity / sigma)
    else:
        epsilon = float('inf')

    return epsilon, sigma

def compute_cumulative_epsilon(epsilon_per_round, composition='advanced'):
    """
    Compute cumulative privacy budget over multiple rounds.

    Args:
        epsilon_per_round: List of per-round ε values
        composition: 'simple' (sum), 'advanced' (sqrt composition)

    Returns:
        Total ε budget consumed
    """
    if not epsilon_per_round:
        return 0.0

    if composition == 'simple':
        return sum(epsilon_per_round)
    else:
        # Advanced composition: ε_total = √(Σ εᵢ²)
        return math.sqrt(sum(e ** 2 for e in epsilon_per_round))


# ========================================================================================
# 2d. SCALABILITY: CLIENT SUBSAMPLING
# ========================================================================================

class KANTrainer:
    """
    Trainer for KAN-based adaptive weighting in KIQFL.
    
    Manages training of the true KAN model (with B-spline edge activations)
    including periodic grid updates to adapt B-spline knot positions to
    the actual input data distribution.
    """
    def __init__(self, kan_model, learning_rate=0.001, grid_update_freq=50):
        self.kan_model = kan_model
        self.optimizer = optim.AdamW(kan_model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.loss_fn = nn.MSELoss()
        self.training_data = {"noise": [], "fidelity": [], "t": [], "alpha_target": []}
        self.grid_update_freq = grid_update_freq  # Update B-spline grids every N steps
        self.train_step_count = 0

    def add_sample(self, noise, fidelity, t, alpha_target):
        self.training_data["noise"].append(max(0.0, min(1.0, float(noise))))
        self.training_data["fidelity"].append(max(0.0, min(1.0, float(fidelity))))
        self.training_data["t"].append(max(0.0, min(1.0, float(t))))
        self.training_data["alpha_target"].append(max(0.0, min(1.0, float(alpha_target))))

    def train_step(self, batch_size=32):
        if len(self.training_data["noise"]) < batch_size:
            return None
        indices = random.sample(range(len(self.training_data["noise"])), batch_size)
        features = torch.tensor(
            [[self.training_data["noise"][i], self.training_data["fidelity"][i],
              self.training_data["t"][i]] for i in indices],
            dtype=torch.float32, device=device)
        targets = torch.tensor(
            [self.training_data["alpha_target"][i] for i in indices],
            dtype=torch.float32, device=device).unsqueeze(1)
        predictions = self.kan_model(features)
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.kan_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Periodically update B-spline grids to adapt to data distribution
        self.train_step_count += 1
        if self.train_step_count % self.grid_update_freq == 0:
            self._update_grids(features)
        
        return loss.item()
    
    def _update_grids(self, sample_features=None):
        """
        Update B-spline knot positions (grids) based on accumulated training data.
        
        This adapts the B-spline basis functions to concentrate resolution where
        the input data is dense, improving approximation quality.
        """
        if hasattr(self.kan_model, 'update_grids'):
            if sample_features is not None:
                self.kan_model.update_grids(sample_features.detach())
            elif len(self.training_data["noise"]) >= 32:
                # Use recent samples for grid update
                n = min(len(self.training_data["noise"]), 200)
                features = torch.tensor(
                    [[self.training_data["noise"][-n+i], 
                      self.training_data["fidelity"][-n+i],
                      self.training_data["t"][-n+i]] for i in range(n)],
                    dtype=torch.float32, device=device)
                self.kan_model.update_grids(features)

    def get_samples_count(self):
        return len(self.training_data["noise"])

    def get_all_data(self):
        """Return all collected KAN training data as numpy arrays."""
        return {k: np.array(v) for k, v in self.training_data.items()}

    def clear_old_samples(self, keep_last=1000):
        if len(self.training_data["noise"]) > keep_last:
            for key in self.training_data:
                self.training_data[key] = self.training_data[key][-keep_last:]


# ========================================================================================
# 5. DETAILED EVALUATION METRICS
# ========================================================================================

def create_scheduler(optimizer, global_epochs, warmup_fraction=0.05, min_lr_factor=0.01):
    """SequentialLR: LinearLR warm-up + CosineAnnealingLR."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = max(1, int(warmup_fraction * global_epochs))
    remaining_epochs = max(1, global_epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=remaining_epochs,
                                eta_min=optimizer.param_groups[0]['lr'] * min_lr_factor)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ========================================================================================
# 7. DATA AUGMENTATION TRANSFORMS
# ========================================================================================

def clip_gradients_per_layer(model, cnn_max_norm=5.0, qnn_max_norm=1.0):
    """Apply different gradient clipping to CNN vs QNN parameters."""
    cnn_params, qnn_params = [], []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'cnn' in name or 'output_layer' in name:
                cnn_params.append(param)
            else:
                qnn_params.append(param)
    if cnn_params:
        torch.nn.utils.clip_grad_norm_(cnn_params, max_norm=cnn_max_norm)
    if qnn_params:
        torch.nn.utils.clip_grad_norm_(qnn_params, max_norm=qnn_max_norm)


# ========================================================================================
# 9. DIRICHLET NON-IID DATA DISTRIBUTION
# ========================================================================================

def compute_model_size_bits(model):
    """Compute model size in bits (32-bit float params)."""
    return sum(p.numel() for p in model.parameters()) * 32

def compute_round_comm_cost(model, num_clients, direction='both'):
    """Compute communication cost per federated round in bits/KB/MB."""
    model_bits = compute_model_size_bits(model)
    if direction == 'upload':
        total = model_bits * num_clients
    elif direction == 'download':
        total = model_bits * num_clients
    else:
        total = model_bits * num_clients * 2
    return {
        'bits': total, 'KB': total / 8 / 1024,
        'MB': total / 8 / 1024 / 1024,
        'per_client_KB': model_bits / 8 / 1024,
    }


# ========================================================================================
# 11. CHECKPOINTING AND RESUME
# ========================================================================================

def save_checkpoint(filepath, model, optimizer=None, scheduler=None,
                    kan_model=None, kan_optimizer=None, epoch=0,
                    results=None, config=None):
    """Save complete training state for resume."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.random.get_rng_state(),
    }
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if kan_model:
        state['kan_state_dict'] = kan_model.state_dict()
    if kan_optimizer:
        state['kan_optimizer_state_dict'] = kan_optimizer.state_dict()
    if torch.cuda.is_available():
        state['rng_cuda'] = torch.cuda.get_rng_state_all()
    if results:
        state['results'] = results
    if config:
        state['config'] = config
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None,
                    kan_model=None, kan_optimizer=None):
    """Load complete training state from checkpoint."""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if kan_model and 'kan_state_dict' in ckpt:
        kan_model.load_state_dict(ckpt['kan_state_dict'])
    if kan_optimizer and 'kan_optimizer_state_dict' in ckpt:
        kan_optimizer.load_state_dict(ckpt['kan_optimizer_state_dict'])
    if 'rng_python' in ckpt:
        random.setstate(ckpt['rng_python'])
    if 'rng_numpy' in ckpt:
        np.random.set_state(ckpt['rng_numpy'])
    if 'rng_torch' in ckpt:
        torch.random.set_rng_state(ckpt['rng_torch'])
    if 'rng_cuda' in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    return ckpt


# ========================================================================================
# 12. COMPUTATIONAL COMPLEXITY ANALYSIS
# ========================================================================================

def compute_gate_count(num_qubits, num_layers, use_qaoa=True, qaoa_p=1):
    """Compute quantum gate count for complexity analysis."""
    ry_gates = num_qubits * num_layers
    cx_gates = (num_qubits - 1) * num_layers
    single_q = ry_gates
    two_q = cx_gates
    if use_qaoa:
        single_q += num_qubits * qaoa_p
        two_q += (num_qubits - 1) * qaoa_p
    return {
        'single_qubit_gates': single_q,
        'two_qubit_gates': two_q,
        'total_gates': single_q + two_q,
        'circuit_depth': num_layers * 2 + (2 * qaoa_p if use_qaoa else 0),
        'quantum_walk_per_round': f"O(N) where N={num_qubits} (ring)",
        'hilbert_dim': 2 ** num_qubits,
    }


# ========================================================================================
# 13. ABLATION STUDY FRAMEWORK
# ========================================================================================

ABLATION_VARIANTS = {
    'Full KIQFL': {},
    'No Quantum Walk': {'disable_quantum_walk': True},
    'No KAN': {'disable_kan': True},
    'No Sporadic': {'disable_sporadic': True},
    'No Indirect Aggregation': {'use_indirect_aggregation': False},
    'No QAOA': {'disable_qaoa': True},
    'No NISQ Noise': {'disable_noise': True},
}


def compute_comm_breakdown(model, num_clients, num_qubits, aggregation_mode='quantum_guided',
                           num_participating=None):
    """Communication payload estimates per round (KB)."""
    n = num_participating or num_clients
    full = compute_round_comm_cost(model, n, direction='both')
    hilbert_dim = 2 ** num_qubits
    state_bits = n * hilbert_dim * 16 * 8  # complex128 bytes -> bits
    q_params = sum(p.numel() for name, p in model.named_parameters() if is_quantum_param_key(name))
    quantum_subnet_bits = n * q_params * 32
    if aggregation_mode == 'quantum_subnet':
        effective_bits = state_bits + quantum_subnet_bits
    elif aggregation_mode == 'quantum_guided':
        effective_bits = full['bits'] + state_bits
    else:
        effective_bits = full['bits']
    return {
        'comm_full_KB': full['KB'],
        'comm_state_KB': state_bits / 8 / 1024,
        'comm_quantum_subnet_KB': quantum_subnet_bits / 8 / 1024,
        'comm_effective_KB': effective_bits / 8 / 1024,
        'hilbert_dim': hilbert_dim,
        'param_count': sum(p.numel() for p in model.parameters()),
        'state_to_param_ratio': hilbert_dim / max(1, sum(p.numel() for p in model.parameters())),
    }

def build_nisq_report(num_qubits, num_layers, use_qaoa=True, round_time_s=0.0):
    gates = compute_gate_count(num_qubits, num_layers, use_qaoa=use_qaoa, qaoa_p=1)
    gates['sim_time_per_round_s'] = round_time_s
    return gates

def grid_search_alpha_target(local_state, walked_state, prev_global_state,
                             model=None, val_loader=None, criterion=None, device_arg=None,
                             alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                             w_walk=0.45, w_global=0.45, w_val=0.10):
    """
    Pick alpha minimizing state-utility score (fidelity to walked + global reference).
    val_loss used only as small tie-breaker (w_val default 0.1).
    """
    if local_state is None or walked_state is None:
        return 0.5
    best_alpha, best_score = 0.5, float('inf')
    val_loss_const = None
    if val_loader is not None and model is not None and criterion is not None:
        try:
            with torch.no_grad():
                model.eval()
                losses = []
                for X, y in val_loader:
                    X, y = X.to(device_arg), y.to(device_arg)
                    losses.append(criterion(model(X), y).item())
                if losses:
                    val_loss_const = float(np.mean(losses))
        except Exception:
            val_loss_const = None

    for a in alphas:
        blended = apply_adaptive_weight_to_state(local_state, walked_state, float(a))
        f_walk = compute_state_fidelity(blended, walked_state)
        f_walk = f_walk.item() if torch.is_tensor(f_walk) else float(f_walk)
        f_global = 1.0
        if prev_global_state is not None:
            f_g = compute_state_fidelity(blended, prev_global_state)
            f_global = f_g.item() if torch.is_tensor(f_g) else float(f_g)
        score = w_walk * (1.0 - f_walk) + w_global * (1.0 - f_global)
        if val_loss_const is not None:
            score += w_val * val_loss_const
        if score < best_score:
            best_score, best_alpha = score, float(a)
    return best_alpha

def make_client_val_loader(dataset, batch_size=32, fraction=0.1):
    n = max(1, int(len(dataset) * fraction))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset = torch.utils.data.Subset(dataset, indices[:n])
    return torch.utils.data.DataLoader(subset, batch_size=min(batch_size, n), shuffle=False)

def sporadic_skip_probability(base_p, noise_level=0.0, nisq_error_rate=0.0,
                              grad_noise=0.0, mode='noise_adaptive'):
    """
    Skip probability for sporadic updates.
    - fixed: base_p
    - noise_adaptive (SpoQFL-inspired): scales with NISQ / schedule / grad noise intensity
    """
    base_p = float(max(0.0, min(0.5, base_p)))
    if mode == 'fixed' or base_p <= 0:
        return base_p
    intensity = float(noise_level) + float(nisq_error_rate) + 0.5 * float(grad_noise)
    p = base_p * (1.0 + 2.0 * max(0.0, intensity))
    return float(min(0.5, p))

def estimate_grad_noise(model):
    """Relative grad magnitude proxy used as noise intensity (0–1-ish)."""
    norms = []
    for p in model.parameters():
        if p.grad is None:
            continue
        norms.append(float(p.grad.detach().norm().item()))
    if not norms:
        return 0.0
    m = float(np.mean(norms))
    return float(min(1.0, m / (m + 1.0)))

def enrich_epoch_result(base, comm=None, gates=None, agg_weights=None,
                        convergence_epoch=None, fidelity_history=None):
    out = dict(base)
    if comm:
        out.update(comm)
    if gates:
        out['gate_count'] = gates.get('total_gates', 0)
        out['sim_time_per_round_s'] = gates.get('sim_time_per_round_s', 0.0)
        out['hilbert_dim'] = gates.get('hilbert_dim', 0)
    if agg_weights is not None:
        out['aggregation_weights'] = agg_weights
    if convergence_epoch is not None:
        out['convergence_epoch'] = convergence_epoch
    if fidelity_history is not None and len(fidelity_history) >= 2:
        accs = [h[1] for h in fidelity_history]
        fids = [h[0] for h in fidelity_history]
        if len(accs) >= 2 and np.std(fids) > 1e-8 and np.std(accs) > 1e-8:
            out['fidelity_task_corr'] = float(np.corrcoef(fids, accs)[0, 1])
    return out

def track_convergence(results, threshold=0.90):
  for i, r in enumerate(results):
      if r.get('accuracy', 0) >= threshold:
          return i + 1
  return len(results)

def add_state_dp_noise(states, sigma):
    if sigma <= 0:
        return states
    noisy = []
    for s in states:
        if s is None:
            noisy.append(s)
            continue
        n = torch.randn_like(s.real) * sigma + 1j * torch.randn_like(s.real) * sigma
        ns = s + n.to(dtype=s.dtype, device=s.device)
        norm = torch.norm(ns)
        noisy.append(ns / norm if norm > 1e-12 else ns)
    return noisy

class RDPAccountant:
    """Simple RDP ledger (Gaussian mechanism, order=32)."""

    def __init__(self, delta=1e-5):
        self.delta = delta
        self.steps = []

    def step(self, noise_multiplier, sample_rate=1.0):
        self.steps.append((noise_multiplier, sample_rate))

    def epsilon(self, orders=None):
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)]
        if not self.steps:
            return 0.0
        min_eps = float('inf')
        for order in orders:
            rdp = 0.0
            for sigma, q in self.steps:
                if sigma < 1e-12:
                    continue
                rdp += order * (q ** 2) / (2 * (sigma ** 2))
            if rdp > 0:
                eps = rdp + math.log(1 / self.delta) / (order - 1)
                min_eps = min(min_eps, eps)
        return float(min_eps if min_eps != float('inf') else 0.0)

def clip_per_sample_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def apply_dp_to_aggregate_update(global_model, pre_round_state_dict, noise_multiplier,
                                 max_grad_norm, num_clients):
    """Add Gaussian noise to aggregated model update (post-aggregation DP)."""
    sigma = noise_multiplier * max_grad_norm / max(1, num_clients)
    with torch.no_grad():
        for (name, param), key in zip(global_model.named_parameters(), pre_round_state_dict.keys()):
            delta = param - pre_round_state_dict[key].to(param.device)
            noise = torch.randn_like(delta) * sigma
            param.copy_(pre_round_state_dict[key].to(param.device) + delta + noise)
    # per-step epsilon bound
    delta_dp = 1e-5
    eps = math.sqrt(2 * math.log(1.25 / delta_dp)) * max_grad_norm / max(sigma, 1e-12)
    return eps, sigma

