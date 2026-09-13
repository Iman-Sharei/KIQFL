import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
from kan import KANLayer

from core.quantum.fidelity import apply_adaptive_weight_to_state
from core.quantum.walk import quantum_walk_transfer_state_list

class KANEdge(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) for adaptive edge weighting in KIQFL.
    
    SCIENTIFIC BACKGROUND:
    The Kolmogorov-Arnold representation theorem states that any multivariate
    continuous function can be represented as:
    f(x₁, ..., xₙ) = Σᵢ Φᵢ(Σⱼ φᵢⱼ(xⱼ))
    
    This implementation uses REAL B-spline-based KAN layers from the pykan library,
    where each edge carries a learnable univariate activation function composed of:
    φ(x) = scale_base · b(x) + scale_sp · spline(x)
    
    - b(x) = SiLU (residual/base function)
    - spline(x) = Σ cᵢ Bᵢ(x)  (B-spline with learnable coefficients)
    - Grid is adaptive and can be updated from data samples
    
    In KIQFL, KAN learns the optimal mixing coefficient α as:
    α = σ(KAN₂(KAN₁([noise, fidelity, t])))
    
    Where:
    - noise: NISQ error rate affecting current quantum state
    - fidelity: Similarity between consecutive states (convergence indicator)
    - t: Normalized training progress (for adaptive exploration/exploitation)
    
    The output α ∈ [0, 1] determines how to blend:
    new_state = (1 - α) * local_state + α * walked_state
    
    Architecture (True KAN with B-spline activations on edges):
    - KANLayer1: 3 → hidden_dim (each of 3×hidden_dim edges has its own B-spline)
    - KANLayer2: hidden_dim → 1 (each of hidden_dim×1 edges has its own B-spline)
    - Sigmoid output → α ∈ [0, 1]
    
    Key differences from MLP:
    - MLP: fixed activations on nodes (ReLU, Tanh, etc.)
    - KAN: learnable B-spline activations on edges (φᵢⱼ per connection)
    - KAN has adaptive grid that refines based on input distribution
    """
    def __init__(self, input_dim=3, hidden_dim=16, grid_size=5, spline_order=3,
                 grid_range=(-1, 1), grid_eps=0.02, noise_scale=0.3):
        super(KANEdge, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Layer 1: input_dim → hidden_dim with B-spline activations on every edge
        # Each of (input_dim × hidden_dim) edges has its own learnable B-spline function
        self.kan_layer1 = KANLayer(
            in_dim=input_dim, 
            out_dim=hidden_dim, 
            num=grid_size,           # Number of grid intervals (G)
            k=spline_order,          # B-spline polynomial order
            noise_scale=noise_scale, # Initial spline noise for exploration
            grid_eps=grid_eps,       # Grid adaptivity (0=percentile, 1=uniform)
            grid_range=list(grid_range),
            scale_base_mu=0.0,       # Residual function magnitude mean
            scale_base_sigma=1.0,    # Residual function magnitude std
            scale_sp=1.0,            # Spline function magnitude
            base_fun=torch.nn.SiLU(),# Residual function b(x)
            sp_trainable=True,       # scale_sp is learnable
            sb_trainable=True,       # scale_base is learnable
        )
        
        # Layer 2: hidden_dim → 1 with B-spline activations on every edge
        self.kan_layer2 = KANLayer(
            in_dim=hidden_dim, 
            out_dim=1, 
            num=grid_size,
            k=spline_order,
            noise_scale=noise_scale,
            grid_eps=grid_eps,
            grid_range=list(grid_range),
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,
            base_fun=torch.nn.SiLU(),
            sp_trainable=True,
            sb_trainable=True,
        )

    def forward(self, features: torch.Tensor):
        """
        Forward pass computing adaptive mixing coefficient using true KAN.
        
        Each KANLayer computes:
        y = Σⱼ [scale_base_ij · SiLU(xⱼ) + scale_sp_ij · spline_ij(xⱼ)] · mask_ij
        
        where spline_ij(x) = Σₖ coef_ijk · B_k(x) uses real B-spline basis functions.
        
        Args:
            features: Tensor of shape (batch_size, 3) with [noise, fidelity, t]
        
        Returns:
            Tensor of shape (batch_size, 1) with α ∈ [0, 1]
        """
        # KAN Layer 1: B-spline edge activations (3 → hidden_dim)
        # Returns: y, preacts, postacts, postspline
        x, _, _, _ = self.kan_layer1(features)
        
        # KAN Layer 2: B-spline edge activations (hidden_dim → 1)
        x, _, _, _ = self.kan_layer2(x)
        
        # Sigmoid to constrain output to [0, 1]
        x = torch.sigmoid(x)
        return x
    
    def update_grids(self, x: torch.Tensor):
        """
        Update B-spline grids based on input data distribution.
        
        This is a key KAN feature: the grid adapts to where the data actually lives,
        allowing more precise function approximation in high-density regions.
        
        Args:
            x: Sample inputs of shape (batch_size, input_dim)
        """
        with torch.no_grad():
            self.kan_layer1.update_grid_from_samples(x)
            # Get intermediate activations to update layer 2's grid
            h, _, _, _ = self.kan_layer1(x)
            self.kan_layer2.update_grid_from_samples(h)


def compute_adaptive_alpha(noise, fidelity, t, kan_model, device='cpu'):
    """
    Compute adaptive mixing coefficient using KAN model.
    
    SCIENTIFIC NOTE:
    The KAN learns the optimal exploration-exploitation tradeoff:
    α = f(noise, fidelity, t)
    
    Where:
    - noise: Current NISQ noise level
    - fidelity: State similarity (convergence indicator)
    - t: Normalized training progress
    
    The output α ∈ [0, 1] determines how much to trust the quantum
    walk propagated state vs. the local state.
    
    Args:
        noise: Current noise level
        fidelity: Fidelity between consecutive states
        t: Normalized training time
        kan_model: Trained KAN model
        device: Compute device
    
    Returns:
        float: Adaptive mixing coefficient α ∈ [0, 1]
    """
    # Ensure inputs are in valid ranges
    noise = max(0.0, min(1.0, float(noise)))
    fidelity = max(0.0, min(1.0, float(fidelity)))
    t = max(0.0, min(1.0, float(t)))
    
    features = torch.tensor([[noise, fidelity, t]], dtype=torch.float32, device=device)
    with torch.no_grad():
        alpha = kan_model(features).item()
    
    # Clamp output to valid range
    return float(max(0.0, min(1.0, alpha)))


# ========================================================================================
# --- Full KIQFL Propagation Step ---
# ========================================================================================

def kiqfl_propagation_step(node_states: List[torch.Tensor],
                           noise_levels: List[float],
                           fidelities: List[float],
                           t: int,
                           kan_model: KANEdge,
                           sporadic_p: float = 0.1,
                           theta_walk: float = math.pi / 6,
                           phi_walk: float = 0.0,
                           walk_steps: int = 3,
                           use_teleportation: bool = False,
                           rng: Optional[random.Random] = None):
    """
    Complete KIQFL propagation step combining all components.
    
    SCIENTIFIC DESCRIPTION:
    This function implements the core of the KIQFL algorithm:
    1. Apply discrete-time quantum walk mixing on ring topology
       (multi-step walk with coin operator, or teleportation)
    2. Use KAN to compute adaptive mixing coefficients
    3. Apply sporadic updates for noise robustness
    4. Combine local and walked states with adaptive weighting
    
    The "indirect guidance" philosophy:
    - Instead of forcing convergence (like FedAvg), we suggest a direction
    - States are gently nudged toward agreement without overwriting
    - Sporadic skipping provides robustness against noisy updates
    
    Args:
        node_states: List of quantum state vectors (one per node)
        noise_levels: Per-node noise levels
        fidelities: Per-node state fidelities (convergence indicators)
        t: Current training epoch
        kan_model: Trained KAN for adaptive weighting
        sporadic_p: Probability of skipping update (noise robustness)
        theta_walk: Quantum walk coin angle
        phi_walk: Quantum walk coin phase
        walk_steps: Number of DTQW steps (more = wider mixing)
        use_teleportation: Use teleportation-based transfer instead of walk
        rng: Random number generator for reproducibility
    
    Returns:
        List of new state vectors after KIQFL propagation
    """
    if rng is None:
        rng = random
    
    # Step 1: Apply quantum walk to mix adjacent states
    walked = quantum_walk_transfer_state_list(
        node_states, theta_walk, phi_walk,
        walk_steps=walk_steps,
        use_teleportation=use_teleportation,
        rng=rng
    )
    
    new_states = []
    for i, (state, walked_state) in enumerate(zip(node_states, walked)):
        # Step 2: Sporadic update - skip with probability p (noise robustness)
        if rng.random() < sporadic_p:
            new_states.append(state.clone())
            continue
        
        # Step 3: Compute adaptive mixing coefficient using KAN
        # t is training progress in [0, 1] (same feature scale as KANTrainer)
        t_norm = float(t)
        if t_norm > 1.0:
            t_norm = 1.0
        alpha = compute_adaptive_alpha(noise_levels[i], fidelities[i], t_norm, kan_model)
        
        # Step 4: Apply adaptive weighting
        new_state = apply_adaptive_weight_to_state(state, walked_state, alpha)
        new_states.append(new_state)
    
    return new_states


# ========================================================================================
