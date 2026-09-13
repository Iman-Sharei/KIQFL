import torch

def state_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """
    Compute quantum state fidelity between two pure states.
    
    Following Qiskit's implementation:
    F(|ψ₁⟩, |ψ₂⟩) = |⟨ψ₁|ψ₂⟩|²
    
    For pure states, this is simply the squared magnitude of the inner product.
    
    Physical interpretation:
    - F = 1.0: Identical states (up to global phase)
    - F = 0.0: Orthogonal states
    - F = 1/dim: Expected value for random states in dim-dimensional Hilbert space
    
    Args:
        state1: First quantum state vector (torch.Tensor, complex)
        state2: Second quantum state vector (torch.Tensor, complex)
        
    Returns:
        float: Fidelity value in [0, 1]
    """
    s1 = state1.flatten()
    s2 = state2.flatten()
    
    if s1.shape[0] != s2.shape[0]:
        raise ValueError(f"State dimensions must match: {s1.shape[0]} vs {s2.shape[0]}")
    
    norm1 = torch.norm(s1)
    norm2 = torch.norm(s2)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    
    s1_normalized = s1 / norm1
    s2_normalized = s2 / norm2
    
    inner_product = torch.sum(torch.conj(s1_normalized) * s2_normalized)
    fidelity = torch.abs(inner_product) ** 2
    
    return float(fidelity.real.item())

def extract_state_vector(qc):
    return qc.state_vector.clone().detach()

def apply_adaptive_weight_to_state(state_a, state_b, alpha):
    """
    Apply KAN-computed adaptive weighting between two quantum states.
    
    new_state = (1 - α) * state_a + α * state_b
    
    SCIENTIFIC NOTE: This is the core of KIQFL's "indirect guidance" philosophy.
    Instead of forcing convergence, we gently suggest a direction by mixing
    the current state with a walked state. The mixing coefficient α is
    adaptively computed by the KAN based on noise level, fidelity, and time.
    
    Physical interpretation:
    - α ≈ 0: Trust current state (high noise or high fidelity)
    - α ≈ 1: Follow walked state (low noise, need improvement)
    - α ≈ 0.5: Balanced exploration
    
    Args:
        state_a: Current state vector
        state_b: Target/walked state vector
        alpha: Mixing coefficient in [0, 1]
    
    Returns:
        New mixed and normalized state vector
    """
    # Ensure alpha is a proper scalar
    if torch.is_tensor(alpha):
        alpha = alpha.item()
    alpha = float(alpha)
    
    # Clamp alpha to valid range
    alpha = max(0.0, min(1.0, alpha))
    
    # Ensure both states are complex and on the same device
    device = state_a.device
    s_a = state_a.to(dtype=torch.cfloat)
    s_b = state_b.to(dtype=torch.cfloat, device=device)
    
    # Linear interpolation in Hilbert space
    new_state = (1 - alpha) * s_a + alpha * s_b
    
    # IMPORTANT: Normalize to maintain valid quantum state
    norm = torch.norm(new_state)
    if norm > 1e-10:
        new_state = new_state / norm
    else:
        # Fallback to state_a if mixed state collapses
        new_state = s_a.clone()
        norm = torch.norm(new_state)
        if norm > 1e-10:
            new_state = new_state / norm
    
    return new_state

def compute_state_fidelity(state_a, state_b):
    """
    Compute quantum state fidelity: F = |⟨ψ_a|ψ_b⟩|²
    
    This follows the Qiskit state_fidelity convention for pure states.
    
    SCIENTIFIC NOTE:
    - F = 1.0: States are identical (up to global phase)
    - F = 0.0: States are orthogonal
    - F = 1/dim: Expected for random states in dim-dimensional space
    """
    # Flatten to 1D for inner product computation
    s_a = state_a.flatten()
    s_b = state_b.flatten()
    
    # Handle dimension mismatch
    if s_a.shape[0] != s_b.shape[0]:
        min_dim = min(s_a.shape[0], s_b.shape[0])
        s_a = s_a[:min_dim]
        s_b = s_b[:min_dim]
    
    # Normalize (essential for proper fidelity)
    norm_a = torch.norm(s_a)
    norm_b = torch.norm(s_b)
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return torch.tensor(0.0, device=s_a.device)
    
    s_a = s_a / norm_a
    s_b = s_b / norm_b
    
    # Inner product: ⟨ψ_a|ψ_b⟩ = Σ conj(a_i) * b_i
    overlap = torch.sum(torch.conj(s_a) * s_b)
    fidelity = torch.abs(overlap) ** 2
    
    # Clamp to [0, 1] for numerical stability
    return torch.clamp(fidelity.real, 0.0, 1.0)


