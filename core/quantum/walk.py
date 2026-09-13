import math
import random
from typing import List, Optional

import torch

def quantum_walk_transfer_state_list(node_states: List[torch.Tensor], theta_walk: float, phi_walk: float,
                                     walk_steps: int = 3, use_teleportation: bool = False,
                                     rng: Optional[random.Random] = None):
    """
    Apply a Discrete-Time Quantum Walk (DTQW) on a ring topology to mix
    quantum states across federated learning clients.

    SCIENTIFIC DESCRIPTION:
    This implements a proper DTQW with:
    - Coin space: C² (left/right directions on ring)
    - Position space: C^N (N = number of nodes on ring)
    - Coin operator: C(θ,φ) = [[cos(θ/2), e^{iφ}sin(θ/2)],
                                [e^{-iφ}sin(θ/2), -cos(θ/2)]]
    - Shift operator: S = Σ_x |x-1⟩⟨x| ⊗ |0⟩⟨0| + |x+1⟩⟨x| ⊗ |1⟩⟨1|  (mod N)
    - Walk unitary per step: W = S · (I_N ⊗ C)
    - After `walk_steps` applications of W, the walker's position amplitudes
      determine the mixing weights for combining node states.

    The key advantage over simple linear combination:
    - Multi-hop information propagation (not just nearest neighbors)
    - Walk *complex amplitudes* (with relative phases) weight the mixture of
      client circuit states used as *guidance* for central-server aggregation
    - Ballistic spreading on the ring: O(t) vs classical O(√t) for the walker
    - Coin parameters (θ,φ) control bias and interference of the walker

    This does **not** replace the FL server: parameters are still aggregated
    centrally; the walk only reshapes guidance states / fidelity weights.

    Args:
        node_states: List of quantum state vectors (one per node/client)
        theta_walk: Coin parameter θ (0=localized, π/4=balanced, π/2=full spread)
        phi_walk: Coin phase parameter φ (controls interference)
        walk_steps: Number of DTQW steps (more steps = wider mixing)
        use_teleportation: If True, apply teleportation-based transfer
        rng: Random number generator for teleportation stochasticity

    Returns:
        List of new mixed state vectors
    """
    if len(node_states) == 0:
        return []
    if rng is None:
        rng = random

    N = len(node_states)
    device = node_states[0].device

    if N == 1:
        return [node_states[0].clone()]

    # ---- Optional teleportation mode -----------------------------------------
    if use_teleportation and N >= 2:
        return _teleportation_state_transfer(node_states, theta_walk, phi_walk, rng)

    # ---- Build walk operator on the ring (coin_dim × N) ----------------------
    # Coin operator C(θ,φ) ∈ SU(2)
    theta = torch.tensor(theta_walk, dtype=torch.float32, device=device)
    phi = torch.tensor(phi_walk, dtype=torch.float32, device=device)
    cos_t = torch.cos(theta / 2)
    sin_t = torch.sin(theta / 2)
    phase_p = torch.exp(1j * phi)
    phase_m = torch.exp(-1j * phi)

    coin = torch.zeros(2, 2, device=device, dtype=torch.cfloat)
    coin[0, 0] = cos_t
    coin[0, 1] = phase_p * sin_t
    coin[1, 0] = phase_m * sin_t
    coin[1, 1] = -cos_t

    # Shift operator S on ring: S|x,0⟩ = |x-1 mod N, 0⟩, S|x,1⟩ = |x+1 mod N, 1⟩
    # Full Hilbert space dimension: 2N (coin ⊗ position)
    dim_walk = 2 * N
    I_N = torch.eye(N, device=device, dtype=torch.cfloat)
    I_C = torch.eye(2, device=device, dtype=torch.cfloat)

    # Shift matrix
    S = torch.zeros(dim_walk, dim_walk, device=device, dtype=torch.cfloat)
    for x in range(N):
        left = (x - 1) % N
        right = (x + 1) % N
        # |left, 0⟩⟨x, 0|
        S[left * 2 + 0, x * 2 + 0] = 1.0
        # |right, 1⟩⟨x, 1|
        S[right * 2 + 1, x * 2 + 1] = 1.0

    # Coin applied to all positions: I_N ⊗ C
    # In our interleaved basis |x,c⟩ where c is the coin index:
    IC = torch.zeros(dim_walk, dim_walk, device=device, dtype=torch.cfloat)
    for x in range(N):
        for c1 in range(2):
            for c2 in range(2):
                IC[x * 2 + c1, x * 2 + c2] = coin[c1, c2]

    # Walk operator per step: W = S · IC
    W = torch.matmul(S, IC)

    # Multi-step walk: W^steps
    W_total = torch.eye(dim_walk, device=device, dtype=torch.cfloat)
    for _ in range(walk_steps):
        W_total = torch.matmul(W, W_total)

    # ---- Compute mixing amplitudes for each starting position ----------------
    # For each starting node i, initialize walker at |i, 0⟩ (coin=left)
    # After walk, marginalise over coin to get position amplitudes
    new_states = []
    for i in range(N):
        # Initial walker state: |i, 0⟩ (start at position i, coin=|0⟩)
        psi_walk = torch.zeros(dim_walk, device=device, dtype=torch.cfloat)
        psi_walk[i * 2 + 0] = 1.0  # position i, coin 0

        # Apply walk
        psi_walk = torch.matmul(W_total, psi_walk)

        # Complex position amplitudes (sum over coin) — keeps walk relative phases.
        # Marginal probs p(j)=|a_j0|²+|a_j1|²; coherent weights use a_j0+a_j1.
        complex_amps = torch.zeros(N, device=device, dtype=torch.cfloat)
        for j in range(N):
            complex_amps[j] = psi_walk[j * 2 + 0] + psi_walk[j * 2 + 1]

        amp_norm = torch.sqrt(torch.sum(torch.abs(complex_amps) ** 2))
        if amp_norm > 1e-15:
            complex_amps = complex_amps / amp_norm

        # Mixed guidance state: Σ_j α_j |ψ_j⟩ with complex walk amplitudes α_j
        new_state = torch.zeros_like(node_states[i], dtype=torch.cfloat)
        for j in range(N):
            new_state = new_state + complex_amps[j] * node_states[j].to(
                dtype=torch.cfloat, device=device)

        norm = torch.norm(new_state)
        if norm > 1e-10:
            new_state = new_state / norm
        new_states.append(new_state)

    return new_states


def _teleportation_state_transfer(node_states: List[torch.Tensor],
                                  theta_walk: float, phi_walk: float,
                                  rng: random.Random) -> List[torch.Tensor]:
    """
    Optional noisy ring channel inspired by teleportation *correction* Paulis.

    HONEST SCOPE (not a bit-exact multi-qubit teleportation protocol):
    Full Hilbert-space client states are already classical tensors in this
    simulator. This routine mixes each node with its ring neighbor after a
    random Pauli {I,X,Z,ZX} on qubit-0 — a stochastic channel analogous to
    uncorrected Bell outcomes — then blends with coin angle (θ, φ).

    Default KIQFL uses DTQW (`use_teleportation=False`). Enable only for
    ablation / noise-robustness experiments; do not claim hardware teleportation.
    """
    N = len(node_states)
    device = node_states[0].device
    theta = torch.tensor(theta_walk, dtype=torch.float32, device=device)
    phi = torch.tensor(phi_walk, dtype=torch.float32, device=device)

    cos_c = torch.cos(theta / 2).to(torch.cfloat)
    sin_c = torch.sin(theta / 2).to(torch.cfloat)
    phase = torch.exp(1j * phi).to(torch.cfloat)

    new_states = []
    for i in range(N):
        neighbor_idx = (i + 1) % N
        local_state = node_states[i].to(dtype=torch.cfloat)
        neighbor_state = node_states[neighbor_idx].to(dtype=torch.cfloat)

        # Simulate Bell measurement outcome (2 classical bits)
        bell_outcome = rng.randint(0, 3)

        # Apply Pauli correction based on Bell measurement
        # This simulates the classical communication + correction step
        corrected_state = neighbor_state.clone()
        if bell_outcome == 1:
            # X correction: flip amplitudes pairwise
            corrected_state = _apply_pauli_x_to_state(corrected_state)
        elif bell_outcome == 2:
            # Z correction: negate odd-index amplitudes
            corrected_state = _apply_pauli_z_to_state(corrected_state)
        elif bell_outcome == 3:
            # ZX correction
            corrected_state = _apply_pauli_x_to_state(corrected_state)
            corrected_state = _apply_pauli_z_to_state(corrected_state)
        # bell_outcome == 0: no correction

        # Mix teleported state with local state
        new_state = cos_c * local_state + (phase * sin_c) * corrected_state
        norm = torch.norm(new_state)
        if norm > 1e-10:
            new_state = new_state / norm
        new_states.append(new_state)

    return new_states


def _apply_pauli_x_to_state(state: torch.Tensor) -> torch.Tensor:
    """Apply Pauli-X on qubit 0 (MSB / kron-first) of a multi-qubit state vector."""
    s = state.flatten().clone()
    dim = s.shape[0]
    msb = dim >> 1
    result = torch.zeros_like(s)
    for i in range(dim):
        result[i ^ msb] = s[i]
    return result.reshape(state.shape)


def _apply_pauli_z_to_state(state: torch.Tensor) -> torch.Tensor:
    """Apply Pauli-Z on qubit 0 (MSB / kron-first) of a multi-qubit state vector."""
    s = state.clone()
    flat = s.flatten()
    msb = flat.shape[0] >> 1
    for i in range(flat.shape[0]):
        if i & msb:
            flat[i] = -flat[i]
    return s


# ========================================================================================
