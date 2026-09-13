import torch
import math
import random
import torch.nn as nn
from typing import List, Optional
# ========================================================================================
# Device setup
# ========================================================================================

def get_device(gpu_no):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')


# ========================================================================================
# Base Quantum Circuit Class
# ========================================================================================

class quantum_circuit:
    
    def __init__(self, num_qubits: int, state_vector=None, device='cuda', gpu_no=0):
        
        if device != 'cuda': 
            self.device = torch.device(device)
        else: 
            self.device = get_device(gpu_no)
            
        # Number of qubits and state dimension
        self.n = num_qubits   
        self.dim = 2 ** self.n  
       
        # Initialize state vector
        if state_vector is None:
            state_vector = torch.zeros(self.dim, device=self.device, dtype=torch.cfloat) 
            state_vector[0] = 1
            self.state_vector = state_vector.reshape(-1, 1)
        else:
            if state_vector.device != self.device:
                self.device = state_vector.device
            if state_vector.shape[0] == self.dim: 
                self.state_vector = state_vector.to(device=self.device, dtype=torch.cfloat)
            else:
                raise ValueError(f'The dimension 2**n ({self.dim}) does NOT match the shape of the state vector ({state_vector.shape[0]}).')
                
        # Define common quantum gates
        self.I        = torch.tensor([[1, 0], [0, 1]], device=self.device, dtype=torch.cfloat)
        self.x_matrix = torch.tensor([[0., 1], [1, 0]], device=self.device, dtype=torch.cfloat)
        self.y_matrix = torch.tensor([[0, -1j], [1j, 0]], device=self.device, dtype=torch.cfloat)
        self.z_matrix = torch.tensor([[1, 0], [0, -1]], device=self.device, dtype=torch.cfloat)
        self.h_matrix = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], device=self.device, dtype=torch.cfloat)
        self.proj_0   = torch.tensor([[1, 0], [0, 0]], device=self.device, dtype=torch.cfloat)
        self.proj_1   = torch.tensor([[0, 0], [0, 1]], device=self.device, dtype=torch.cfloat)
        

    # ====================================================================================
    # Single and Controlled Gates
    # ====================================================================================

    def single_qubit_gate(self, target: int, gate: torch.Tensor):
        if target < 0 or self.n <= target: 
            print('0 <= target <= num_qubits - 1 is NOT satisfied!')
            return
        
        single_q_gate = torch.tensor(1, device=self.device, dtype=torch.cfloat)
        for k in range(self.n):
            if k == target:
                single_q_gate = torch.kron(single_q_gate, gate)
            else:
                single_q_gate = torch.kron(single_q_gate, self.I)
        self.state_vector = torch.matmul(single_q_gate, self.state_vector)
        return self.state_vector

    
    def controlled_gate(self, control: int, target: int, gate: torch.Tensor):
        if control < 0 or self.n <= control: 
            print('0 <= control <= num_qubits - 1 is NOT satisfied!')   
            return
        elif target < 0 or self.n <= target: 
            print('0 <= target <= num_qubits - 1 is NOT satisfied!') 
            return
        elif control == target:
            print('control and target qubits must be different!')
            return

        control_gate_part_0 = torch.tensor(1, device=self.device, dtype=torch.cfloat)
        control_gate_part_1 = torch.tensor(1, device=self.device, dtype=torch.cfloat)
        
        for k in range(self.n):
            if k == control:
                control_gate_part_0 = torch.kron(control_gate_part_0, self.proj_0)
                control_gate_part_1 = torch.kron(control_gate_part_1, self.proj_1)
            elif k == target:
                control_gate_part_0 = torch.kron(control_gate_part_0, self.I)
                control_gate_part_1 = torch.kron(control_gate_part_1, gate)
            else:
                control_gate_part_0 = torch.kron(control_gate_part_0, self.I)
                control_gate_part_1 = torch.kron(control_gate_part_1, self.I)
        
        control_gate = control_gate_part_0 + control_gate_part_1
        self.state_vector = torch.matmul(control_gate, self.state_vector)
        return self.state_vector


    # ====================================================================================
    # Basic Gates
    # ====================================================================================

    def x(self, target: int): return self.single_qubit_gate(target, self.x_matrix)
    def y(self, target: int): return self.single_qubit_gate(target, self.y_matrix)
    def z(self, target: int): return self.single_qubit_gate(target, self.z_matrix)
    def h(self, target: int): return self.single_qubit_gate(target, self.h_matrix)


    # ====================================================================================
    # Rotation Gates
    # ====================================================================================

    def Rx(self, target: int, theta):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        Rx_mat = torch.stack([torch.stack([co, -1j * si]), torch.stack([-1j * si, co])])
        self.single_qubit_gate(target, Rx_mat)
            
    def Ry(self, target: int, theta):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        Ry_mat = torch.stack([torch.stack([co, -si]), torch.stack([si, co])])
        self.single_qubit_gate(target, Ry_mat)
        
    def Rz(self, target: int, theta):
        """
        Rz(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
        
        Standard quantum rotation convention with θ/2 factor.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        exp_neg = torch.exp(-1j * theta / 2)
        exp_pos = torch.exp(1j * theta / 2)
        zero = torch.tensor(0, device=self.device, dtype=torch.cfloat)
        Rz_mat = torch.stack([torch.stack([exp_neg, zero]), torch.stack([zero, exp_pos])])
        self.single_qubit_gate(target, Rz_mat)

    def R(self, target: int, theta, phi, lamda):
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float32, device=self.device)
        if not isinstance(lamda, torch.Tensor):
            lamda = torch.tensor(lamda, dtype=torch.float32, device=self.device)
        a = torch.cos(theta / 2)
        b = -torch.exp(1j * lamda) * torch.sin(theta / 2)
        c = torch.exp(1j * phi) * torch.sin(theta / 2)
        d = torch.exp(1j * (phi + lamda)) * torch.cos(theta / 2)
        R_mat = torch.stack([torch.stack([a, b]), torch.stack([c, d])])
        self.single_qubit_gate(target, R_mat)
        

    # ====================================================================================
    # Layers of rotations
    # ====================================================================================

    def Ry_layer(self, angs: torch.Tensor):
        """
        Apply tensor product of Ry(θ) rotations to all qubits.
        Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        
        SCIENTIFIC NOTE: The factor of 1/2 is ESSENTIAL for proper quantum rotations.
        Without it, a full rotation would be π instead of 2π.
        """
        # First qubit rotation - note the θ/2 factor
        cos, sin = torch.cos(angs[0] / 2), torch.sin(angs[0] / 2)
        rot = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])
        
        # Tensor product for remaining qubits
        for i in range(1, len(angs)):
            cos, sin = torch.cos(angs[i] / 2), torch.sin(angs[i] / 2)
            rot = torch.kron(rot, torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])]))
        
        self.state_vector = torch.matmul(rot, self.state_vector)
        return self.state_vector

    def Rz_layer(self, angs: torch.Tensor):
        """
        Apply tensor product of Rz(θ) rotations to all qubits.
        Rz(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
        
        SCIENTIFIC NOTE: The proper Rz rotation uses exp(±iθ/2) on the diagonal.
        This ensures correct phase evolution and maintains unitarity.
        """
        # First qubit rotation with proper phase factors
        exp_neg = torch.exp(-1j * angs[0] / 2)
        exp_pos = torch.exp(1j * angs[0] / 2)
        zero = torch.tensor(0, device=self.device, dtype=torch.cfloat)
        rot = torch.stack([torch.stack([exp_neg, zero]), torch.stack([zero, exp_pos])])
        
        # Tensor product for remaining qubits
        for i in range(1, len(angs)):
            exp_neg = torch.exp(-1j * angs[i] / 2)
            exp_pos = torch.exp(1j * angs[i] / 2)
            rot = torch.kron(rot, torch.stack([torch.stack([exp_neg, zero]), torch.stack([zero, exp_pos])]))
        
        self.state_vector = torch.matmul(rot, self.state_vector)
        return self.state_vector


    # ====================================================================================
    # Controlled Layers
    # ====================================================================================

    def cx(self, control: int, target: int): self.controlled_gate(control, target, self.x_matrix)
    def cz(self, control: int, target: int): self.controlled_gate(control, target, self.z_matrix)

    def cx_linear_layer(self):
        self.controlled_gate(self.n - 2, self.n - 1, self.x_matrix)
        for i in range(self.n - 3, -1, -1):
            self.controlled_gate(i, i + 1, self.x_matrix)

    def cz_linear_layer(self):
        self.controlled_gate(self.n - 2, self.n - 1, self.z_matrix)
        for i in range(self.n - 3, -1, -1):
            self.controlled_gate(i, i + 1, self.z_matrix)


    # ====================================================================================
    # Measurement
    # ====================================================================================

    def probabilities(self):
        return self.state_vector.conj() * self.state_vector
    
    def get_statevector(self):
        """
        Return a copy of the current state vector (Qiskit-compatible naming).
        This is the TRUE quantum state of the circuit.
        """
        return self.state_vector.clone()
    
    def get_state_vector_normalized(self):
        """
        Return normalized state vector as 1D tensor.
        
        SCIENTIFIC NOTE: Quantum states must be normalized (⟨ψ|ψ⟩ = 1).
        This method ensures the state vector is properly normalized.
        """
        sv = self.state_vector.flatten()
        norm = torch.norm(sv)
        if norm > 1e-10:
            sv = sv / norm
        else:
            # If norm is essentially zero, return |0⟩ state
            sv = torch.zeros_like(sv)
            sv[0] = 1.0
        return sv

    # ====================================================================================
    # NISQ Noise Simulation
    # ====================================================================================

    def _build_single_qubit_operator(self, target: int, gate: torch.Tensor):
        """
        Build a full 2^n × 2^n operator from a single-qubit gate on qubit `target`.
        Used for constructing Kraus operators in the full Hilbert space.
        Unlike single_qubit_gate(), this does NOT apply the operator to the state.
        """
        op = torch.tensor(1, device=self.device, dtype=torch.cfloat)
        for k in range(self.n):
            if k == target:
                op = torch.kron(op, gate)
            else:
                op = torch.kron(op, self.I)
        return op

    def apply_noise(self, noise_model='depolarizing', error_rate=0.01, rng=None):
        """
        Apply NISQ noise using physically correct quantum channel formalism.

        SCIENTIFIC BACKGROUND:
        Real NISQ devices experience various types of decoherence.
        Each noise model is applied independently per qubit using the correct
        operator formalism.

        1. Depolarizing channel (per qubit, Monte Carlo trajectory):
           ε(ρ) = (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
           State-vector implementation: for each qubit, with probability p
           apply a uniformly random Pauli gate {X, Y, Z}.

        2. Bit-phase flip (per qubit):
           Independent bit-flip (X) and phase-flip (Z) channels.
           Each applied with probability p/2 per qubit.

        3. Amplitude damping (per qubit, Kraus operators + quantum trajectory):
           K₀ = [[1, 0], [0, √(1-γ)]]    (no-decay branch)
           K₁ = [[0, √γ], [0, 0]]         (decay branch)
           Satisfies completeness: K₀†K₀ + K₁†K₁ = I
           State-vector trajectory: stochastically select K₀ or K₁
           weighted by their respective outcome probabilities.

        Typical NISQ error rates: 0.001 - 0.05 per gate.

        Args:
            noise_model: 'depolarizing', 'bit_phase_flip', or 'amplitude_damping'
            error_rate: error probability per qubit per operation (0 to 1)
            rng: Optional random.Random instance for reproducibility.
                 Falls back to global random module if None.

        Returns:
            None (modifies state_vector in place)
        """
        if rng is None:
            rng = random

        if noise_model == 'depolarizing':
            # Per-qubit depolarizing channel (Monte Carlo trajectory)
            # For each qubit: with prob (1-p) identity, with prob p a random Pauli
            # This correctly implements ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
            for q in range(self.n):
                if rng.random() < error_rate:
                    pauli_choice = rng.randint(0, 2)  # 0=X, 1=Y, 2=Z
                    if pauli_choice == 0:
                        self.x(q)
                    elif pauli_choice == 1:
                        self.y(q)
                    else:
                        self.z(q)

        elif noise_model == 'bit_phase_flip':
            # Per-qubit independent bit-flip and phase-flip
            for q in range(self.n):
                if rng.random() < error_rate * 0.5:
                    self.x(q)  # Bit-flip
                if rng.random() < error_rate * 0.5:
                    self.z(q)  # Phase-flip

        elif noise_model == 'amplitude_damping':
            # Per-qubit amplitude damping via Kraus operators (quantum trajectory)
            # K₀ = [[1, 0], [0, √(1-γ)]]  — no-decay operator
            # K₁ = [[0, √γ], [0, 0]]       — decay operator (|1⟩ → |0⟩)
            gamma = error_rate
            sqrt_1mg = math.sqrt(max(0.0, 1.0 - gamma))
            sqrt_g = math.sqrt(max(0.0, gamma))

            K0 = torch.tensor([[1.0, 0.0], [0.0, sqrt_1mg]],
                              device=self.device, dtype=torch.cfloat)
            K1 = torch.tensor([[0.0, sqrt_g], [0.0, 0.0]],
                              device=self.device, dtype=torch.cfloat)

            for q in range(self.n):
                # Build full-space Kraus operators for qubit q
                K0_full = self._build_single_qubit_operator(q, K0)
                K1_full = self._build_single_qubit_operator(q, K1)

                # Compute unnormalized post-measurement states
                psi_0 = torch.matmul(K0_full, self.state_vector)
                psi_1 = torch.matmul(K1_full, self.state_vector)

                # Outcome probabilities: p_k = ⟨ψ|K_k†K_k|ψ⟩ = ||K_k|ψ⟩||²
                p0 = torch.real(torch.sum(psi_0.conj() * psi_0)).item()
                p1 = torch.real(torch.sum(psi_1.conj() * psi_1)).item()
                p_total = p0 + p1

                if p_total < 1e-15:
                    continue

                # Stochastic trajectory selection + renormalization
                if rng.random() < p0 / p_total:
                    self.state_vector = psi_0 / math.sqrt(p0) if p0 > 1e-15 else psi_0
                else:
                    self.state_vector = psi_1 / math.sqrt(p1) if p1 > 1e-15 else psi_1

    # ====================================================================================
    # QAOA Methods
    # ====================================================================================

    def apply_cost_layer(self, gamma, edge_list: List[tuple]):
        """
        Apply cost unitary U_C(gamma) = exp(-i gamma H_C)
        For MaxCut-like H_C = 1/2 sum_{(i,j,w)} w (I - Z_i Z_j)

        ``gamma`` may be a Python float or a torch scalar/0-dim tensor so that
        autograd can flow through variational QAOA parameters.
        """
        for (i, j, w) in edge_list:
            self.cx(i, j)
            self.Rz(j, 2.0 * gamma * w)
            self.cx(i, j)
        return

    def apply_mixer_layer(self, beta):
        """
        Apply mixer U_B(beta) = prod_j Rx_j(2*beta).

        ``beta`` may be a float or a torch tensor (keeps QAOA grads intact).
        """
        for q in range(self.n):
            self.Rx(q, 2.0 * beta)
        return

    def run_qaoa(self, gammas: List[float], betas: List[float], edge_list: List[tuple]):
        """
        Run full QAOA with given lists gammas and betas (length p).
        """
        if len(gammas) != len(betas):
            raise ValueError("Length of gammas and betas must be equal (p layers).")
        for q in range(self.n):
            self.h(q)
        p = len(gammas)
        for k in range(p):
            self.apply_cost_layer(gammas[k], edge_list)
            self.apply_mixer_layer(betas[k])
        cost = self.compute_cost_expectation(edge_list)
        return cost

    def expectation_ZZ(self, i: int, j: int):
        """
        Compute expectation value of Z_i Z_j on current pure state.

        Qubit indexing matches gate construction: qubit 0 is the most-significant
        Kronecker factor (MSB). Basis index bit ``(n-1-q)`` corresponds to qubit ``q``.
        """
        sv = self.state_vector.reshape(-1)
        probs = (sv.conj() * sv).real
        dim = sv.shape[0]
        idxs = torch.arange(dim, device=self.device)
        # MSB-first (same as kron order in single_qubit_gate / controlled_gate)
        bi = ((idxs >> (self.n - 1 - i)) & 1).to(torch.int64)
        bj = ((idxs >> (self.n - 1 - j)) & 1).to(torch.int64)
        zi = 1 - 2 * bi
        zj = 1 - 2 * bj
        val = (zi * zj).to(torch.float32)
        expect = torch.dot(probs.to(torch.float32), val)
        return float(expect.item())

    def compute_cost_expectation(self, edge_list: List[tuple]):
        """
        For MaxCut-like Hamiltonian H_C = 1/2 sum w (1 - Z_i Z_j)
        """
        total = 0.0
        for (i, j, w) in edge_list:
            zz = self.expectation_ZZ(i, j)
            total += 0.5 * w * (1.0 - zz)
        return float(total)


# ========================================================================================
