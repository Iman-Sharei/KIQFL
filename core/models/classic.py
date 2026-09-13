import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.quantum.circuit import quantum_circuit

class SimpleCNN(nn.Module):
    """Classical CNN baseline for FedAvg"""
    def __init__(self, input_channels, num_classes, image_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = 64 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QNN_Base(nn.Module):
    """Base Quantum Neural Network with variational circuit"""
    def __init__(self, n, L, num_classes=10, use_qaoa=False, qaoa_p=1, 
                 use_nisq_noise=False, nisq_error_rate=0.01):
        super(QNN_Base, self).__init__()
        self.n = n
        self.L = L
        self.num_classes = num_classes
        self.use_qaoa = use_qaoa
        self.qaoa_p = qaoa_p
        self.use_nisq_noise = use_nisq_noise
        self.nisq_error_rate = nisq_error_rate
        self.flatten = nn.Flatten()
        
        # Store the last circuit state for fidelity computation
        self._last_circuit_state = None
        
        # Variational parameters
        angles = torch.empty((L, n), dtype=torch.float32)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        
        # QAOA parameters
        if use_qaoa:
            self.gammas = nn.Parameter(torch.empty(qaoa_p, dtype=torch.float32))
            self.betas = nn.Parameter(torch.empty(qaoa_p, dtype=torch.float32))
            torch.nn.init.uniform_(self.gammas, 0.0, math.pi)
            torch.nn.init.uniform_(self.betas, 0.0, math.pi)
        
        self.linear = nn.Linear(2**n, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        target_dim = 2 ** self.n
        
        # Pad or truncate
        if x.shape[1] > target_dim:
            x = x[:, :target_dim]
        elif x.shape[1] < target_dim:
            pad = torch.zeros(x.shape[0], target_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Normalize to unit norm (amplitude encoding)
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        x = x / norm
        
        # Quantum circuit
        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)
        
        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
            if self.use_nisq_noise:
                qc.apply_noise(noise_model='bit_phase_flip', error_rate=self.nisq_error_rate)
        
        if self.use_qaoa:
            edge_list = [(i, (i + 1) % self.n, 1.0) for i in range(self.n)]
            for k in range(self.qaoa_p):
                # Keep tensors so QAOA parameters receive gradients
                qc.apply_cost_layer(self.gammas[k], edge_list)
                qc.apply_mixer_layer(self.betas[k])
                if self.use_nisq_noise:
                    qc.apply_noise(noise_model='depolarizing', error_rate=self.nisq_error_rate)
        
        # Store circuit state for fidelity computation (detached to avoid memory issues)
        self._last_circuit_state = qc.get_statevector().detach()
        
        probs = qc.probabilities()
        x = torch.real(probs).T
        x = self.linear(x)
        return x
    
    def get_circuit_state(self):
        """Return the quantum state from the last forward pass"""
        return self._last_circuit_state


class QNN_MNIST(QNN_Base):
    """QNN for MNIST dataset"""
    def __init__(self, n, L, use_qaoa=False, qaoa_p=1, use_nisq_noise=False, nisq_error_rate=0.01):
        super().__init__(n, L, num_classes=10, use_qaoa=use_qaoa, qaoa_p=qaoa_p,
                        use_nisq_noise=use_nisq_noise, nisq_error_rate=nisq_error_rate)
    
    def forward(self, x):
        # Pad MNIST 28x28 to 32x32 for consistency
        x = F.pad(x, (2, 2, 2, 2), "constant", 0)
        return super().forward(x)


class QNN_CIFAR(QNN_Base):
    """QNN for CIFAR dataset"""
    def __init__(self, n, L, num_classes=10, use_qaoa=False, qaoa_p=1, 
                 use_nisq_noise=False, nisq_error_rate=0.01):
        super().__init__(n, L, num_classes=num_classes, use_qaoa=use_qaoa, qaoa_p=qaoa_p,
                        use_nisq_noise=use_nisq_noise, nisq_error_rate=nisq_error_rate)


class QNN_Regressor(nn.Module):
    """QNN for regression tasks"""
    def __init__(self, n, L, use_qaoa=False, qaoa_p=1, use_nisq_noise=False, nisq_error_rate=0.01):
        super(QNN_Regressor, self).__init__()
        self.n = n
        self.L = L
        self.use_qaoa = use_qaoa
        self.qaoa_p = qaoa_p
        self.use_nisq_noise = use_nisq_noise
        self.nisq_error_rate = nisq_error_rate
        self.flatten = nn.Flatten()
        
        angles = torch.empty((L, n), dtype=torch.float32)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        
        if use_qaoa:
            self.gammas = nn.Parameter(torch.empty(qaoa_p, dtype=torch.float32))
            self.betas = nn.Parameter(torch.empty(qaoa_p, dtype=torch.float32))
            torch.nn.init.uniform_(self.gammas, 0.0, math.pi)
            torch.nn.init.uniform_(self.betas, 0.0, math.pi)
        
        self.linear = nn.Linear(2**n, 1)

    def forward(self, x):
        x = self.flatten(x)
        target_dim = 2 ** self.n
        
        if x.shape[1] > target_dim:
            x = x[:, :target_dim]
        elif x.shape[1] < target_dim:
            pad = torch.zeros(x.shape[0], target_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        x = x / norm
        
        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)
        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
            if self.use_nisq_noise:
                qc.apply_noise(noise_model='bit_phase_flip', error_rate=self.nisq_error_rate)
        
        if self.use_qaoa:
            edge_list = [(i, (i + 1) % self.n, 1.0) for i in range(self.n)]
            for k in range(self.qaoa_p):
                qc.apply_cost_layer(self.gammas[k], edge_list)
                qc.apply_mixer_layer(self.betas[k])
        
        probs = qc.probabilities()
        x = torch.real(probs).T
        x = self.linear(x)
        return x


# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================
