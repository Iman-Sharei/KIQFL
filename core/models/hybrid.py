import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.quantum.circuit import quantum_circuit

class Hybrid_QNN(nn.Module):
    """
    Hybrid Classical-Quantum Neural Network:
    CNN backbone -> dim reduction -> amplitude encoding ->
    variational circuit (Ry + CX + NISQ noise + QAOA) ->
    measurement -> classical output.
    """
    def __init__(self, n=None, L=None, cnn_model=None, num_qubits=8, num_layers=1,
                 num_classes=10, use_qaoa=False, qaoa_p=1, nisq_error_rate=0.0):
        super(Hybrid_QNN, self).__init__()
        self.num_qubits = n if n is not None else num_qubits
        self.num_layers = L if L is not None else num_layers
        self.num_classes = num_classes
        self.use_qaoa = use_qaoa
        self.qaoa_p = qaoa_p
        self.nisq_error_rate = nisq_error_rate
        if cnn_model is None:
            class _SCNN(nn.Module):
                def __init__(self, ic, nc, isz):
                    super().__init__()
                    self.conv1 = nn.Conv2d(ic, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fdim = 64 * (isz // 4) * (isz // 4)
                    self.fc1 = nn.Linear(self.fdim, 128)
                    self.fc2 = nn.Linear(128, nc)
                def forward(self, x):
                    x = self.pool(F.relu(self.conv1(x)))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = x.view(x.size(0), -1)
                    return F.relu(self.fc1(x))
            cnn_model = _SCNN(1, 64, 28) if num_classes == 10 else _SCNN(3, 128, 32)
        self.cnn = cnn_model
        angles = torch.empty((self.num_layers, self.num_qubits), dtype=torch.float32)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        if use_qaoa:
            self.gammas = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.betas = nn.Parameter(torch.empty(1, dtype=torch.float32))
            torch.nn.init.uniform_(self.gammas, 0.0, math.pi)
            torch.nn.init.uniform_(self.betas, 0.0, math.pi)
        self.output_layer = nn.Linear(2**self.num_qubits, num_classes)
        self._last_circuit_state = None

    def get_circuit_state(self):
        return self._last_circuit_state

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.cnn(x)
        batch_size = x.shape[0]
        target_dim = 2 ** self.num_qubits
        if x.shape[1] > target_dim:
            x = x[:, :target_dim]
        elif x.shape[1] < target_dim:
            pad = torch.zeros(batch_size, target_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        x = x / norm
        qc = quantum_circuit(num_qubits=self.num_qubits, state_vector=x.T)
        for l in range(self.num_layers):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
            if self.nisq_error_rate > 0:
                qc.apply_noise(noise_model='bit_phase_flip', error_rate=self.nisq_error_rate)
        if self.use_qaoa:
            edge_list = [(i, (i + 1) % self.num_qubits, 1.0) for i in range(self.num_qubits)]
            # Keep tensors (no .item()) so QAOA parameters receive gradients
            qc.apply_cost_layer(self.gammas[0], edge_list)
            qc.apply_mixer_layer(self.betas[0])
            self._last_qaoa_cost = qc.compute_cost_expectation(edge_list)
        self._last_circuit_state = qc.get_statevector().detach()
        probs = qc.probabilities()
        x = torch.real(probs).T
        x = self.output_layer(x)
        return x

    def get_qaoa_cost(self):
        return getattr(self, '_last_qaoa_cost', 0.0)

