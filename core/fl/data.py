import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from core.quantum.fidelity import compute_state_fidelity, state_fidelity
from core.fl.metrics import evaluate_comprehensive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def partition_dataset_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset using Dirichlet distribution for realistic non-IID.
    Lower alpha = more heterogeneous. alpha=0.1: very skewed, alpha=100: near IID.
    """
    np.random.seed(seed)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        labels = np.array([y for _, y in dataset])
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        remainder = len(class_idx) - splits.sum()
        for i in range(int(remainder)):
            splits[i % num_clients] += 1
        current = 0
        for cid in range(num_clients):
            client_indices[cid].extend(class_idx[current:current + splits[cid]].tolist())
            current += splits[cid]
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return client_indices


# ========================================================================================
# 10. COMMUNICATION COST TRACKING
# ========================================================================================

def get_augmented_transform(dataset_name='mnist'):
    """Return data augmentation transforms for training."""
    if dataset_name in ('mnist', 'fashion_mnist'):
        norm = (0.1307,), (0.3081,) if dataset_name == 'mnist' else (0.5,), (0.5,)
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(*norm),
        ])
    elif dataset_name in ('cifar10', 'cifar100'):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([transforms.ToTensor()])


# ========================================================================================
# 8. PER-LAYER ADAPTIVE GRADIENT CLIPPING
# ========================================================================================

def subsample_clients(num_clients, participation_rate=1.0, rng=None):
    """
    Select a subset of clients for each federated round.

    For large-scale FL (50+ clients), full participation is impractical.
    This implements random client subsampling as in McMahan et al. (2017).

    Args:
        num_clients: Total number of clients
        participation_rate: Fraction of clients to select (0, 1]
        rng: Optional random.Random instance for reproducibility

    Returns:
        List of selected client indices
    """
    if rng is None:
        rng = random
    k = max(1, int(num_clients * participation_rate))
    if k >= num_clients:
        return list(range(num_clients))
    return sorted(rng.sample(range(num_clients), k))


def partition_dataset(dataset, num_clients, topology='ring', non_iid_alpha=None, seed=42):
    """Partition dataset among clients. If non_iid_alpha is set, use Dirichlet distribution."""
    if non_iid_alpha is not None:
        return partition_dataset_dirichlet(dataset, num_clients, alpha=non_iid_alpha, seed=seed)
    indices = list(range(len(dataset)))
    if topology == 'ring':
        client_indices = np.array_split(indices, num_clients)
    else:
        random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
    return [list(arr) for arr in client_indices]

def average_weights(state_dicts):
    """Equal-weight average (1/K). Prefer weighted_average_weights for FedAvg."""
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] = avg_state_dict[key] + state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict

def weighted_average_weights(state_dicts, data_sizes):
    """
    FedAvg aggregation (McMahan et al., 2017):
        w_global = sum_k (n_k / n) * w_k
    Falls back to equal weights if sizes missing/invalid.
    """
    if not state_dicts:
        raise ValueError("empty state_dicts")
    if data_sizes is None or len(data_sizes) != len(state_dicts):
        return average_weights(state_dicts)
    total = float(sum(data_sizes))
    if total <= 0:
        return average_weights(state_dicts)
    weights = [float(n) / total for n in data_sizes]
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        acc = weights[0] * state_dicts[0][key].float()
        for i in range(1, len(state_dicts)):
            acc = acc + weights[i] * state_dicts[i][key].float()
        avg_state_dict[key] = acc.to(dtype=state_dicts[0][key].dtype)
    return avg_state_dict

def _client_data_sizes(client_indices):
    """Number of examples per client partition."""
    return [len(idx) for idx in client_indices]

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on test data"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def build_epoch_evaluation(global_model, test_loader, criterion, config, epoch, global_epochs,
                           comm, fidelity_val, cumulative_comm_kb,
                           client_holdout_loaders=None, client_indices=None, all_labels=None):
    """Merge task + FL + QFL metrics for one federated round."""
    qaoa_cost = global_model.get_qaoa_cost() if hasattr(global_model, 'get_qaoa_cost') else 0.0
    return evaluate_comprehensive(
        global_model, test_loader, criterion, device, config=config,
        client_loaders=client_holdout_loaders,
        client_indices=client_indices,
        all_labels=all_labels,
        epoch=epoch,
        global_epochs=global_epochs,
        comm_effective_kb=comm.get('comm_effective_KB', 0.0),
        cumulative_comm_kb=cumulative_comm_kb,
        fidelity_mean=fidelity_val,
        fidelity_std=0.0,
        qaoa_cost=qaoa_cost,
    )

def get_client_state_vectors(num_clients, num_qubits):
    """
    Initialize quantum state vectors for KIQFL clients.
    Each client starts with a DIFFERENT random state to simulate heterogeneous initialization.
    This is physically realistic - each client has its own local quantum register.
    """
    state_vectors = []
    dim = 2**num_qubits
    for i in range(num_clients):
        # Create random Haar-distributed state vector
        # This is the correct way to sample uniformly from quantum state space
        real_part = torch.randn(dim, device=device)
        imag_part = torch.randn(dim, device=device)
        state = torch.complex(real_part, imag_part)
        # Normalize to unit vector
        state = state / torch.norm(state)
        state_vectors.append(state.reshape(-1, 1))
    return state_vectors

def compute_real_fidelity(old_states, new_states):
    """
    Compute REAL quantum fidelity between old and new states using Qiskit-compatible formula.
    
    Fidelity F = |⟨ψ_old|ψ_new⟩|² (following Qiskit's state_fidelity implementation)
    
    PHYSICAL INTERPRETATION:
    - F ≈ 0.0-0.3: States are very different (early training, high learning)
    - F ≈ 0.3-0.6: Moderate similarity (active learning phase)
    - F ≈ 0.6-0.8: States becoming similar (convergence beginning)
    - F ≈ 0.8-0.95: Near convergence
    - F ≈ 0.95-1.0: Converged (states barely changing)
    
    For random states in high-dimensional Hilbert space, expected fidelity ≈ 1/dim
    So for 8 qubits (dim=256), random fidelity ≈ 0.004
    """
    if len(old_states) == 0 or len(new_states) == 0:
        return 0.0  # No states = no fidelity
    
    total_fidelity = 0.0
    count = 0
    
    for old_state, new_state in zip(old_states, new_states):
        if old_state is None or new_state is None:
            continue
        
        # Use the Qiskit-compatible state_fidelity function from circuit.py
        fid_val = state_fidelity(old_state, new_state)
        total_fidelity += fid_val
        count += 1
    
    return total_fidelity / max(1, count) if count > 0 else 0.0

def compute_noise_levels(epoch, num_epochs, base_noise=0.05):
    """Compute decreasing noise levels as training progresses"""
    progress = epoch / max(1, num_epochs)
    noise = base_noise * (1.0 - 0.8 * progress)  # Decrease from base to 20% of base
    return noise

def load_dataset(dataset_name, data_fraction=0.1):
    """Load and prepare dataset with optional subsampling"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_channels, image_size, num_classes = 1, 28, 10
        
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_channels, image_size, num_classes = 1, 28, 10
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_channels, image_size, num_classes = 3, 32, 10
        
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        input_channels, image_size, num_classes = 3, 32, 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Subsample for faster training
    if data_fraction < 1.0:
        train_size = int(len(trainset) * data_fraction)
        test_size = int(len(testset) * data_fraction)
        trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size])
        testset, _ = random_split(testset, [test_size, len(testset) - test_size])
    
    return trainset, testset, input_channels, image_size, num_classes


# ========================================================================================
# ALGORITHM IMPLEMENTATIONS
# ========================================================================================

# ========================================================================================
# COLORED LOGGING UTILITIES
# ========================================================================================

