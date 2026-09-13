# evaluation_metrics.py — unified FL / QFL / task metrics for KIQFL

import math
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_task_metrics(model, dataloader, criterion, device_arg, num_classes=10):
    """Accuracy, loss, F1, precision, recall, balanced accuracy."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device_arg), labels.to(device_arg)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = max(1, len(all_labels))
    accuracy = float(np.mean(all_preds == all_labels))
    avg_loss = total_loss / n
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        balanced_accuracy_score, confusion_matrix,
    )
    f1_macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    f1_weighted = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    balanced_acc = float(balanced_accuracy_score(all_labels, all_preds))
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_matrix,
    }


def jain_fairness_index(values):
    """Jain's fairness index in [1/n, 1]."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 1.0
    s = np.sum(v)
    if s < 1e-12:
        return 0.0
    return float((s ** 2) / (len(v) * np.sum(v ** 2) + 1e-12))


def label_skew_hhi(client_indices, all_labels):
    """Mean Herfindahl index of per-client label distributions (higher = more skewed)."""
    hhis = []
    for idx in client_indices:
        labels = all_labels[idx] if isinstance(idx, np.ndarray) else [all_labels[i] for i in idx]
        if len(labels) == 0:
            continue
        _, counts = np.unique(labels, return_counts=True)
        props = counts / counts.sum()
        hhis.append(float(np.sum(props ** 2)))
    return float(np.mean(hhis)) if hhis else 0.0


def evaluate_per_client_accuracy(model, client_loaders, device_arg):
    """Holdout accuracy per participating client."""
    model.eval()
    accs = []
    with torch.no_grad():
        for loader in client_loaders:
            if loader is None or len(loader.dataset) == 0:
                accs.append(0.0)
                continue
            correct, total = 0, 0
            for X, y in loader:
                X, y = X.to(device_arg), y.to(device_arg)
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            accs.append(correct / max(1, total))
    return accs


def bootstrap_ci_95(values, n_boot=1000, seed=42):
    """95% bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        m = float(np.mean(v)) if len(v) else 0.0
        return m, m, m
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(v)), float(lo), float(hi)


def should_report_detailed(epoch, global_epochs, config):
    """Report full per-client metrics on final epoch and milestone rounds."""
    if config.get('report_detailed_metrics', True) is False:
        return epoch + 1 == global_epochs
    milestones = config.get('metric_milestones', [10, 25, 50])
    return (epoch + 1) == global_epochs or (epoch + 1) in milestones


def evaluate_comprehensive(
    model,
    test_loader,
    criterion,
    device_arg,
    config=None,
    client_loaders=None,
    client_indices=None,
    all_labels=None,
    epoch=0,
    global_epochs=1,
    comm_effective_kb=0.0,
    cumulative_comm_kb=0.0,
    fidelity_mean=0.0,
    fidelity_std=0.0,
    qaoa_cost=0.0,
):
    """
    Unified evaluation: task + FL fairness + optional QFL extras.
    Returns a flat dict suitable for merging into epoch results.
    """
    config = config or {}
    num_classes = config.get('num_classes', 10)
    detailed = should_report_detailed(epoch, global_epochs, config)

    task = compute_task_metrics(model, test_loader, criterion, device_arg, num_classes)
    out = {
        'loss': task['loss'],
        'accuracy': task['accuracy'],
        'f1_macro': task['f1_macro'],
        'f1_weighted': task['f1_weighted'],
        'precision_macro': task['precision_macro'],
        'recall_macro': task['recall_macro'],
        'balanced_accuracy': task['balanced_accuracy'],
        'fidelity_mean': fidelity_mean,
        'fidelity_std': fidelity_std,
        'qaoa_cost': float(qaoa_cost),
        'cumulative_comm_KB': float(cumulative_comm_kb),
        'comm_efficiency': float(task['accuracy'] / max(cumulative_comm_kb, 1e-6)),
    }

    if detailed and client_loaders:
        client_accs = evaluate_per_client_accuracy(model, client_loaders, device_arg)
        out['client_accuracies'] = client_accs
        out['client_acc_std'] = float(np.std(client_accs)) if client_accs else 0.0
        out['client_acc_min'] = float(np.min(client_accs)) if client_accs else 0.0
        out['client_acc_max'] = float(np.max(client_accs)) if client_accs else 0.0
        out['jain_fairness'] = jain_fairness_index(client_accs)
    else:
        out['jain_fairness'] = None

    if detailed and client_indices is not None and all_labels is not None:
        out['label_skew_hhi'] = label_skew_hhi(client_indices, all_labels)

    conv_th = config.get('convergence_acc_threshold', 0.90)
    if task['accuracy'] >= conv_th:
        out['rounds_to_target_acc'] = epoch + 1

    return out


def make_client_holdout_loaders(client_datasets, batch_size=32, fraction=0.1, seed=42):
    """10% holdout per client for fairness evaluation."""
    import random
    rng = random.Random(seed)
    loaders = []
    for ds in client_datasets:
        n = len(ds)
        if n < 2:
            loaders.append(None)
            continue
        n_hold = max(1, int(n * fraction))
        indices = list(range(n))
        rng.shuffle(indices)
        hold = torch.utils.data.Subset(ds, indices[:n_hold])
        loaders.append(torch.utils.data.DataLoader(
            hold, batch_size=min(batch_size, n_hold), shuffle=False))
    return loaders


def extract_labels_from_dataset(trainset, client_datasets=None):
    """Get label array aligned with trainset subset indices."""
    if hasattr(trainset, 'targets'):
        base = np.array(trainset.targets)
    elif hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'targets'):
        base = np.array(trainset.dataset.targets)
        if hasattr(trainset, 'indices'):
            base = base[np.array(trainset.indices)]
    else:
        base = np.array([y for _, y in trainset])
    return base


def compute_detailed_metrics(model, dataloader, criterion, device_arg, num_classes=10):
    """
    Compute: loss, accuracy, F1 (macro/weighted), precision, recall,
    per-class accuracy, confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device_arg), labels.to(device_arg)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))
    avg_loss = total_loss / max(1, len(all_labels))
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    f1_macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    f1_weighted = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        per_class_acc.append(float(np.mean(all_preds[mask] == all_labels[mask])) if mask.sum() > 0 else 0.0)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return {
        'loss': avg_loss, 'accuracy': accuracy,
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'precision': precision, 'recall': recall,
        'per_class_accuracy': per_class_acc, 'confusion_matrix': conf_matrix,
    }


# ========================================================================================
# 6. LEARNING RATE SCHEDULER FACTORY
# ========================================================================================
