import math

import torch

from core.fl.aggregation import is_quantum_param_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_parameter_shift_fisher_diag(model, sample_batch, criterion, device_arg):
    """Diagonal quantum Fisher via parameter-shift on angles (and QAOA params)."""
    X, y = sample_batch
    X, y = X.to(device_arg), y.to(device_arg)
    fisher = {}
    shift = math.pi / 2.0

    if hasattr(model, 'angles'):
        orig = model.angles.data.clone()
        diag = torch.zeros_like(orig)
        for l in range(orig.shape[0]):
            for q in range(orig.shape[1]):
                model.angles.data[l, q] = orig[l, q] + shift
                l_plus = _batch_loss(model, X, y, criterion)
                model.angles.data[l, q] = orig[l, q] - shift
                l_minus = _batch_loss(model, X, y, criterion)
                model.angles.data[l, q] = orig[l, q]
                deriv = (l_plus - l_minus) / 2.0
                diag[l, q] = deriv * deriv
        fisher['angles'] = diag.clamp(min=1e-8)

    for pname in ('gammas', 'betas'):
        if hasattr(model, pname):
            param = getattr(model, pname)
            orig = param.data.clone()
            diag = torch.zeros_like(orig)
            pdata = param.data.view(-1)
            ddata = diag.view(-1)
            for i in range(pdata.numel()):
                pdata[i] = orig.view(-1)[i] + shift
                l_plus = _batch_loss(model, X, y, criterion)
                pdata[i] = orig.view(-1)[i] - shift
                l_minus = _batch_loss(model, X, y, criterion)
                pdata[i] = orig.view(-1)[i]
                ddata[i] = ((l_plus - l_minus) / 2.0) ** 2
            param.data.copy_(orig)
            fisher[pname] = diag.clamp(min=1e-8)
    return fisher

def compute_diagonal_fisher_diag(model):
    """Diagonal Fisher from squared gradients (empirical fallback)."""
    diag = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if is_quantum_param_key(name):
            diag[name] = p.grad.detach().pow(2).clamp(min=1e-8)
    return diag

def compute_fisher_diag(model, sample_batch, criterion, device_arg, method='parameter_shift'):
    if method == 'parameter_shift' and sample_batch is not None:
        return compute_parameter_shift_fisher_diag(model, sample_batch, criterion, device_arg)
    return compute_diagonal_fisher_diag(model)

def apply_diagonal_natural_gradient(model, fisher_diag, eps=1e-4):
    """Scale gradients by inverse diagonal Fisher for quantum parameters."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name in fisher_diag:
            p.grad.data = p.grad.data / (fisher_diag[name] + eps)

def compute_layer_block_fisher(model, fisher_diag=None, structure='layer_block', damp=1e-3):
    """
    Layer-wise block-diagonal metric for quantum params (Qi/Stokes-style approx).

    - layer_block: for each angles[l,:], dense Q×Q block from grad outer-product,
      blended with parameter-shift diagonal when available.
    - layer_diag: independent layers, diagonal within each layer (cheap).

    Classical/CNN params keep Euclidean geometry (identity).
    Returns dict name -> list of (block_matrix, flat_index_slice) or diag tensor.
    """
    blocks = {}
    for name, p in model.named_parameters():
        if p.grad is None or not is_quantum_param_key(name):
            continue
        g = p.grad.detach()
        if name.endswith('angles') or name == 'angles':
            # Expected shape [L, Q]
            if g.dim() != 2:
                diag = fisher_diag.get('angles', g.pow(2)).to(g.device) if fisher_diag else g.pow(2)
                blocks[name] = {'type': 'diag', 'diag': diag.clamp(min=1e-8)}
                continue
            L, Q = g.shape
            layer_blocks = []
            ps_diag = None
            if fisher_diag and 'angles' in fisher_diag:
                ps_diag = fisher_diag['angles'].to(g.device)
            for l in range(L):
                gl = g[l].reshape(-1).float()
                if structure == 'layer_diag':
                    d = gl.pow(2)
                    if ps_diag is not None:
                        d = 0.5 * d + 0.5 * ps_diag[l].reshape(-1).float()
                    layer_blocks.append(('diag', d.clamp(min=1e-8)))
                else:
                    F = torch.outer(gl, gl)
                    if ps_diag is not None:
                        F = F + torch.diag(ps_diag[l].reshape(-1).float().clamp(min=1e-8))
                    F = F + damp * torch.eye(Q, device=g.device, dtype=torch.float32)
                    layer_blocks.append(('dense', F))
            blocks[name] = {'type': 'layers', 'layers': layer_blocks, 'shape': (L, Q)}
        else:
            # gammas / betas / other quantum: one dense or diag block
            flat = g.reshape(-1).float()
            key = name.split('.')[-1]
            ps = fisher_diag.get(key) if fisher_diag else None
            if structure == 'layer_diag' or flat.numel() > 64:
                d = flat.pow(2)
                if ps is not None:
                    d = 0.5 * d + 0.5 * ps.reshape(-1).float().to(d.device)
                blocks[name] = {'type': 'diag', 'diag': d.clamp(min=1e-8)}
            else:
                F = torch.outer(flat, flat)
                if ps is not None:
                    F = F + torch.diag(ps.reshape(-1).float().to(F.device).clamp(min=1e-8))
                F = F + damp * torch.eye(flat.numel(), device=g.device, dtype=torch.float32)
                blocks[name] = {'type': 'dense', 'matrix': F, 'shape': tuple(g.shape)}
    return blocks

def apply_block_natural_gradient(model, fisher_blocks, fisher_diag=None, eps=1e-4):
    """
    Precondition grads: g^+ ∇L using layer-block (or diagonal) metric.
    Leaves non-quantum parameters unchanged (Euclidean).
    """
    if not fisher_blocks and fisher_diag:
        apply_diagonal_natural_gradient(model, fisher_diag, eps=eps)
        return
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        info = fisher_blocks.get(name)
        if info is None:
            if fisher_diag and name in fisher_diag:
                p.grad.data = p.grad.data / (fisher_diag[name] + eps)
            continue
        g = p.grad.data
        if info['type'] == 'diag':
            p.grad.data = g / (info['diag'].to(g.device).reshape_as(g) + eps)
        elif info['type'] == 'dense':
            flat = g.reshape(-1).float()
            ng = torch.linalg.solve(info['matrix'], flat)
            p.grad.data = ng.reshape_as(g).to(dtype=g.dtype)
        elif info['type'] == 'layers':
            L, Q = info['shape']
            out = g.clone().float()
            for l, (btype, mat) in enumerate(info['layers']):
                gl = g[l].reshape(-1).float()
                if btype == 'diag':
                    out[l] = (gl / (mat + eps)).reshape(Q)
                else:
                    out[l] = torch.linalg.solve(mat, gl).reshape(Q)
            p.grad.data = out.to(dtype=g.dtype)

def apply_quantum_natural_gradient(model, sample_batch, criterion, device_arg,
                                   method='parameter_shift', structure='layer_block',
                                   eps=1e-4, damp=1e-3, fisher_diag=None):
    """
    Full local QNG step preconditioning (paper-inspired):
    1) optional parameter-shift diagonal Fisher
    2) layer-block metric + solve for quantum params

    If ``fisher_diag`` is provided, the expensive parameter-shift estimate is
    reused (periodic / representative-batch Fisher). Method and structure are
    unchanged; only the *frequency* of Fisher estimation is reduced.
    """
    if fisher_diag is None:
        if method == 'parameter_shift' and sample_batch is not None:
            fisher_diag = compute_parameter_shift_fisher_diag(
                model, sample_batch, criterion, device_arg)
        elif method == 'empirical':
            fisher_diag = compute_diagonal_fisher_diag(model)
    if structure in ('layer_block', 'layer_diag'):
        blocks = compute_layer_block_fisher(
            model, fisher_diag=fisher_diag, structure=structure, damp=damp)
        apply_block_natural_gradient(model, blocks, fisher_diag=fisher_diag, eps=eps)
    else:
        if fisher_diag is None:
            fisher_diag = compute_diagonal_fisher_diag(model)
        apply_diagonal_natural_gradient(model, fisher_diag, eps=eps)
    return fisher_diag

def extract_grad_state_dict(model):
    """Clone current .grad tensors as a state_dict-like mapping (zeros if missing)."""
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            out[name] = torch.zeros_like(p.data)
        else:
            out[name] = p.grad.detach().clone()
    return out

def zeros_like_grad_dict(grad_dict):
    return {k: torch.zeros_like(v) for k, v in grad_dict.items()}

def accumulate_grad_dict(acc, grads, weight=1.0):
    for k, v in grads.items():
        acc[k] = acc[k] + weight * v

def average_grad_dict(acc, n_steps):
    n = max(1, int(n_steps))
    return {k: v / n for k, v in acc.items()}

def weighted_average_grad_dicts(grad_dicts, data_sizes):
    """Σ (n_k/N) * g_k  — FQNGD server aggregation (Qi et al. Eq. 13 weights)."""
    if not grad_dicts:
        raise ValueError("empty grad_dicts")
    total = float(sum(data_sizes)) if data_sizes else float(len(grad_dicts))
    if total <= 0:
        total = float(len(grad_dicts))
        weights = [1.0 / len(grad_dicts)] * len(grad_dicts)
    else:
        weights = [float(n) / total for n in data_sizes]
    keys = grad_dicts[0].keys()
    out = {}
    for key in keys:
        acc = weights[0] * grad_dicts[0][key].float()
        for i in range(1, len(grad_dicts)):
            acc = acc + weights[i] * grad_dicts[i][key].float()
        out[key] = acc.to(dtype=grad_dicts[0][key].dtype)
    return out

def apply_grad_dict_to_model(model, grad_dict, lr):
    """Global Euclidean step: θ ← θ − η ḡ (ḡ already natural-preconditioned locally)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in grad_dict:
                continue
            p.data.add_(grad_dict[name].to(device=p.device, dtype=p.data.dtype), alpha=-float(lr))

def _batch_loss(model, X, y, criterion):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return criterion(model(X), y).item()
    finally:
        model.train(was_training)

