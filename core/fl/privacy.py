# privacy.py — Opacus-based DP-SGD for KIQFL local training

import copy

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    PrivacyEngine = None
    ModuleValidator = None
    OPACUS_AVAILABLE = False


def ensure_opacus():
    if not OPACUS_AVAILABLE:
        raise ImportError(
            "opacus is required for use_dp=True. Install: pip install opacus"
        )


def prepare_model_for_dp(model):
    """Fix BatchNorm / incompatible layers for Opacus."""
    ensure_opacus()
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
    return model


class FederatedPrivacyTracker:
    """Wrap Opacus PrivacyEngine for per-client local DP training."""

    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
        ensure_opacus()
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self._engines = []
        self._epsilons = []

    def make_private(self, model, optimizer, dataloader):
        model = prepare_model_for_dp(copy.deepcopy(model))
        engine = PrivacyEngine()
        model, optimizer, dataloader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        self._engines.append(engine)
        return model, optimizer, dataloader, engine

    def record_engine(self, engine):
        try:
            eps = engine.get_epsilon(self.delta)
            self._epsilons.append(float(eps))
        except Exception:
            self._epsilons.append(0.0)

    @property
    def cumulative_epsilon(self):
        return max(self._epsilons) if self._epsilons else 0.0

    @property
    def last_epsilon(self):
        return self._epsilons[-1] if self._epsilons else 0.0


class ClientPrivacySession:
    """One client local training round with Opacus DP-SGD."""

    def __init__(self, model, optimizer, dataloader,
                 noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
        ensure_opacus()
        self.delta = delta
        model = prepare_model_for_dp(model)
        self.engine = PrivacyEngine()
        self.model, self.optimizer, self.dataloader = self.engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    def epsilon(self):
        return float(self.engine.get_epsilon(self.delta))
