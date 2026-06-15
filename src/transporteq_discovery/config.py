"""Configuration dataclasses used by fractional PDE discovery."""

from __future__ import annotations

from transporteq_discovery.compat import dataclass


@dataclass(slots=True)
class DatasetConfig:
    """File-backed dataset loading options for simple NPZ workflows."""

    path: str | None = None
    time_key: str = "time"
    position_key: str = "position"
    concentration_key: str = "concentration"
    inlet_key: str | None = "inlet"
    name: str = "transport-dataset"


@dataclass(slots=True)
class STRidgeConfig:
    """Sparse regression settings."""

    ridge_lambda: float = 1.0e-6
    inner_iterations: int = 25
    normalize_order: int = 2
    train_fraction: float = 0.8
    tolerance_min: float = 1.0e-4
    tolerance_max: float = 2.5e-1
    num_tolerances: int = 32
    l0_penalty: float | None = None
    l0_condition_multiplier: float | None = None
    bootstrap_samples: int = 24
    bootstrap_fraction: float = 0.8
    random_seed: int = 0
