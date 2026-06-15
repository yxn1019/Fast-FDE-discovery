"""Shared models for fractional PDE discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from transporteq_discovery.compat import dataclass, field

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LibraryTerm:
    """A named candidate term used to build a regression library."""

    name: str
    values: FloatArray
    mechanism_specific: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).reshape(-1)


@dataclass(slots=True)
class SparseModel:
    """Sparse linear model returned by STRidge."""

    target_name: str
    term_names: tuple[str, ...]
    coefficients: FloatArray
    tolerance: float
    ridge_lambda: float
    mse: float
    residual_norm: float
    information_criterion: float
    support_frequency: FloatArray
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def support_mask(self) -> NDArray[np.bool_]:
        return np.abs(self.coefficients) > 0.0

    @property
    def support_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, active in zip(self.term_names, self.support_mask) if active
        )

    @property
    def active_term_count(self) -> int:
        return int(np.count_nonzero(self.support_mask))

    @property
    def mean_support_frequency(self) -> float:
        if self.active_term_count == 0:
            return 0.0
        return float(np.mean(self.support_frequency[self.support_mask]))

    def equation(self, precision: int = 4) -> str:
        """Render the discovered equation in a compact human-readable form."""

        active = []
        for coefficient, name in zip(self.coefficients, self.term_names):
            if np.isclose(coefficient, 0.0):
                continue
            active.append(f"{coefficient:.{precision}g}*{name}")
        rhs = " + ".join(active) if active else "0"
        return f"{self.target_name} = {rhs}"
