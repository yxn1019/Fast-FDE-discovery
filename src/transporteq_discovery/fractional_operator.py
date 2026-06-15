"""Fractional derivative operators for 1D spatiotemporal fields."""

from __future__ import annotations

from math import ceil

import numpy as np
from numpy.typing import NDArray

from transporteq_discovery.compat import dataclass, field

FloatArray = NDArray[np.float64]


def _require_scipy_special():
    try:
        from scipy import special
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "Fractional operators require scipy. Install scipy or use the legacy wrappers."
        ) from exc
    return special


@dataclass(slots=True)
class FractionalDifferentialOperator:
    """Compute integer and Caputo time-fractional derivatives on a grid.

    The field shape is ``(time, position)`` to match ``TransportDataset``.
    Caputo derivatives are evaluated by the same Gauss-Jacobi idea used in the
    legacy DL-FDE scripts: interpolate the required integer derivative at the
    quadrature points on ``[t0, t]`` and apply the singular-kernel weights.
    """

    field: FloatArray
    time: FloatArray
    position: FloatArray
    quadrature_points: int = 8
    _time_derivative_cache: dict[int, FloatArray] = field(default_factory=dict, init=False)
    _space_derivative_cache: dict[int, FloatArray] = field(default_factory=dict, init=False)
    _caputo_cache: dict[float, FloatArray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.field = np.asarray(self.field, dtype=float)
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.position = np.asarray(self.position, dtype=float).reshape(-1)
        if self.field.ndim != 2:
            raise ValueError("field must have shape (time, position)")
        if self.field.shape != (self.time.size, self.position.size):
            raise ValueError("field shape must match time and position axes")
        if self.time.size < 3 or self.position.size < 3:
            raise ValueError("at least three time and position samples are required")
        if np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time values must be strictly increasing")
        if np.any(np.diff(self.position) <= 0.0):
            raise ValueError("position values must be strictly increasing")
        if self.quadrature_points < 2:
            raise ValueError("quadrature_points must be at least 2")

    def time_derivative(self, order: int = 1) -> FloatArray:
        """Return the integer time derivative of the requested order."""

        if order < 0:
            raise ValueError("order must be non-negative")
        if order == 0:
            return self.field.copy()
        if order in self._time_derivative_cache:
            return self._time_derivative_cache[order].copy()

        values = self.field.copy()
        for _ in range(order):
            values = np.gradient(values, self.time, axis=0, edge_order=2)
        values = np.asarray(values, dtype=float)
        self._time_derivative_cache[order] = values
        return values.copy()

    def space_derivative(self, order: int = 1) -> FloatArray:
        """Return the integer spatial derivative of the requested order."""

        if order < 0:
            raise ValueError("order must be non-negative")
        if order == 0:
            return self.field.copy()
        if order in self._space_derivative_cache:
            return self._space_derivative_cache[order].copy()

        values = self.field.copy()
        for _ in range(order):
            values = np.gradient(values, self.position, axis=1, edge_order=2)
        values = np.asarray(values, dtype=float)
        self._space_derivative_cache[order] = values
        return values.copy()

    def caputo_time_derivative(self, alpha: float) -> FloatArray:
        """Return the Caputo time derivative ``D_t^alpha field``.

        v1 targets ``0 < alpha < 1`` but the implementation also handles
        non-integer orders below 2 when enough smooth finite differences exist.
        Exact integer orders use the fast integer derivative path.
        """

        alpha = float(alpha)
        if alpha <= 0.0 or alpha >= 2.0:
            raise ValueError("alpha must satisfy 0 < alpha < 2 for the v1 operator")

        nearest_integer = round(alpha)
        if abs(alpha - nearest_integer) < 1.0e-12 and nearest_integer >= 1:
            return self.time_derivative(int(nearest_integer))

        cache_key = round(alpha, 12)
        if cache_key in self._caputo_cache:
            return self._caputo_cache[cache_key].copy()

        special = _require_scipy_special()
        integer_order = int(ceil(alpha))
        derivative_grid = self.time_derivative(integer_order)
        jacobi_alpha = integer_order - alpha - 1.0
        nodes, weights = special.roots_jacobi(self.quadrature_points, 0.0, jacobi_alpha)
        gamma_factor = special.gamma(integer_order - alpha)

        start = float(self.time[0])
        span = self.time - start
        transformed = (
            self.time.reshape(-1, 1)
            - 0.5 * span.reshape(-1, 1) * (nodes.reshape(1, -1) + 1.0)
        )
        scale = np.zeros_like(span, dtype=float)
        positive = span > 0.0
        scale[positive] = ((span[positive] / 2.0) ** (integer_order - alpha)) / gamma_factor

        result = np.zeros_like(self.field, dtype=float)
        sample_points = transformed.reshape(-1)
        for column in range(self.field.shape[1]):
            interpolated = np.interp(sample_points, self.time, derivative_grid[:, column])
            interpolated = interpolated.reshape(self.time.size, self.quadrature_points)
            result[:, column] = (interpolated @ weights) * scale

        self._caputo_cache[cache_key] = result
        return result.copy()


def analytic_caputo_power_law(time: FloatArray, power: float, alpha: float) -> FloatArray:
    """Analytic Caputo derivative of ``t**power`` for unit tests."""

    special = _require_scipy_special()
    time = np.asarray(time, dtype=float)
    coefficient = special.gamma(power + 1.0) / special.gamma(power + 1.0 - alpha)
    return coefficient * np.power(time, power - alpha)
