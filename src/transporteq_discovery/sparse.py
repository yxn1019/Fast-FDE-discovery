"""STRidge sparse regression utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from transporteq_discovery.compat import dataclass
from transporteq_discovery.config import STRidgeConfig
from transporteq_discovery.models import SparseModel

FloatArray = NDArray[np.float64]


def _normalize_columns(X: FloatArray, order: int) -> tuple[FloatArray, FloatArray]:
    if order == 0:
        return X.copy(), np.ones(X.shape[1], dtype=float)
    scales = np.linalg.norm(X, ord=order, axis=0)
    scales[scales == 0.0] = 1.0
    return X / scales, scales


def _least_squares(X: FloatArray, y: FloatArray, ridge_lambda: float) -> FloatArray:
    if ridge_lambda > 0.0:
        gram = X.T @ X + ridge_lambda * np.eye(X.shape[1], dtype=float)
        rhs = X.T @ y
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def sequential_threshold_ridge(
    X: FloatArray,
    y: FloatArray,
    ridge_lambda: float,
    max_iterations: int,
    tolerance: float,
    normalize_order: int,
) -> FloatArray:
    """Run a single STRidge solve for a fixed threshold."""

    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    normalized_X, scales = _normalize_columns(X, normalize_order)
    weights = _least_squares(normalized_X, y, ridge_lambda).reshape(-1)
    active = np.flatnonzero(np.abs(weights) >= tolerance)

    for _ in range(max_iterations):
        inactive = np.flatnonzero(np.abs(weights) < tolerance)
        inactive_set = set(inactive.tolist())
        new_active = np.array(
            [index for index in range(weights.size) if index not in inactive_set],
            dtype=int,
        )
        if np.array_equal(active, new_active):
            break
        active = new_active
        if active.size == 0:
            return np.zeros(X.shape[1], dtype=float)
        weights[inactive] = 0.0
        subweights = _least_squares(normalized_X[:, active], y, ridge_lambda).reshape(-1)
        weights[active] = subweights

    if active.size == 0:
        return np.zeros(X.shape[1], dtype=float)

    weights[:] = 0.0
    weights[active] = np.linalg.lstsq(normalized_X[:, active], y, rcond=None)[0].reshape(-1)
    return weights / scales


@dataclass(slots=True)
class STRidgeSelector:
    """Wrapper that scans tolerances and estimates support stability."""

    config: STRidgeConfig

    def _build_tolerance_grid(self) -> FloatArray:
        return np.linspace(
            self.config.tolerance_min,
            self.config.tolerance_max,
            num=self.config.num_tolerances,
            dtype=float,
        )

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        term_names: list[str] | tuple[str, ...],
        target_name: str,
    ) -> SparseModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        rng = np.random.default_rng(self.config.random_seed)
        sample_count = X.shape[0]
        train_count = min(sample_count - 1, max(3, int(self.config.train_fraction * sample_count)))
        train_indices = np.sort(rng.choice(sample_count, size=train_count, replace=False))
        test_mask = np.ones(sample_count, dtype=bool)
        test_mask[train_indices] = False
        if not np.any(test_mask):
            test_mask[train_indices[-1]] = True
        test_indices = np.flatnonzero(test_mask)

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        l0_penalty = self.config.l0_penalty
        if self.config.l0_condition_multiplier is not None:
            condition_number = np.linalg.cond(X)
            if not np.isfinite(condition_number):
                condition_number = 1.0e12
            l0_penalty = max(self.config.l0_condition_multiplier * condition_number, 1.0e-12)
        if l0_penalty is None:
            l0_penalty = max(self.config.ridge_lambda * np.linalg.cond(X_train), 1.0e-8)

        best_tolerance = 0.0
        best_objective = self._objective(X_test, y_test, np.linalg.lstsq(X_train, y_train, rcond=None)[0], l0_penalty)

        for tolerance in self._build_tolerance_grid():
            weights = sequential_threshold_ridge(
                X=X_train,
                y=y_train,
                ridge_lambda=self.config.ridge_lambda,
                max_iterations=self.config.inner_iterations,
                tolerance=float(tolerance),
                normalize_order=self.config.normalize_order,
            )
            objective = self._objective(X_test, y_test, weights, l0_penalty)
            if objective <= best_objective:
                best_objective = objective
                best_tolerance = float(tolerance)

        final_weights = sequential_threshold_ridge(
            X=X,
            y=y,
            ridge_lambda=self.config.ridge_lambda,
            max_iterations=self.config.inner_iterations,
            tolerance=best_tolerance,
            normalize_order=self.config.normalize_order,
        )
        residual = y - X @ final_weights
        mse = float(np.mean(residual**2))
        residual_norm = float(np.linalg.norm(residual, ord=2))
        active_count = int(np.count_nonzero(final_weights))
        information_criterion = float(sample_count * np.log(mse + 1.0e-12) + 2.0 * active_count)
        support_frequency = self._bootstrap_support(X, y, best_tolerance)
        return SparseModel(
            target_name=target_name,
            term_names=tuple(term_names),
            coefficients=final_weights.astype(float),
            tolerance=best_tolerance,
            ridge_lambda=self.config.ridge_lambda,
            mse=mse,
            residual_norm=residual_norm,
            information_criterion=information_criterion,
            support_frequency=support_frequency,
            sample_count=sample_count,
            metadata={
                "validation_objective": float(best_objective),
                "l0_penalty": float(l0_penalty),
                "l0_condition_multiplier": self.config.l0_condition_multiplier,
            },
        )

    @staticmethod
    def _objective(X: FloatArray, y: FloatArray, weights: FloatArray, l0_penalty: float) -> float:
        residual = y - X @ weights
        mse = np.mean(residual**2)
        return float(mse + l0_penalty * np.count_nonzero(weights) / max(len(y), 1))

    def _bootstrap_support(self, X: FloatArray, y: FloatArray, tolerance: float) -> FloatArray:
        if self.config.bootstrap_samples <= 0:
            return np.zeros(X.shape[1], dtype=float)

        rng = np.random.default_rng(self.config.random_seed + 1)
        support = np.zeros(X.shape[1], dtype=float)
        subset_size = min(X.shape[0], max(3, int(self.config.bootstrap_fraction * X.shape[0])))
        for _ in range(self.config.bootstrap_samples):
            indices = rng.choice(X.shape[0], size=subset_size, replace=True)
            weights = sequential_threshold_ridge(
                X=X[indices],
                y=y[indices],
                ridge_lambda=self.config.ridge_lambda,
                max_iterations=self.config.inner_iterations,
                tolerance=tolerance,
                normalize_order=self.config.normalize_order,
            )
            support += (np.abs(weights) > 0.0).astype(float)
        return support / self.config.bootstrap_samples
