"""FDE-style Laplace-Taylor discovery for the paper tsFADE benchmark."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from transporteq_discovery.data import TransportDataset
from transporteq_discovery.models import SparseModel


DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "models"
DEFAULT_ALPHA0_GRID = tuple(round(1.0 - 0.1 * i, 10) for i in range(10))
FDE_LIBRARY_TERMS = (
    "1",
    "H",
    "Hx",
    "Hxxx",
    "H^2",
    "H*Hx",
    "H*Hxx",
    "H*Hxxx",
    "H^2*Hx",
    "H^2*Hxx",
    "H^2*Hxxx",
)

_ANALYTIC_TFADE_PARAMS = {"alpha": 0.8, "diffusion": 0.5, "velocity": -1.0}


@dataclass(frozen=True)
class FractionalDiscoveryConfig:
    """Visible knobs for the current FDE-style mainline."""

    case_name: str = "tsfade_fft"
    checkpoint_file: Path | str | None = None
    model_root: Path | str = DEFAULT_MODEL_ROOT
    model_file: str = "ade"
    trained_point: int = 4000
    noise_level: float = 25.0
    activation: str = "tanh"
    hidden_layers: int = 8
    neurons: int = 20
    checkpoint_iteration: int = 5000
    x_min: float = 4.0
    x_max: float = 26.0
    x_step: float = 0.1
    t_min: float = 3.0
    t_max: float = 14.0
    t_step: float = 0.1
    fit_x_min: float | None = None
    fit_x_max: float | None = None
    fit_t_min: float | None = None
    fit_t_max: float | None = None
    alpha0_grid: tuple[float, ...] = DEFAULT_ALPHA0_GRID
    alpha_est_bounds: tuple[float, float] = (0.01, 1.05)
    laplace_s_min: float = 0.05
    laplace_s_max: float = 5.0
    laplace_s_points: int = 160
    laguerre_nodes: int = 15
    ridge_lambda: float = 2.0
    d_tol: float = 0.1
    maxit: int = 25
    str_iters: int = 10
    normalize: int = 0
    split: float = 0.8
    sparsity_lamb: float = 1.0e-3
    fractional_correction_sparsity_lamb: float | None = None
    fractional_correction_tol_scale: float = 0.1
    alpha_correction_tol: float = 1.0e-2
    beta_correction_tol: float = 1.0e-2
    delta_alpha_prune_threshold: float = 1.0e-2
    delta_alpha_bounds: tuple[float, float] = (-0.1, 0.0)
    delta_beta_prune_threshold: float = 1.0e-2
    enable_spatial_fractional: bool = False
    beta_bounds: tuple[float, float] = (1.01, 2.0)
    beta_reference_orders: tuple[float, ...] = (2.0,)
    spatial_log_quad_points: int = 10
    spatial_log_quad_method: str = "hardcoded"
    spatial_correction_method: str = "fourier_taylor"
    spatial_fractional_mode: str = "fourier_taylor"
    space_derivative_mode: str = "fft"
    time_operator_mode: str = "laplace_taylor"
    spatial_fractional_lower_bound: float = 0.0
    spatial_boundary_mode: str = "left_constant_subtract"
    spatial_boundary_value: float | None = None
    selection_objective: str = "physical-refit"
    order_radius_mode: str = "penalty"
    order_radius_tolerance: float = 0.05
    refit_mode: str = "linear"
    order_update_mode: str = "grid"
    allowed_physical_terms: tuple[str, ...] | None = None
    iter_start_alpha: float = 0.99
    iter_start_beta: float = 2.0
    iter_start_beta_values: tuple[float, ...] | None = None
    iter_max_iters: int = 8
    iter_order_tol: float = 0.005
    iter_damping: float = 1.0
    iter_max_step_alpha: float = 0.20
    iter_max_step_beta: float = 0.25
    iter_delta_alpha_bounds: tuple[float, float] = (-0.25, 0.25)
    iter_delta_beta_bounds: tuple[float, float] = (-0.15, 0.15)
    iter_stop_on_non_decreasing_objective: bool = True
    iter_objective_min_delta: float = 0.0
    iter_selection_mode: str = "best_objective"
    candidate_search: str = "fixed"
    generated_candidates: int = 100
    generated_max_terms: int = 5
    generated_top_structures: int = 30
    generated_seed: int = 7
    eqgpt_dictionary_path: Path | str | None = None
    eqgpt_model_checkpoint: Path | str | None = None
    eqgpt_optimize_epochs: int = 5
    eqgpt_finetune_epochs: int = 5
    eqgpt_learning_rate: float = 1.0e-5
    eqgpt_reward_sparsity_alpha: float = 0.2
    eqgpt_selection_mode: str = "stridge_objective"
    eqgpt_random_exploration: float = 0.2
    random_seed: int = 7

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_file is not None:
            return Path(self.checkpoint_file)
        noise = int(self.noise_level) if float(self.noise_level).is_integer() else self.noise_level
        return (
            Path(self.model_root)
            / f"{self.model_file}-{self.trained_point}-{noise}"
            / f"{self.model_file}-{self.checkpoint_iteration}.pkl"
        )


@dataclass(frozen=True)
class AlphaScanResult:
    alpha0: float
    alpha: float
    delta_alpha: float
    objective: float
    model: SparseModel


@dataclass(frozen=True)
class FractionalDiscoveryResult:
    alpha: float
    model: SparseModel
    initial_model: SparseModel
    alpha_scan: tuple[AlphaScanResult, ...]
    ranked_remainders: tuple[Any, ...] = ()
    used_remainder_terms: tuple[str, ...] = ()
    ga_triggered: bool = False

    @property
    def equation(self) -> str:
        return self.model.equation()


@dataclass(frozen=True)
class ReconstructedField:
    time: np.ndarray
    position: np.ndarray
    H: np.ndarray
    terms: dict[str, np.ndarray]
    checkpoint_path: Path
    training_metadata: dict[str, Any]


class FractionalPDEDiscoverer:
    """Discover a time-fractional PDE using the DL-FDE NN workflow."""

    def __init__(self, config: FractionalDiscoveryConfig | None = None):
        self.config = config or FractionalDiscoveryConfig()

    def discover(self, dataset: TransportDataset | None = None) -> FractionalDiscoveryResult:
        del dataset
        if self.config.case_name not in {"tsfade_fft", "analytic_tfade"}:
            raise NotImplementedError("The paper workflow supports tsfade_fft and analytic_tfade.")
        if self.config.time_operator_mode == "l1_pycaputo":
            return self._discover_l1_pycaputo()
        full_field = self.reconstruct_field()
        field = self._fit_window_field(full_field)
        s_values = np.linspace(
            self.config.laplace_s_min,
            self.config.laplace_s_max,
            self.config.laplace_s_points,
        )
        transforms, out_of_bounds_fraction = self._laplace_library(field, s_values)
        alpha_scan = self._scan_alpha0(field, transforms, s_values, out_of_bounds_fraction)
        self._score_scan_by_physical_refit(alpha_scan, field, transforms, s_values, operator_field=full_field)
        best = self._select_best_order_result(alpha_scan)
        final_model = self._refit_physical_model(field, transforms, s_values, best, operator_field=full_field)
        initial = next((item.model for item in alpha_scan if np.isclose(item.alpha0, 1.0)), alpha_scan[0].model)
        return FractionalDiscoveryResult(
            alpha=final_model.metadata.get("alpha", best.alpha),
            model=final_model,
            initial_model=initial,
            alpha_scan=tuple(alpha_scan),
            used_remainder_terms=("alpha_correction",) if "alpha_correction" in best.model.support_names else (),
            ga_triggered=False,
        )

    def _fit_window_field(self, field: ReconstructedField) -> ReconstructedField:
        x_mask = np.ones(field.position.shape, dtype=bool)
        t_mask = np.ones(field.time.shape, dtype=bool)
        if self.config.fit_x_min is not None:
            x_mask &= field.position >= float(self.config.fit_x_min)
        if self.config.fit_x_max is not None:
            x_mask &= field.position < float(self.config.fit_x_max)
        if self.config.fit_t_min is not None:
            t_mask &= field.time >= float(self.config.fit_t_min)
        if self.config.fit_t_max is not None:
            t_mask &= field.time < float(self.config.fit_t_max)
        if np.all(x_mask) and np.all(t_mask):
            return field
        if np.count_nonzero(x_mask) < 2 or np.count_nonzero(t_mask) < 2:
            raise ValueError("fit window must keep at least two x points and two t points")
        return ReconstructedField(
            time=field.time[t_mask],
            position=field.position[x_mask],
            H=field.H[np.ix_(t_mask, x_mask)],
            terms={name: values[np.ix_(t_mask, x_mask)] for name, values in field.terms.items()},
            checkpoint_path=field.checkpoint_path,
            training_metadata=field.training_metadata,
        )

    def _select_best_order_result(self, results: list[AlphaScanResult]) -> AlphaScanResult:
        """Select a valid, converged order candidate by regularized objective."""

        if not results:
            raise RuntimeError("No order candidates were evaluated.")
        for item in results:
            item.model.metadata["iteration_selected"] = False
            item.model.metadata["selected_by"] = f"minimum_valid_{self.config.selection_objective}_objective"
        valid = [item for item in results if self._order_result_is_valid(item)]
        if not valid:
            raise RuntimeError("No discovered alpha/beta candidate satisfies the configured order bounds.")
        finite_valid = [item for item in valid if np.isfinite(float(item.objective))]
        best = min(finite_valid or valid, key=self._selection_objective_value)
        best.model.metadata["iteration_selected"] = True
        best.model.metadata["selected_by"] = f"minimum_valid_{self.config.selection_objective}_objective"
        return best

    def _selection_objective_value(self, item: AlphaScanResult) -> float:
        if self.config.selection_objective == "augmented":
            return float(item.objective)
        if self.config.selection_objective != "physical-refit":
            raise ValueError("selection_objective must be 'augmented' or 'physical-refit'")
        value = item.model.metadata.get("physical_refit_selection_objective")
        return float(value) if isinstance(value, (int, float, np.floating)) else float(item.objective)

    def _order_result_is_valid(self, item: AlphaScanResult) -> bool:
        alpha = float(item.alpha)
        if not self.config.alpha_est_bounds[0] <= alpha <= self.config.alpha_est_bounds[1]:
            return False
        metadata = item.model.metadata
        if metadata.get("invalid_order", False):
            return False
        if not any(not self._is_fractional_correction_term(name) for name in item.model.support_names):
            return False
        if metadata.get("spatial_beta_model_valid") is False:
            return False
        if not self.config.enable_spatial_fractional:
            return True
        beta = metadata.get("spatial_beta")
        if beta is None:
            return not bool(metadata.get("spatial_correction_active", False))
        if not isinstance(beta, (int, float, np.floating)):
            return False
        return bool(self.config.beta_bounds[0] <= float(beta) <= self.config.beta_bounds[1])

    def reconstruct_field(self) -> ReconstructedField:
        if self.config.case_name == "analytic_tfade":
            return self._reconstruct_analytic_tfade_field()
        torch = self._torch()
        checkpoint_path = self.config.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing trained DL-FDE checkpoint: {checkpoint_path}")

        training_metadata = self._load_training_metadata(checkpoint_path)
        network_spec = self._network_spec(training_metadata)
        input_norm = self._input_normalization_spec(training_metadata)
        output_norm = self._output_normalization_spec(training_metadata)
        net = self._build_network(
            torch,
            activation=str(network_spec["activation"]),
            hidden_layers=int(network_spec["hidden_layers"]),
            neurons=int(network_spec["neurons"]),
        )
        try:
            state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint_path), map_location="cpu")
        net.load_state_dict(state)
        net.eval()

        x = torch.arange(self.config.x_min, self.config.x_max, self.config.x_step, dtype=torch.float32)
        t = torch.arange(self.config.t_min, self.config.t_max, self.config.t_step, dtype=torch.float32)
        tt, xx = torch.meshgrid(t, x, indexing="ij")
        if input_norm["mode"] == "unit_box":
            x_min = float(input_norm["x_range"][0])
            t_min = float(input_norm["t_range"][0])
            x_scale = float(input_norm["x_scale"])
            t_scale = float(input_norm["t_scale"])
            xx_net = (xx - x_min) / x_scale
            tt_net = (tt - t_min) / t_scale
        else:
            x_scale = 1.0
            xx_net = xx
            tt_net = tt
        database = torch.stack((xx_net.reshape(-1), tt_net.reshape(-1)), dim=1).requires_grad_(True)

        H = self._apply_output_activation(torch, net(database), training_metadata)
        if output_norm["mode"] == "unit_interval":
            H = H * float(output_norm["scale"]) + float(output_norm["shift"])
        nt = int(t.numel())
        nx = int(x.numel())
        H_np = self._tensor_grid(H, nt, nx)
        Hx_np, Hxx_np, Hxxx_np = self._linear_space_terms(
            H_np,
            torch=torch,
            database=database,
            H=H,
            nt=nt,
            nx=nx,
            x_scale=x_scale,
        )
        terms = {
            "1": np.ones_like(H_np),
            "H": H_np,
            "Hx": Hx_np,
            "Hxx": Hxx_np,
            "Hxxx": Hxxx_np,
            "H^2": H_np**2,
            "H*Hx": H_np * Hx_np,
            "H*Hxx": H_np * Hxx_np,
            "H*Hxxx": H_np * Hxxx_np,
            "H^2*Hx": H_np**2 * Hx_np,
            "H^2*Hxx": H_np**2 * Hxx_np,
            "H^2*Hxxx": H_np**2 * Hxxx_np,
        }
        if self.config.enable_spatial_fractional:
            terms.update(self._spatial_fractional_terms(torch, net, t, x, H_np, Hxx_np, training_metadata, output_norm))
        self._assert_finite("NN reconstruction", [H_np, *terms.values()])
        return ReconstructedField(
            time=t.detach().cpu().numpy().astype(float),
            position=x.detach().cpu().numpy().astype(float),
            H=H_np,
            terms=terms,
            checkpoint_path=checkpoint_path,
            training_metadata=training_metadata,
        )

    def _reconstruct_analytic_tfade_field(self) -> ReconstructedField:
        from tools.generate_analytic_tfade_sine import mittag_leffler_series

        alpha_true = _ANALYTIC_TFADE_PARAMS["alpha"]
        diffusion = _ANALYTIC_TFADE_PARAMS["diffusion"]
        velocity = _ANALYTIC_TFADE_PARAMS["velocity"]
        lam = diffusion + velocity**2 / (4.0 * diffusion)
        nx = max(2, int(round((self.config.x_max - self.config.x_min) / self.config.x_step)))
        nt = max(2, int(round((self.config.t_max - self.config.t_min) / self.config.t_step)))
        x_np = np.linspace(self.config.x_min, self.config.x_max, nx)
        t_np = np.linspace(self.config.t_min, self.config.t_max, nt)
        X, T = np.meshgrid(x_np, t_np)
        H_np = np.exp((velocity / (2.0 * diffusion)) * X) * np.sin(X) * mittag_leffler_series(
            -lam * T**alpha_true,
            alpha_true,
            terms=100,
        )
        Hx_np, Hxx_np, Hxxx_np = self._fft_linear_space_terms(H_np)
        terms = {
            "1": np.ones_like(H_np),
            "H": H_np,
            "Hx": Hx_np,
            "Hxx": Hxx_np,
            "Hxxx": Hxxx_np,
            "H^2": H_np**2,
            "H*Hx": H_np * Hx_np,
            "H*Hxx": H_np * Hxx_np,
            "H*Hxxx": H_np * Hxxx_np,
            "H^2*Hx": H_np**2 * Hx_np,
            "H^2*Hxx": H_np**2 * Hxx_np,
            "H^2*Hxxx": H_np**2 * Hxxx_np,
        }
        if self.config.enable_spatial_fractional:
            terms.update(self._spatial_fractional_terms(None, None, t_np, x_np, H_np, Hxx_np, {}))
        self._assert_finite("analytic reconstruction", [H_np, *terms.values()])
        training_metadata = {
            "config": {
                "case": "analytic_tfade",
                "alpha": alpha_true,
                "D": diffusion,
                "v": velocity,
                "lambda": lam,
                "field_shape": list(H_np.shape),
            },
            "summary": "Analytic time-fractional advection-diffusion field",
        }
        return ReconstructedField(
            time=t_np.astype(float),
            position=x_np.astype(float),
            H=H_np,
            terms=terms,
            checkpoint_path=Path("analytic_tfade"),
            training_metadata=training_metadata,
        )

    def _fft_linear_space_terms(self, H_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute integer spatial derivatives with the same FFT family as the generator."""

        from transporteq_discovery.spatial_fractional import fourier_fractional_derivative

        dx = float(self.config.x_step)
        Hx_np = fourier_fractional_derivative(H_np, dx=dx, beta=1.0, spatial_axis=-1)
        Hxx_np = fourier_fractional_derivative(H_np, dx=dx, beta=2.0, spatial_axis=-1)
        Hxxx_np = fourier_fractional_derivative(H_np, dx=dx, beta=3.0, spatial_axis=-1)
        return Hx_np, Hxx_np, Hxxx_np

    def _autodiff_linear_space_terms(
        self,
        torch: Any,
        database: Any,
        H: Any,
        nt: int,
        nx: int,
        x_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad = torch.autograd.grad(H.sum(), database, create_graph=True)[0]
        hx = grad[:, 0:1]
        hxx = torch.autograd.grad(hx.sum(), database, create_graph=True)[0][:, 0:1]
        hxxx = torch.autograd.grad(hxx.sum(), database, create_graph=True)[0][:, 0:1]
        x_scale_safe = float(max(x_scale, np.finfo(float).eps))
        return (
            self._tensor_grid(hx, nt, nx) / x_scale_safe,
            self._tensor_grid(hxx, nt, nx) / (x_scale_safe**2),
            self._tensor_grid(hxxx, nt, nx) / (x_scale_safe**3),
        )

    def _linear_space_terms(
        self,
        H_np: np.ndarray,
        *,
        torch: Any | None = None,
        database: Any | None = None,
        H: Any | None = None,
        nt: int | None = None,
        nx: int | None = None,
        x_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.space_derivative_mode == "fft":
            return self._fft_linear_space_terms(H_np)
        if self.config.space_derivative_mode != "autodiff":
            raise ValueError("space_derivative_mode must be 'fft' or 'autodiff'")
        if torch is None or database is None or H is None or nt is None or nx is None:
            return self._fft_linear_space_terms(H_np)
        return self._autodiff_linear_space_terms(torch, database, H, nt, nx, x_scale)

    def _discover_l1_pycaputo(self) -> FractionalDiscoveryResult:
        full_field = self.reconstruct_field()
        field = self._fit_window_field(full_field)
        scan = self._scan_alpha0_l1(field)
        self._score_scan_by_physical_refit(scan, field, operator_field=full_field)
        best = self._select_best_order_result(scan)
        final_model = self._refit_physical_model_direct(field, best, operator_field=full_field)
        initial = next((item.model for item in scan if np.isclose(item.alpha0, 1.0)), scan[0].model)
        return FractionalDiscoveryResult(
            alpha=final_model.metadata.get("alpha", best.alpha),
            model=final_model,
            initial_model=initial,
            alpha_scan=tuple(scan),
            used_remainder_terms=("alpha_correction",) if "alpha_correction" in best.model.support_names else (),
            ga_triggered=False,
        )

    def _caputo_l1_columnwise(self, values: np.ndarray, time: np.ndarray, alpha: float) -> np.ndarray:
        from pycaputo.differentiation.caputo import L1, diff
        from pycaputo.grid import make_uniform_points

        alpha_float = float(alpha)
        if not 0.0 < alpha_float <= 1.0:
            raise ValueError("L1/pycaputo time operator currently supports 0 < alpha <= 1")
        alpha_float = min(alpha_float, 1.0 - 1.0e-6)
        values = np.asarray(values, dtype=float)
        time = np.asarray(time, dtype=float)
        points = make_uniform_points(time.size, a=float(time[0]), b=float(time[-1]))
        result = np.zeros_like(values, dtype=float)
        method = L1(alpha=alpha_float)
        for column in range(values.shape[1]):
            derivative = np.asarray(diff(method, values[:, column], points), dtype=float)
            finite = np.isfinite(derivative)
            if not np.any(finite):
                raise FloatingPointError("L1/pycaputo produced no finite time-derivative entries")
            first_finite = int(np.flatnonzero(finite)[0])
            derivative[:first_finite] = derivative[first_finite]
            result[:, column] = derivative
        return result

    def _alpha_correction_l1(self, values: np.ndarray, time: np.ndarray, alpha0: float) -> np.ndarray:
        alpha0_float = float(alpha0)
        alpha0_float = min(alpha0_float, 1.0 - 1.0e-6)
        max_step = min(0.05, 0.45 * max(alpha0_float, 1.0e-3), 0.45 * max(1.0 - alpha0_float, 1.0e-3))
        step = max(1.0e-3, max_step)
        if np.isclose(alpha0_float, 1.0):
            main = self._caputo_l1_columnwise(values, time, alpha0_float)
            lower = self._caputo_l1_columnwise(values, time, alpha0_float - step)
            derivative = (main - lower) / step
        else:
            upper = self._caputo_l1_columnwise(values, time, alpha0_float + step)
            lower = self._caputo_l1_columnwise(values, time, alpha0_float - step)
            derivative = (upper - lower) / (2.0 * step)
        return -derivative

    def _scan_alpha0_l1(self, field: ReconstructedField) -> list[AlphaScanResult]:
        results: list[AlphaScanResult] = []
        beta0_options: tuple[float | None, ...]
        if self.config.enable_spatial_fractional:
            beta0_options = tuple(float(value) for value in self.config.beta_reference_orders)
        else:
            beta0_options = (None,)
        for alpha0 in self.config.alpha0_grid:
            target_grid = self._caputo_l1_columnwise(field.H, field.time, float(alpha0))
            alpha_correction_grid = self._alpha_correction_l1(field.H, field.time, float(alpha0))
            for beta0 in beta0_options:
                library_terms = self._library_terms_for_beta0(field, beta0)
                term_names = library_terms + ("alpha_correction",)
                theta = np.column_stack(
                    [field.terms[name].reshape(-1) for name in library_terms]
                    + [alpha_correction_grid.reshape(-1)]
                )
                target = target_grid.reshape(-1, 1)
                coefficients, loss, tolerance, l0_penalty, stridge_details = self._fit_fde_stridge(
                    theta,
                    target,
                    protected_indices=(),
                    term_names=term_names,
                )
                delta_alpha = float(coefficients[-1, 0])
                if abs(delta_alpha) < self.config.alpha_correction_tol:
                    coefficients[-1, 0] = 0.0
                    delta_alpha = 0.0
                alpha_est = float(alpha0 + delta_alpha)
                if self.config.enable_spatial_fractional:
                    self._prune_small_beta_corrections(term_names, coefficients)
                residual = target - theta @ coefficients
                mse = float(np.mean(residual**2))
                candidate_indices = tuple(range(theta.shape[1]))
                stridge_details = self._score_coefficients(
                    theta,
                    target,
                    coefficients,
                    l0_penalty,
                    candidate_indices,
                    term_names=term_names,
                )
                objective = float(stridge_details["validation_residual_norm"] + stridge_details["complexity_penalty"])
                if not self.config.delta_alpha_bounds[0] <= delta_alpha <= self.config.delta_alpha_bounds[1]:
                    objective = float("inf")
                if not self.config.alpha_est_bounds[0] <= alpha_est <= self.config.alpha_est_bounds[1]:
                    objective = float("inf")
                spatial_metadata = self._spatial_beta_metadata(term_names, coefficients.reshape(-1))
                network_spec = self._network_spec(field.training_metadata)
                metadata = {
                    "route": "fde_nn_l1_pycaputo",
                    "case": self.config.case_name,
                    "checkpoint": str(field.checkpoint_path),
                    "training_metadata": field.training_metadata,
                    "training_best": self._format_training_best(field.training_metadata),
                    "activation": network_spec["activation"],
                    "hidden_layers": network_spec["hidden_layers"],
                    "neurons": network_spec["neurons"],
                    "model_file": self.config.model_file,
                    "trained_point": self.config.trained_point,
                    "noise_level": self.config.noise_level,
                    "grid_shape": field.H.shape,
                    "x_range": (float(field.position[0]), float(field.position[-1])),
                    "t_range": (float(field.time[0]), float(field.time[-1])),
                    "initial_condition_time": float(field.time[0]),
                    "initial_condition_note": "L1/pycaputo acts directly on the sampled time grid",
                    "s_range": None,
                    "s_points": 0,
                    "laguerre_nodes": 0,
                    "laplace_out_of_bounds_fraction": 0.0,
                    "alpha0": float(alpha0),
                    "beta0": None if beta0 is None else float(beta0),
                    "delta_alpha": delta_alpha,
                    "alpha_correction_mode": "candidate_with_independent_delta_threshold",
                    "alpha_correction_coefficient": delta_alpha,
                    "time_operator_mode": self.config.time_operator_mode,
                    "space_derivative_mode": self.config.space_derivative_mode,
                    "spatial_fractional_mode": self.config.spatial_fractional_mode,
                    "objective": objective,
                    "fde_stridge_loss": float(loss),
                    "l0_penalty": float(l0_penalty),
                    "validation_residual_norm": float(stridge_details["validation_residual_norm"]),
                    "active_count_for_loss": int(stridge_details["active_count"]),
                    "complexity_penalty": float(stridge_details["complexity_penalty"]),
                    "loss_formula": "validation_residual_norm + cond(Theta)*sum(active_term_lamb)",
                    "tolerance": float(tolerance),
                    "matrix_shape": theta.shape,
                    "term_names": term_names,
                    "ridge_lambda": self.config.ridge_lambda,
                    "d_tol": self.config.d_tol,
                    "maxit": self.config.maxit,
                    "STR_iters": self.config.str_iters,
                    "normalize": self.config.normalize,
                    "split": self.config.split,
                    "sparsity_lamb": self.config.sparsity_lamb,
                    "fractional_correction_sparsity_lamb": self._fractional_correction_sparsity_lamb(),
                    "fractional_correction_tol_scale": self.config.fractional_correction_tol_scale,
                    "alpha_correction_tol": self.config.alpha_correction_tol,
                    "beta_correction_tol": self.config.beta_correction_tol,
                    "delta_alpha_prune_threshold": self.config.delta_alpha_prune_threshold,
                    "delta_alpha_bounds": self.config.delta_alpha_bounds,
                    "delta_beta_prune_threshold": self.config.delta_beta_prune_threshold,
                    "alpha0_count": len(self.config.alpha0_grid),
                    "beta0_count": len(beta0_options),
                    **spatial_metadata,
                }
                if not spatial_metadata["spatial_beta_model_valid"]:
                    objective = float("inf")
                    metadata["objective"] = objective
                model = SparseModel(
                    target_name="D_t^alpha H",
                    term_names=term_names,
                    coefficients=coefficients.reshape(-1),
                    tolerance=float(tolerance),
                    ridge_lambda=self.config.ridge_lambda,
                    mse=mse,
                    residual_norm=self._norm2(residual),
                    information_criterion=objective,
                    support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
                    sample_count=int(theta.shape[0]),
                    metadata=metadata,
                )
                results.append(AlphaScanResult(float(alpha0), alpha_est, delta_alpha, objective, model))
        return results

    def _scan_alpha0(
        self,
        field: ReconstructedField,
        transforms: dict[str, np.ndarray],
        s_values: np.ndarray,
        out_of_bounds_fraction: float,
    ) -> list[AlphaScanResult]:
        U = transforms["H"]
        u0 = field.H[0, :]
        results: list[AlphaScanResult] = []
        beta0_options: tuple[float | None, ...]
        if self.config.enable_spatial_fractional:
            beta0_options = tuple(float(value) for value in self.config.beta_reference_orders)
        else:
            beta0_options = (None,)

        for alpha0 in self.config.alpha0_grid:
            for beta0 in beta0_options:
                library_terms = self._library_terms_for_beta0(field, beta0)
                term_names = library_terms + ("alpha_correction",)
                base_matrix = np.column_stack([transforms[name].reshape(-1) for name in library_terms])
                s_col = s_values.reshape(-1, 1)
                y_grid = s_col**alpha0 * U - s_col ** (alpha0 - 1.0) * u0.reshape(1, -1)
                alpha_correction = -y_grid * np.log(s_col)
                theta = np.column_stack((base_matrix, alpha_correction.reshape(-1)))
                target = y_grid.reshape(-1, 1)
                protected_indices: tuple[int, ...] = ()
                coefficients, loss, tolerance, l0_penalty, stridge_details = self._fit_fde_stridge(
                    theta,
                    target,
                    protected_indices=protected_indices,
                    term_names=term_names,
                )
                delta_alpha = float(coefficients[-1, 0])
                if abs(delta_alpha) < self.config.alpha_correction_tol:
                    coefficients[-1, 0] = 0.0
                    delta_alpha = 0.0
                alpha_est = float(alpha0 + delta_alpha)
                if self.config.enable_spatial_fractional:
                    self._prune_small_beta_corrections(term_names, coefficients)
                residual = target - theta @ coefficients
                mse = float(np.mean(residual**2))
                protected_set = set(protected_indices)
                candidate_indices = tuple(index for index in range(theta.shape[1]) if index not in protected_set)
                stridge_details = self._score_coefficients(
                    theta,
                    target,
                    coefficients,
                    l0_penalty,
                    candidate_indices,
                    term_names=term_names,
                )
                loss = float(stridge_details["validation_residual_norm"] + stridge_details["complexity_penalty"])
                objective = float(loss)
                if not self.config.delta_alpha_bounds[0] <= delta_alpha <= self.config.delta_alpha_bounds[1]:
                    objective = float("inf")
                if not self.config.alpha_est_bounds[0] <= alpha_est <= self.config.alpha_est_bounds[1]:
                    objective = float("inf")
                spatial_metadata = self._spatial_beta_metadata(term_names, coefficients.reshape(-1))
                metadata = {
                    "route": "fde_nn_laplace_taylor",
                    "case": self.config.case_name,
                    "checkpoint": str(field.checkpoint_path),
                    "training_metadata": field.training_metadata,
                    "training_best": self._format_training_best(field.training_metadata),
                    "activation": self._network_spec(field.training_metadata)["activation"],
                    "hidden_layers": self._network_spec(field.training_metadata)["hidden_layers"],
                    "neurons": self._network_spec(field.training_metadata)["neurons"],
                    "model_file": self.config.model_file,
                    "trained_point": self.config.trained_point,
                    "noise_level": self.config.noise_level,
                    "grid_shape": field.H.shape,
                    "x_range": (float(field.position[0]), float(field.position[-1])),
                    "t_range": (float(field.time[0]), float(field.time[-1])),
                    "initial_condition_time": float(field.time[0]),
                    "initial_condition_note": "Caputo-Laplace term uses H(x, initial_condition_time)",
                    "s_range": (float(s_values[0]), float(s_values[-1])),
                    "s_points": int(s_values.size),
                    "laguerre_nodes": self.config.laguerre_nodes,
                    "space_derivative_mode": self.config.space_derivative_mode,
                    "spatial_fractional_mode": self.config.spatial_fractional_mode,
                    "time_operator_mode": self.config.time_operator_mode,
                    "laplace_out_of_bounds_fraction": out_of_bounds_fraction,
                    "alpha0": float(alpha0),
                    "beta0": None if beta0 is None else float(beta0),
                    "delta_alpha": delta_alpha,
                    "alpha_correction_mode": "candidate_with_independent_delta_threshold",
                    "alpha_correction_coefficient": delta_alpha,
                    "alpha_correction_tol": self.config.alpha_correction_tol,
                    "beta_correction_tol": self.config.beta_correction_tol,
                    "delta_alpha_prune_threshold": self.config.delta_alpha_prune_threshold,
                    "delta_alpha_bounds": self.config.delta_alpha_bounds,
                    "alpha0_count": len(self.config.alpha0_grid),
                    "beta0_count": len(beta0_options),
                    "objective": objective,
                    "fde_stridge_loss": float(loss),
                    "l0_penalty": float(l0_penalty),
                    "validation_residual_norm": float(stridge_details["validation_residual_norm"]),
                    "active_count_for_loss": int(stridge_details["active_count"]),
                    "complexity_penalty": float(stridge_details["complexity_penalty"]),
                    "loss_formula": "validation_residual_norm + cond(Theta)*sum(active_term_lamb)",
                    "tolerance": float(tolerance),
                    "matrix_shape": theta.shape,
                    "term_names": term_names,
                    **spatial_metadata,
                    "ridge_lambda": self.config.ridge_lambda,
                    "d_tol": self.config.d_tol,
                    "maxit": self.config.maxit,
                    "STR_iters": self.config.str_iters,
                    "normalize": self.config.normalize,
                    "split": self.config.split,
                    "sparsity_lamb": self.config.sparsity_lamb,
                    "fractional_correction_sparsity_lamb": self._fractional_correction_sparsity_lamb(),
                    "fractional_correction_tol_scale": self.config.fractional_correction_tol_scale,
                    "delta_beta_prune_threshold": self.config.delta_beta_prune_threshold,
                }
                if not spatial_metadata["spatial_beta_model_valid"]:
                    objective = float("inf")
                    metadata["objective"] = objective
                model = SparseModel(
                    target_name="D_t^alpha H",
                    term_names=term_names,
                    coefficients=coefficients.reshape(-1),
                    tolerance=float(tolerance),
                    ridge_lambda=self.config.ridge_lambda,
                    mse=mse,
                    residual_norm=self._norm2(residual),
                    information_criterion=objective,
                    support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
                    sample_count=int(theta.shape[0]),
                    metadata=metadata,
                )
                results.append(
                    AlphaScanResult(
                        alpha0=float(alpha0),
                        alpha=alpha_est,
                        delta_alpha=delta_alpha,
                        objective=objective,
                        model=model,
                    )
                )
        return results

    def _laplace_library(
        self,
        field: ReconstructedField,
        s_values: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float]:
        from scipy.special import roots_laguerre

        nodes, weights = roots_laguerre(self.config.laguerre_nodes)
        tau = field.time - field.time[0]
        query_tau = nodes.reshape(-1, 1) / s_values.reshape(1, -1)
        out_of_bounds = (query_tau < tau[0]) | (query_tau > tau[-1])
        out_of_bounds_fraction = float(np.count_nonzero(out_of_bounds) / out_of_bounds.size)
        transforms: dict[str, np.ndarray] = {}
        for name in self._library_terms(field):
            sampled = self._sample_time(field.terms[name], tau, query_tau)
            transforms[name] = np.einsum("n,nsx->sx", weights, sampled) / s_values.reshape(-1, 1)
        self._assert_finite("Laplace library", transforms.values())
        return transforms, out_of_bounds_fraction

    def _laplace_term(self, values: np.ndarray, field: ReconstructedField, s_values: np.ndarray) -> np.ndarray:
        from scipy.special import roots_laguerre

        nodes, weights = roots_laguerre(self.config.laguerre_nodes)
        tau = field.time - field.time[0]
        query_tau = nodes.reshape(-1, 1) / s_values.reshape(1, -1)
        sampled = self._sample_time(values, tau, query_tau)
        return np.einsum("n,nsx->sx", weights, sampled) / s_values.reshape(-1, 1)

    def _score_scan_by_physical_refit(
        self,
        scan: list[AlphaScanResult],
        field: ReconstructedField,
        transforms: dict[str, np.ndarray] | None = None,
        s_values: np.ndarray | None = None,
        *,
        operator_field: ReconstructedField | None = None,
    ) -> None:
        """Score each scan candidate with a physical refit objective.

        When transforms/s_values are None (L1/pycaputo route), falls back to
        _refit_physical_model_direct instead of the Laplace-domain refit.
        """
        direct = transforms is None or s_values is None
        for item in scan:
            metadata = item.model.metadata
            if not np.isfinite(float(item.objective)) or not self._order_result_is_valid(item):
                metadata["physical_refit_selection_objective"] = float("inf")
                continue
            if direct:
                refit = self._refit_physical_model_direct(field, item, operator_field=operator_field)
            else:
                refit = self._refit_physical_model(field, transforms, s_values, item, operator_field=operator_field)
            refit_meta = refit.metadata
            residual_norm = float(refit_meta.get("physical_refit_validation_residual_norm", refit.residual_norm))
            condition_penalty = float(refit_meta.get("physical_refit_condition_penalty", 0.0))
            radius_penalty = self._order_radius_penalty(item, field)
            objective = float(residual_norm + condition_penalty + radius_penalty)
            if self.config.order_radius_mode == "strict" and radius_penalty > 0.0:
                objective = float("inf")
            metadata.update(
                {
                    "physical_refit_selection_objective": objective,
                    "physical_refit_selection_residual_norm": residual_norm,
                    "physical_refit_selection_condition_penalty": condition_penalty,
                    "order_radius_penalty": radius_penalty,
                    "selection_objective_mode": self.config.selection_objective,
                    "order_radius_mode": self.config.order_radius_mode,
                    "order_radius_tolerance": self.config.order_radius_tolerance,
                }
            )

    def _order_radius_penalty(self, item: AlphaScanResult, field: ReconstructedField) -> float:
        mode = self.config.order_radius_mode
        if mode == "off":
            return 0.0
        if mode not in {"penalty", "strict"}:
            raise ValueError("order_radius_mode must be 'off', 'penalty', or 'strict'")
        radius_alpha, radius_beta = self._effective_order_radii(field)
        excess = 0.0
        if radius_alpha > 0.0:
            excess += max(0.0, abs(float(item.delta_alpha)) / radius_alpha - 1.0) ** 2
        beta_delta = item.model.metadata.get("spatial_delta_beta")
        if isinstance(beta_delta, (int, float, np.floating)) and radius_beta > 0.0:
            excess += max(0.0, abs(float(beta_delta)) / radius_beta - 1.0) ** 2
        return float(1.0e3 * excess)

    def _effective_order_radii(self, field: ReconstructedField) -> tuple[float, float]:
        a = self._first_order_remainder_radius(float(self.config.order_radius_tolerance))
        s_min = max(float(self.config.laplace_s_min), 1.0e-12)
        s_max = max(float(self.config.laplace_s_max), s_min * (1.0 + 1.0e-12))
        mt = 0.5 * np.log(s_max / s_min)
        alpha_radius = float("inf") if mt <= 0.0 else float(a / mt)
        x_span = float(field.position[-1] - field.position[0] + self.config.x_step)
        dx = float(self.config.x_step)
        if x_span <= 0.0 or dx <= 0.0:
            return alpha_radius, float("inf")
        k_min = 2.0 * np.pi / x_span
        k_max = np.pi / dx
        mk = np.sqrt(0.25 * np.log(k_max / k_min) ** 2 + (0.5 * np.pi) ** 2)
        beta_radius = float("inf") if mk <= 0.0 else float(a / mk)
        return alpha_radius, beta_radius

    @staticmethod
    def _first_order_remainder_radius(tolerance: float) -> float:
        eta = max(float(tolerance), 0.0)
        if eta <= 0.0:
            return 0.0
        lo, hi = 0.0, 1.0
        while np.exp(hi) - 1.0 - hi < eta:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.exp(mid) - 1.0 - mid <= eta:
                lo = mid
            else:
                hi = mid
        return float(lo)

    def _refit_physical_model(
        self,
        field: ReconstructedField,
        transforms: dict[str, np.ndarray],
        s_values: np.ndarray,
        best: AlphaScanResult,
        *,
        operator_field: ReconstructedField | None = None,
    ) -> SparseModel:
        """Refit physical coefficients after alpha/beta have been identified.

        The Taylor-augmented coefficients identify the local order correction.
        When a spatial beta is available, the reported diffusion coefficient is
        re-estimated on the actual Fourier column ``D_x^beta H``.
        """

        metadata = dict(best.model.metadata)
        beta = metadata.get("spatial_beta")
        active = list(best.model.support_names)
        term_names: list[str] = []
        columns: list[np.ndarray] = []
        added_fractional_space = False

        for name in active:
            if name == "alpha_correction" or name.startswith("spatial_correction_beta0="):
                continue
            if name.startswith("spatial_main_beta0="):
                if isinstance(beta, (int, float)) and not added_fractional_space:
                    term_names.append(f"D_x^{float(beta):.8g} H")
                    columns.append(self._laplace_final_spatial_beta(field, s_values, float(beta), operator_field=operator_field))
                    added_fractional_space = True
                continue
            if name == "Hxx" and isinstance(beta, (int, float)) and np.isclose(metadata.get("beta0"), 2.0):
                if not added_fractional_space:
                    term_names.append(f"D_x^{float(beta):.8g} H")
                    columns.append(self._laplace_final_spatial_beta(field, s_values, float(beta), operator_field=operator_field))
                    added_fractional_space = True
                continue
            if name in transforms:
                term_names.append(name)
                columns.append(transforms[name].reshape(-1))

        if not term_names:
            return best.model

        U = transforms["H"]
        u0 = field.H[0, :]
        s_col = s_values.reshape(-1, 1)
        target = (s_col**best.alpha * U - s_col ** (best.alpha - 1.0) * u0.reshape(1, -1)).reshape(-1, 1)
        matrix = np.column_stack(columns)
        coefficients = self._lstsq(matrix, target)
        residual = target - matrix @ coefficients
        validation_residual_norm = self._validation_residual_norm(matrix, target, coefficients)
        condition = self._finite_condition(matrix)
        condition_penalty = float(self.config.sparsity_lamb * condition * np.count_nonzero(coefficients))
        metadata.update(
            {
                "physical_refit": True,
                "physical_refit_note": "coefficients refit after replacing Taylor beta columns by D_x^beta H",
                "physical_refit_terms": tuple(term_names),
                "physical_refit_spatial_fractional_mode": self.config.spatial_fractional_mode,
                "physical_refit_space_derivative_mode": self.config.space_derivative_mode,
                "physical_refit_time_operator_mode": self.config.time_operator_mode,
                "physical_refit_mse": float(np.mean(residual**2)),
                "physical_refit_residual_norm": self._norm2(residual),
                "physical_refit_validation_residual_norm": validation_residual_norm,
                "physical_refit_condition": condition,
                "physical_refit_condition_penalty": condition_penalty,
                "internal_augmented_terms": best.model.term_names,
                "internal_augmented_coefficients": tuple(float(v) for v in best.model.coefficients),
                "alpha": float(best.alpha),
            }
        )
        return SparseModel(
            target_name="D_t^alpha H",
            term_names=tuple(term_names),
            coefficients=coefficients.reshape(-1),
            tolerance=best.model.tolerance,
            ridge_lambda=best.model.ridge_lambda,
            mse=float(np.mean(residual**2)),
            residual_norm=self._norm2(residual),
            information_criterion=best.model.information_criterion,
            support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
            sample_count=int(matrix.shape[0]),
            metadata=metadata,
        )

    def _refit_physical_model_direct(
        self,
        field: ReconstructedField,
        best: AlphaScanResult,
        *,
        operator_field: ReconstructedField | None = None,
    ) -> SparseModel:
        metadata = dict(best.model.metadata)
        beta = metadata.get("spatial_beta")
        active = list(best.model.support_names)
        term_names: list[str] = []
        columns: list[np.ndarray] = []
        added_fractional_space = False
        source = operator_field or field

        for name in active:
            if name == "alpha_correction" or name.startswith("spatial_correction_beta0="):
                continue
            if name.startswith("spatial_main_beta0="):
                if isinstance(beta, (int, float)) and not added_fractional_space:
                    values = self._spatial_fractional_values(
                        source.H,
                        source.position,
                        float(beta),
                        u_xx_values=source.terms.get("Hxx"),
                    )
                    if source is not field:
                        values = self._restrict_field_values(values, source, field)
                    term_names.append(f"D_x^{float(beta):.8g} H")
                    columns.append(values.reshape(-1))
                    added_fractional_space = True
                continue
            if name == "Hxx" and isinstance(beta, (int, float)) and np.isclose(metadata.get("beta0"), 2.0):
                if not added_fractional_space:
                    values = self._spatial_fractional_values(
                        source.H,
                        source.position,
                        float(beta),
                        u_xx_values=source.terms.get("Hxx"),
                    )
                    if source is not field:
                        values = self._restrict_field_values(values, source, field)
                    term_names.append(f"D_x^{float(beta):.8g} H")
                    columns.append(values.reshape(-1))
                    added_fractional_space = True
                continue
            if name in field.terms:
                term_names.append(name)
                columns.append(field.terms[name].reshape(-1))

        if not term_names:
            return best.model

        target = self._caputo_l1_columnwise(field.H, field.time, float(best.alpha)).reshape(-1, 1)
        matrix = np.column_stack(columns)
        coefficients = self._lstsq(matrix, target)
        residual = target - matrix @ coefficients
        validation_residual_norm = self._validation_residual_norm(matrix, target, coefficients)
        condition = self._finite_condition(matrix)
        condition_penalty = float(self.config.sparsity_lamb * condition * np.count_nonzero(coefficients))
        metadata.update(
            {
                "physical_refit": True,
                "physical_refit_note": "coefficients refit after replacing Taylor beta columns by D_x^beta H on the L1 time grid",
                "physical_refit_terms": tuple(term_names),
                "physical_refit_spatial_fractional_mode": self.config.spatial_fractional_mode,
                "physical_refit_space_derivative_mode": self.config.space_derivative_mode,
                "physical_refit_time_operator_mode": self.config.time_operator_mode,
                "physical_refit_mse": float(np.mean(residual**2)),
                "physical_refit_residual_norm": self._norm2(residual),
                "physical_refit_validation_residual_norm": validation_residual_norm,
                "physical_refit_condition": condition,
                "physical_refit_condition_penalty": condition_penalty,
                "internal_augmented_terms": best.model.term_names,
                "internal_augmented_coefficients": tuple(float(v) for v in best.model.coefficients),
                "alpha": float(best.alpha),
            }
        )
        return SparseModel(
            target_name="D_t^alpha H",
            term_names=tuple(term_names),
            coefficients=coefficients.reshape(-1),
            tolerance=best.model.tolerance,
            ridge_lambda=best.model.ridge_lambda,
            mse=float(np.mean(residual**2)),
            residual_norm=self._norm2(residual),
            information_criterion=best.model.information_criterion,
            support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
            sample_count=int(matrix.shape[0]),
            metadata=metadata,
        )

    def _validation_residual_norm(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        coefficients: np.ndarray,
    ) -> float:
        rng = np.random.default_rng(self.config.random_seed)
        n_rows = matrix.shape[0]
        train_count = int(n_rows * self.config.split)
        train = rng.choice(n_rows, train_count, replace=False)
        test_mask = np.ones(n_rows, dtype=bool)
        test_mask[train] = False
        test = np.flatnonzero(test_mask)
        return self._norm2(target[test, :] - matrix[test, :] @ coefficients)

    def _laplace_final_spatial_beta(
        self,
        field: ReconstructedField,
        s_values: np.ndarray,
        beta: float,
        *,
        operator_field: ReconstructedField | None = None,
    ) -> np.ndarray:
        source = operator_field or field
        values = self._spatial_fractional_values(
            source.H,
            source.position,
            float(beta),
            u_xx_values=source.terms.get("Hxx"),
        )
        if source is not field:
            values = self._restrict_field_values(values, source, field)
        return self._laplace_term(values, field, s_values).reshape(-1)

    def _spatial_fractional_values(
        self,
        values: np.ndarray,
        x_grid: np.ndarray,
        beta: float,
        *,
        u_xx_values: np.ndarray | None = None,
    ) -> np.ndarray:
        from transporteq_discovery.spatial_fractional import (
            fourier_fractional_derivative,
            homogenize_left_boundary,
            spatial_order_derivative_beta,
        )

        mode = self.config.spatial_fractional_mode
        dx = float(x_grid[1] - x_grid[0])
        beta_float = float(beta)
        if mode == "fourier_taylor":
            return fourier_fractional_derivative(values, dx=dx, beta=beta_float, spatial_axis=-1)
        boundary_slope = None
        if self.config.spatial_boundary_mode == "left_caputo":
            boundary_slope = np.gradient(values, x_grid, axis=-1, edge_order=2)[..., :1]
        operator_values = homogenize_left_boundary(
            values,
            x_grid,
            mode=self.config.spatial_boundary_mode,
            boundary_value=self.config.spatial_boundary_value,
            boundary_slope=boundary_slope,
        )
        main, _ = spatial_order_derivative_beta(
            operator_values,
            x_grid,
            beta_0=beta_float,
            n_quad=int(self.config.spatial_log_quad_points),
            method=mode,
            u_xx_values=u_xx_values,
        )
        return main

    @staticmethod
    def _restrict_field_values(
        values: np.ndarray,
        source: ReconstructedField,
        target: ReconstructedField,
    ) -> np.ndarray:
        t_indices = [int(np.argmin(np.abs(source.time - time))) for time in target.time]
        x_indices = [int(np.argmin(np.abs(source.position - position))) for position in target.position]
        if not np.allclose(source.time[t_indices], target.time) or not np.allclose(
            source.position[x_indices], target.position
        ):
            raise ValueError("target fit window is not aligned with the source reconstruction grid")
        return values[np.ix_(t_indices, x_indices)]

    def _library_terms(self, field: ReconstructedField) -> tuple[str, ...]:
        extras = tuple(name for name in field.terms if name not in FDE_LIBRARY_TERMS and name != "Hxx")
        return FDE_LIBRARY_TERMS + extras

    def _library_terms_for_beta0(self, field: ReconstructedField, beta0: float | None) -> tuple[str, ...]:
        if beta0 is None:
            return FDE_LIBRARY_TERMS
        beta0_float = float(beta0)
        terms = list(self._base_terms_for_spatial_beta(beta0_float))
        main = f"spatial_main_beta0={beta0_float:.1f}"
        correction = f"spatial_correction_beta0={beta0_float:.1f}"
        if main in field.terms:
            terms.append(main)
        if correction in field.terms:
            terms.append(correction)
        return tuple(terms)

    def _base_terms_for_spatial_beta(self, beta0: float) -> tuple[str, ...]:
        del beta0
        return FDE_LIBRARY_TERMS

    def _spatial_fractional_terms(
        self,
        torch: Any,
        net: Any,
        t: Any,
        x: Any,
        H_np: np.ndarray,
        Hxx_np: np.ndarray,
        training_metadata: dict[str, Any],
        output_norm: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        from transporteq_discovery.spatial_fractional import (
            build_spatial_fractional_candidates,
            homogenize_left_boundary,
            spatial_correction_column,
            spatial_order_derivative_beta,
        )

        terms: dict[str, np.ndarray] = {}
        use_fourier = self.config.spatial_fractional_mode == "fourier_taylor"
        if use_fourier and (torch is None or net is None):
            dx = float(self.config.x_step)
            for beta0 in self.config.beta_reference_orders:
                beta0_float = float(beta0)
                main, correction, _ = build_spatial_fractional_candidates(
                    H_np,
                    dx=dx,
                    beta_0=beta0_float,
                    spatial_axis=-1,
                )
                terms[f"spatial_main_beta0={beta0_float:.1f}"] = main
                terms[f"spatial_correction_beta0={beta0_float:.1f}"] = correction
            return terms
        if torch is None or net is None:
            x_np = np.asarray(x, dtype=float)
            if use_fourier:
                raise ValueError("Fourier spatial fractional mode expected a checkpoint-backed NN surrogate here")
            hxx_values = Hxx_np if Hxx_np.shape == H_np.shape else None
            operator_values = homogenize_left_boundary(
                H_np,
                x_np,
                mode=self.config.spatial_boundary_mode,
                boundary_value=self.config.spatial_boundary_value,
                boundary_slope=np.gradient(H_np, x_np, axis=-1, edge_order=2)[..., :1]
                if self.config.spatial_boundary_mode == "left_caputo"
                else None,
            )
            for beta0 in self.config.beta_reference_orders:
                beta0_float = float(beta0)
                if np.isclose(beta0_float, 2.0):
                    correction = spatial_correction_column(
                        u_data=operator_values,
                        x_grid=x_np,
                        dx=float(self.config.x_step),
                        beta_0=beta0_float,
                        n_quad=int(self.config.spatial_log_quad_points),
                        u_xx_values=hxx_values,
                        method=self.config.spatial_log_quad_method,
                        correction_method=self.config.spatial_fractional_mode,
                    )
                    terms[f"spatial_correction_beta0={beta0_float:.1f}"] = -correction
                    continue
                main, correction = spatial_order_derivative_beta(
                    operator_values,
                    x_np,
                    beta_0=beta0_float,
                    n_quad=int(self.config.spatial_log_quad_points),
                    method=self.config.spatial_fractional_mode,
                    u_xx_values=hxx_values,
                )
                terms[f"spatial_main_beta0={beta0_float:.1f}"] = main
                terms[f"spatial_correction_beta0={beta0_float:.1f}"] = -correction
            return terms
        if use_fourier:
            lower, upper = self._fourier_spatial_bounds(training_metadata)
        else:
            lower = float(self.config.spatial_fractional_lower_bound)
            upper = float(self.config.x_max)
        x_aux = torch.arange(float(lower), float(upper), self.config.x_step, dtype=torch.float32)
        tt_aux, xx_aux = torch.meshgrid(t, x_aux, indexing="ij")
        input_norm = self._input_normalization_spec(training_metadata)
        if input_norm["mode"] == "unit_box":
            x_min = float(input_norm["x_range"][0])
            t_min = float(input_norm["t_range"][0])
            x_scale = float(input_norm["x_scale"])
            t_scale = float(input_norm["t_scale"])
            xx_aux_net = (xx_aux - x_min) / x_scale
            tt_aux_net = (tt_aux - t_min) / t_scale
        else:
            x_scale = 1.0
            xx_aux_net = xx_aux
            tt_aux_net = tt_aux
        database_aux = torch.stack((xx_aux_net.reshape(-1), tt_aux_net.reshape(-1)), dim=1).requires_grad_(True)
        H_aux = self._apply_output_activation(torch, net(database_aux), training_metadata)
        if output_norm is not None and output_norm.get("mode") == "unit_interval":
            H_aux = H_aux * float(output_norm["scale"]) + float(output_norm["shift"])
        nt = int(t.numel())
        nx_aux = int(x_aux.numel())
        H_aux_np = self._tensor_grid(H_aux, nt, nx_aux)
        x_aux_np = x_aux.detach().cpu().numpy().astype(float)
        x_np = x.detach().cpu().numpy().astype(float)
        if use_fourier:
            H_operator_np = H_aux_np
            Hxx_aux_np = np.zeros_like(H_aux_np)
            Hx_aux_np = np.zeros_like(H_aux_np)
        else:
            grad_aux = torch.autograd.grad(H_aux.sum(), database_aux, create_graph=True)[0]
            Hx_aux = grad_aux[:, 0:1]
            Hxx_aux = torch.autograd.grad(Hx_aux.sum(), database_aux, create_graph=True)[0][:, 0:1]
            x_scale_safe = float(max(x_scale, np.finfo(float).eps))
            Hx_aux_np = self._tensor_grid(Hx_aux, nt, nx_aux) / x_scale_safe
            Hxx_aux_np = self._tensor_grid(Hxx_aux, nt, nx_aux) / (x_scale_safe**2)
            H_operator_np = homogenize_left_boundary(
                H_aux_np,
                x_aux_np,
                mode=self.config.spatial_boundary_mode,
                boundary_value=self.config.spatial_boundary_value,
                boundary_slope=Hx_aux_np[:, 0] if self.config.spatial_boundary_mode == "left_caputo" else None,
            )

        for beta0 in self.config.beta_reference_orders:
            beta0_float = float(beta0)
            if use_fourier:
                main_aux, correction_aux, is_integer = build_spatial_fractional_candidates(
                    H_operator_np,
                    dx=float(self.config.x_step),
                    beta_0=beta0_float,
                    spatial_axis=-1,
                )
                correction = np.empty_like(H_np)
                for row in range(correction_aux.shape[0]):
                    correction[row, :] = np.interp(x_np, x_aux_np, correction_aux[row, :])
                main = np.empty_like(H_np)
                for row in range(main_aux.shape[0]):
                    main[row, :] = np.interp(x_np, x_aux_np, main_aux[row, :])
                terms[f"spatial_main_beta0={beta0_float:.1f}"] = main
                terms[f"spatial_correction_beta0={beta0_float:.1f}"] = correction
                continue

            if np.isclose(beta0_float, 2.0):
                correction_aux = spatial_correction_column(
                    u_data=H_operator_np,
                    x_grid=x_aux_np,
                    dx=float(self.config.x_step),
                    beta_0=beta0_float,
                    n_quad=int(self.config.spatial_log_quad_points),
                    u_xx_values=Hxx_aux_np,
                    method=self.config.spatial_log_quad_method,
                    correction_method=self.config.spatial_fractional_mode,
                )
                correction = np.empty_like(H_np)
                for row in range(correction_aux.shape[0]):
                    correction[row, :] = np.interp(x_np, x_aux_np, correction_aux[row, :])
                terms[f"spatial_correction_beta0={beta0_float:.1f}"] = -correction
                continue

            main_aux, correction_aux = spatial_order_derivative_beta(
                H_operator_np,
                x_aux_np,
                beta_0=beta0_float,
                n_quad=int(self.config.spatial_log_quad_points),
                method=self.config.spatial_fractional_mode,
                u_xx_values=None,
            )
            main = np.empty_like(H_np)
            correction = np.empty_like(H_np)
            for row in range(main_aux.shape[0]):
                main[row, :] = np.interp(x_np, x_aux_np, main_aux[row, :])
                correction[row, :] = np.interp(x_np, x_aux_np, correction_aux[row, :])
            terms[f"spatial_main_beta0={beta0_float:.1f}"] = main
            terms[f"spatial_correction_beta0={beta0_float:.1f}"] = -correction
        return terms

    def _prune_small_beta_corrections(
        self,
        term_names: tuple[str, ...],
        coefficients: np.ndarray,
    ) -> None:
        """Zero spatial-correction columns whose implied |delta_beta| is below the prune threshold."""

        threshold = float(self.config.beta_correction_tol)
        if threshold <= 0.0:
            return
        name_to_index = {name: idx for idx, name in enumerate(term_names)}
        for name, idx in name_to_index.items():
            if not name.startswith("spatial_correction_beta0="):
                continue
            correction_coef = float(coefficients[idx, 0])
            if correction_coef == 0.0:
                continue
            beta0_token = name.split("=", 1)[1]
            try:
                beta0_value = float(beta0_token)
            except ValueError:
                continue
            main_name = f"spatial_main_beta0={beta0_token}"
            main_idx = name_to_index.get(main_name)
            if main_idx is None:
                coefficients[idx, 0] = 0.0
                continue
            main_coef = float(coefficients[main_idx, 0])
            if main_coef == 0.0:
                coefficients[idx, 0] = 0.0
                continue
            delta_beta = correction_coef / main_coef
            if abs(delta_beta) < threshold:
                coefficients[idx, 0] = 0.0

    def _spatial_beta_metadata_defaults(self) -> dict[str, Any]:
        return {
            "spatial_fractional_enabled": self.config.enable_spatial_fractional,
            "spatial_beta_bounds": self.config.beta_bounds,
            "spatial_beta_reference_orders": self.config.beta_reference_orders,
            "spatial_log_quad_points": self.config.spatial_log_quad_points,
            "spatial_log_quad_method": self.config.spatial_log_quad_method,
            "spatial_correction_method": self.config.spatial_correction_method,
            "spatial_fractional_mode": self.config.spatial_fractional_mode,
            "space_derivative_mode": self.config.space_derivative_mode,
            "time_operator_mode": self.config.time_operator_mode,
            "spatial_boundary_mode": self.config.spatial_boundary_mode,
            "spatial_boundary_value": self.config.spatial_boundary_value,
            "spatial_operator_note": (
                "fourier_taylor follows the periodic Fourier multiplier; "
                "gj_richardson and gl_pycaputo are left-sided RL diagnostics for noisy surrogates"
            ),
            "alpha_correction_tol": self.config.alpha_correction_tol,
            "beta_correction_tol": self.config.beta_correction_tol,
            "spatial_beta_status": "disabled",
            "spatial_beta": None,
            "spatial_delta_beta": None,
            "spatial_beta_correction": None,
            "spatial_beta_in_bounds": None,
            "spatial_correction_active": False,
            "spatial_beta_model_valid": True,
            "spatial_beta_candidates": (),
        }

    def _spatial_beta_metadata(self, term_names: tuple[str, ...], coefficients: np.ndarray) -> dict[str, Any]:
        metadata: dict[str, Any] = self._spatial_beta_metadata_defaults()
        if not self.config.enable_spatial_fractional:
            return metadata
        coef_map = dict(zip(term_names, coefficients))
        candidates: list[dict[str, Any]] = []
        for beta0 in self.config.beta_reference_orders:
            beta0_float = float(beta0)
            correction_name = f"spatial_correction_beta0={beta0_float:.1f}"
            correction_coef = float(coef_map.get(correction_name, 0.0))
            main_name = f"spatial_main_beta0={beta0_float:.1f}"
            main_coef = float(coef_map.get(main_name, 0.0))
            main_active = not np.isclose(main_coef, 0.0)
            correction_active = not np.isclose(correction_coef, 0.0)
            active = correction_active and main_active
            entry: dict[str, Any] = {
                "beta0": beta0_float,
                "main_term": main_name,
                "main_coefficient": main_coef,
                "correction_term": correction_name,
                "correction_coefficient": correction_coef,
                "active": active,
                "main_active": main_active,
                "correction_active": correction_active,
                "delta_beta": None,
                "beta_correction": None,
                "beta": None,
                "in_bounds": None,
                "correction_in_bounds": None,
            }
            if main_active and not correction_active:
                beta = beta0_float
                beta_correction = 0.0
                entry["delta_beta"] = 0.0
                entry["beta_correction"] = beta_correction
                entry["beta"] = beta
                entry["in_bounds"] = bool(self.config.beta_bounds[0] <= beta <= self.config.beta_bounds[1])
                entry["correction_in_bounds"] = True
            if active:
                delta_beta = correction_coef / main_coef
                beta = beta0_float - delta_beta
                beta_correction = delta_beta
                entry["delta_beta"] = float(delta_beta)
                entry["beta_correction"] = float(beta_correction)
                entry["beta"] = float(beta)
                entry["in_bounds"] = bool(self.config.beta_bounds[0] <= beta <= self.config.beta_bounds[1])
                entry["correction_in_bounds"] = True
            candidates.append(entry)

        metadata["spatial_beta_candidates"] = candidates
        active_candidates = [item for item in candidates if item["active"]]
        main_only_candidates = [
            item
            for item in candidates
            if item["main_active"]
            and not item["correction_active"]
            and item["in_bounds"]
        ]
        invalid_main_only_candidates = [
            item
            for item in candidates
            if item["main_active"]
            and not item["correction_active"]
            and not item["in_bounds"]
        ]
        metadata["spatial_correction_active"] = bool(active_candidates)
        if not active_candidates:
            if main_only_candidates:
                chosen = min(
                    main_only_candidates,
                    key=lambda item: abs(float(item["beta_correction"]))
                    if item["beta_correction"] is not None
                    else float("inf"),
                )
                metadata["spatial_beta_status"] = "correction_inactive"
                metadata["spatial_beta"] = chosen["beta"]
                metadata["spatial_delta_beta"] = chosen["delta_beta"]
                metadata["spatial_beta_correction"] = chosen["beta_correction"]
                metadata["spatial_beta_in_bounds"] = chosen["in_bounds"]
                metadata["spatial_beta0"] = chosen["beta0"]
                metadata["spatial_main_term"] = chosen["main_term"]
                metadata["spatial_main_coefficient"] = chosen["main_coefficient"]
                metadata["spatial_correction_term"] = chosen["correction_term"]
                metadata["spatial_correction_coefficient"] = chosen["correction_coefficient"]
            elif invalid_main_only_candidates:
                metadata["spatial_beta_model_valid"] = False
                metadata["spatial_beta_status"] = "beta_out_of_bounds"
            else:
                metadata["spatial_beta_status"] = "correction_inactive"
            return metadata
        invalid_corrections = [
            item for item in active_candidates if not item["in_bounds"]
        ]
        if invalid_corrections:
            metadata["spatial_beta_model_valid"] = False
            metadata["spatial_beta_status"] = "correction_out_of_bounds"
        in_bounds_candidates = [
            item for item in active_candidates if item["in_bounds"]
        ]
        chosen = min(
            in_bounds_candidates or active_candidates,
            key=lambda item: abs(float(item["beta_correction"])) if item["beta_correction"] is not None else float("inf"),
        )
        if metadata["spatial_beta_status"] != "correction_out_of_bounds":
            metadata["spatial_beta_status"] = "estimated"
        metadata["spatial_beta"] = chosen["beta"]
        metadata["spatial_delta_beta"] = chosen["delta_beta"]
        metadata["spatial_beta_correction"] = chosen["beta_correction"]
        metadata["spatial_beta_in_bounds"] = chosen["in_bounds"]
        metadata["spatial_beta0"] = chosen["beta0"]
        metadata["spatial_main_term"] = chosen["main_term"]
        metadata["spatial_main_coefficient"] = chosen["main_coefficient"]
        metadata["spatial_correction_term"] = chosen["correction_term"]
        metadata["spatial_correction_coefficient"] = chosen["correction_coefficient"]
        return metadata

    @staticmethod
    def _order_token(value: float) -> str:
        token = f"{float(value):.8g}"
        if "e" not in token and "." not in token:
            token += ".0"
        return token

    @staticmethod
    def _sample_time(values: np.ndarray, tau: np.ndarray, query_tau: np.ndarray) -> np.ndarray:
        flat_query = query_tau.reshape(-1)
        sampled = np.empty((flat_query.size, values.shape[1]), dtype=float)
        for j in range(values.shape[1]):
            sampled[:, j] = np.interp(flat_query, tau, values[:, j])
        return sampled.reshape(query_tau.shape[0], query_tau.shape[1], values.shape[1])

    def _fit_fde_stridge(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        protected_indices: tuple[int, ...] = (),
        term_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, float, float, float, dict[str, float | int | str]]:
        rng = np.random.default_rng(self.config.random_seed)
        n_rows = matrix.shape[0]
        if term_names is None:
            term_names = tuple(str(index) for index in range(matrix.shape[1]))
        if len(term_names) != matrix.shape[1]:
            raise ValueError("term_names length must match matrix columns")
        train_count = int(n_rows * self.config.split)
        train = rng.choice(n_rows, train_count, replace=False)
        test_mask = np.ones(n_rows, dtype=bool)
        test_mask[train] = False
        test = np.flatnonzero(test_mask)
        train_matrix = matrix[train, :]
        test_matrix = matrix[test, :]
        train_target = target[train, :]
        test_target = target[test, :]
        # Revision diagnostics expose the paper-result behavior and two
        # alternative data-use conventions without changing the default.
        # In train_only mode, the reserved 20% subset cannot affect threshold,
        # support, coefficient, or order selection.
        selection_mode = os.environ.get("GJ_STRIDGE_SELECTION_MODE")
        if selection_mode is None:
            selection_mode = (
                "train_validation"
                if os.environ.get("GJ_STRIDGE_FIT_ON_TRAIN", "0") == "1"
                else "legacy_full_validation"
            )
        valid_selection_modes = {
            "legacy_full_validation",
            "train_validation",
            "train_only",
        }
        if selection_mode not in valid_selection_modes:
            raise ValueError(
                "GJ_STRIDGE_SELECTION_MODE must be one of "
                f"{sorted(valid_selection_modes)}, got {selection_mode!r}"
            )
        if selection_mode == "legacy_full_validation":
            fit_matrix = matrix
            fit_target = target
            score_matrix = test_matrix
            score_target = test_target
            selection_residual_role = "reserved_subset_after_full_fit"
        elif selection_mode == "train_validation":
            fit_matrix = train_matrix
            fit_target = train_target
            score_matrix = test_matrix
            score_target = test_target
            selection_residual_role = "validation_subset"
        else:
            fit_matrix = train_matrix
            fit_target = train_target
            score_matrix = train_matrix
            score_target = train_target
            selection_residual_role = "structure_training_subset"
        condition_matrix = train_matrix if selection_mode == "train_only" else matrix
        condition = self._finite_condition(condition_matrix)
        l0_penalty = self.config.sparsity_lamb * condition

        best_tol = float(self.config.d_tol)
        d_tol = float(self.config.d_tol)
        tol = float(d_tol)
        best_weights = self._stridge(
            fit_matrix,
            fit_target,
            self.config.ridge_lambda,
            self.config.str_iters,
            tol,
            normalize=self.config.normalize,
            protected_indices=protected_indices,
            term_names=term_names,
            correction_tol_scale=self.config.fractional_correction_tol_scale,
        )
        best_residual_norm = self._norm2(score_target - score_matrix @ best_weights)
        protected_set = set(protected_indices)
        candidate_indices = tuple(index for index in range(matrix.shape[1]) if index not in protected_set)
        best_active_count = self._active_count(best_weights, candidate_indices)
        best_complexity_penalty = self._complexity_penalty(best_weights, term_names, candidate_indices, condition)
        best_active_lamb_sum = self._active_lamb_sum(best_weights, term_names, candidate_indices)
        best_error = float(best_residual_norm + best_complexity_penalty)

        for iteration in range(self.config.maxit):
            weights = self._stridge(
                fit_matrix,
                fit_target,
                self.config.ridge_lambda,
                self.config.str_iters,
                tol,
                normalize=self.config.normalize,
                protected_indices=protected_indices,
                term_names=term_names,
                correction_tol_scale=self.config.fractional_correction_tol_scale,
            )
            residual_norm = self._norm2(score_target - score_matrix @ weights)
            active_count = self._active_count(weights, candidate_indices)
            complexity_penalty = self._complexity_penalty(weights, term_names, candidate_indices, condition)
            active_lamb_sum = self._active_lamb_sum(weights, term_names, candidate_indices)
            error = float(residual_norm + complexity_penalty)
            if error <= best_error:
                best_error = error
                best_weights = weights
                best_residual_norm = residual_norm
                best_active_count = active_count
                best_complexity_penalty = complexity_penalty
                best_active_lamb_sum = active_lamb_sum
                best_tol = tol
                tol += d_tol
            else:
                tol = max(0.0, tol - 2.0 * d_tol)
                d_tol = 2.0 * d_tol / max(1, self.config.maxit - iteration)
                tol += d_tol
        if selection_mode == "train_validation":
            # This diagnostic follows the conventional validation workflow:
            # choose the support on train/validation, then refit its coefficients
            # on the full fitting window. Train-only structure discovery does not
            # refit here because that would feed reserved rows into later order and
            # support updates.
            support = np.flatnonzero(np.abs(best_weights.reshape(-1)) > 1.0e-12)
            if support.size > 0:
                best_weights = best_weights.copy()
                best_weights[support] = self._lstsq(matrix[:, support], target)
                best_residual_norm = self._norm2(test_target - test_matrix @ best_weights)
                best_error = float(best_residual_norm + best_complexity_penalty)
        return (
            best_weights,
            best_error,
            best_tol,
            l0_penalty,
            {
                "validation_residual_norm": best_residual_norm,
                "active_count": best_active_count,
                "complexity_penalty": float(best_complexity_penalty),
                "active_lamb_sum": float(best_active_lamb_sum),
                "condition_number": float(condition),
                "stridge_selection_mode": selection_mode,
                "selection_residual_role": selection_residual_role,
                "selection_residual_norm": float(best_residual_norm),
            },
        )

    def _score_coefficients(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        coefficients: np.ndarray,
        l0_penalty: float,
        candidate_indices: tuple[int, ...],
        term_names: tuple[str, ...] | None = None,
    ) -> dict[str, float | int]:
        rng = np.random.default_rng(self.config.random_seed)
        n_rows = matrix.shape[0]
        if term_names is None:
            term_names = tuple(str(index) for index in range(matrix.shape[1]))
        if len(term_names) != matrix.shape[1]:
            raise ValueError("term_names length must match matrix columns")
        train_count = int(n_rows * self.config.split)
        train = rng.choice(n_rows, train_count, replace=False)
        test_mask = np.ones(n_rows, dtype=bool)
        test_mask[train] = False
        test = np.flatnonzero(test_mask)
        validation_residual_norm = self._norm2(target[test, :] - matrix[test, :] @ coefficients)
        active_count = self._active_count(coefficients, candidate_indices)
        condition = self._condition_from_l0_penalty(l0_penalty, matrix)
        return {
            "validation_residual_norm": validation_residual_norm,
            "active_count": active_count,
            "complexity_penalty": float(
                self._complexity_penalty(coefficients, term_names, candidate_indices, condition)
            ),
        }

    def _complexity_penalty(
        self,
        weights: np.ndarray,
        term_names: tuple[str, ...],
        candidate_indices: tuple[int, ...],
        condition: float,
    ) -> float:
        return float(condition * self._active_lamb_sum(weights, term_names, candidate_indices))

    def _active_lamb_sum(
        self,
        weights: np.ndarray,
        term_names: tuple[str, ...],
        candidate_indices: tuple[int, ...],
    ) -> float:
        flat_weights = weights.reshape(-1)
        active_lamb_sum = 0.0
        for index in candidate_indices:
            if abs(float(flat_weights[index])) <= 1.0e-12:
                continue
            active_lamb_sum += self._sparsity_lamb_for_term(term_names[index])
        return float(active_lamb_sum)

    def _sparsity_lamb_for_term(self, term_name: str) -> float:
        if self._is_fractional_correction_term(term_name):
            return self._fractional_correction_sparsity_lamb()
        return float(self.config.sparsity_lamb)

    def _fractional_correction_sparsity_lamb(self) -> float:
        if self.config.fractional_correction_sparsity_lamb is None:
            return float(self.config.sparsity_lamb)
        return float(self.config.fractional_correction_sparsity_lamb)

    @staticmethod
    def _is_fractional_correction_term(term_name: str) -> bool:
        return term_name == "alpha_correction" or term_name.startswith("spatial_correction_beta0=")

    def _condition_from_l0_penalty(self, l0_penalty: float, matrix: np.ndarray) -> float:
        if self.config.sparsity_lamb != 0.0:
            return float(l0_penalty / self.config.sparsity_lamb)
        return self._finite_condition(matrix)

    @staticmethod
    def _stridge(
        matrix0: np.ndarray,
        target: np.ndarray,
        ridge_lambda: float,
        maxit: int,
        tol: float,
        normalize: int = 2,
        protected_indices: tuple[int, ...] = (),
        term_names: tuple[str, ...] | None = None,
        correction_tol_scale: float = 1.0,
    ) -> np.ndarray:
        n_rows, n_cols = matrix0.shape
        if normalize != 0:
            matrix = np.zeros((n_rows, n_cols), dtype=float)
            scales = np.zeros((n_cols, 1), dtype=float)
            for index in range(n_cols):
                norm = FractionalPDEDiscoverer._vector_norm(matrix0[:, index], normalize)
                scales[index] = 1.0 / norm if norm > 0.0 else 1.0
                matrix[:, index] = scales[index, 0] * matrix0[:, index]
        else:
            matrix = matrix0
            scales = np.ones((n_cols, 1), dtype=float)

        if ridge_lambda != 0.0:
            weights = FractionalPDEDiscoverer._ridge_lstsq(matrix, target, ridge_lambda)
        else:
            weights = FractionalPDEDiscoverer._lstsq(matrix, target)

        protected_set = set(protected_indices)
        active_count = n_cols - len(protected_set)
        biginds = np.where(np.abs(weights.reshape(-1)) > tol)[0]
        for j in range(maxit):
            flat_weights = np.abs(weights.reshape(-1))
            column_tols = np.full(n_cols, float(tol), dtype=float)
            if term_names is not None:
                scale = max(float(correction_tol_scale), 0.0)
                for index, name in enumerate(term_names):
                    if FractionalPDEDiscoverer._is_fractional_correction_term(name):
                        column_tols[index] = 0.0 if scale <= 0.0 else float(tol) * scale
            smallinds = np.array(
                [index for index in np.where(flat_weights < column_tols)[0] if index not in protected_set],
                dtype=int,
            )
            new_biginds = np.array([i for i in range(n_cols) if i not in set(smallinds)], dtype=int)
            if active_count == len(new_biginds):
                break
            active_count = len(new_biginds)
            if len(new_biginds) == 0:
                if j == 0:
                    return weights * scales
                break
            biginds = new_biginds
            weights[smallinds] = 0.0
            if ridge_lambda != 0.0:
                weights[biginds] = FractionalPDEDiscoverer._ridge_lstsq(matrix[:, biginds], target, ridge_lambda)
            else:
                weights[biginds] = FractionalPDEDiscoverer._lstsq(matrix[:, biginds], target)
        if len(biginds) > 0:
            weights[biginds] = FractionalPDEDiscoverer._lstsq(matrix[:, biginds], target)
        return weights * scales

    @staticmethod
    def _active_count(weights: np.ndarray, candidate_indices: tuple[int, ...]) -> int:
        if not candidate_indices:
            return 0
        return int(np.count_nonzero(np.abs(weights[list(candidate_indices)]) > 1.0e-12))

    @staticmethod
    def _finite_condition(matrix: np.ndarray) -> float:
        torch = sys.modules.get("torch")
        if torch is not None:
            tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float64)
            condition = float(torch.linalg.cond(tensor).detach().cpu().item())
        else:
            condition = float(np.linalg.cond(matrix))
        if np.isfinite(condition):
            return condition
        return 1.0e12

    @staticmethod
    def _lstsq(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
        torch = sys.modules.get("torch")
        if torch is not None:
            matrix_tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float64)
            target_tensor = torch.as_tensor(np.asarray(target), dtype=torch.float64)
            solution = torch.linalg.lstsq(matrix_tensor, target_tensor).solution
            return solution.detach().cpu().numpy()
        return np.linalg.lstsq(matrix, target, rcond=None)[0]

    @staticmethod
    def _ridge_lstsq(matrix: np.ndarray, target: np.ndarray, ridge_lambda: float) -> np.ndarray:
        torch = sys.modules.get("torch")
        if torch is not None:
            matrix_tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float64)
            target_tensor = torch.as_tensor(np.asarray(target), dtype=torch.float64)
            lhs = matrix_tensor.T @ matrix_tensor
            lhs = lhs + float(ridge_lambda) * torch.eye(lhs.shape[0], dtype=lhs.dtype, device=lhs.device)
            rhs = matrix_tensor.T @ target_tensor
            solution = torch.linalg.solve(lhs, rhs)
            return solution.detach().cpu().numpy()
        return np.linalg.lstsq(
            matrix.T @ matrix + ridge_lambda * np.eye(matrix.shape[1]),
            matrix.T @ target,
            rcond=None,
        )[0]

    @staticmethod
    def _norm2(values: np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        return float(np.sqrt(np.sum(array * array)))

    @staticmethod
    def _vector_norm(values: np.ndarray, order: int | float) -> float:
        array = np.asarray(values, dtype=float)
        if order == 1:
            return float(np.sum(np.abs(array)))
        if order == 2:
            return float(np.sqrt(np.sum(array * array)))
        if order == np.inf:
            return float(np.max(np.abs(array)))
        return float(np.sum(np.abs(array) ** order) ** (1.0 / order))

    @staticmethod
    def _tensor_grid(tensor: Any, nt: int, nx: int) -> np.ndarray:
        return tensor.detach().cpu().numpy().reshape(nt, nx).astype(float)

    @staticmethod
    def _assert_finite(label: str, arrays: Any) -> None:
        for array in arrays:
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"{label} generated NaN or Inf values")

    def _torch(self) -> Any:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - exercised only when torch is unavailable
            raise RuntimeError("Torch is required for the FDE NN reconstruction; use the sr environment.") from exc
        # Reproducibility: pin torch to single-threaded execution so the NN-derived
        # candidate columns are bitwise-stable run-to-run. The OMP_NUM_THREADS env var
        # does not reliably constrain torch on Windows, and multithreaded torch reductions
        # perturb near-degenerate STRidge selections (notably the low-lambda clean tsfade
        # case). Thread count only -- do not enable use_deterministic_algorithms or reseed,
        # which switch autograd kernels and change the derivative values themselves.
        # Set GJ_TORCH_MULTITHREAD=1 to bypass the pin (reproducibility diagnostic
        # only; results are then not bitwise stable).
        if os.environ.get("GJ_TORCH_MULTITHREAD", "0") != "1":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        return torch

    def _build_network(self, torch: Any, *, activation: str, hidden_layers: int, neurons: int) -> Any:
        from torch import nn

        class Sin(nn.Module):
            def forward(self, x: Any) -> Any:
                return torch.sin(x)

        class Gaussian(nn.Module):
            def __init__(self, scale: float = 0.5):
                super().__init__()
                self.scale = float(scale)

            def forward(self, x: Any) -> Any:
                return torch.exp(-0.5 * (self.scale * x) ** 2)

        activation_key = activation.lower()
        if activation_key == "tanh":
            activation_factory = nn.Tanh
        elif activation_key == "sin":
            activation_factory = Sin
        elif activation_key.startswith("gaussian"):
            if ":" in activation_key:
                scale = float(activation_key.split(":", 1)[1])
            else:
                scale = 0.5
            activation_factory = lambda: Gaussian(scale)
        else:
            raise NotImplementedError(f"Unsupported activation: {activation}")

        layers: list[Any] = [nn.Linear(2, neurons), activation_factory()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), activation_factory()])
        layers.append(nn.Linear(neurons, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _load_training_metadata(checkpoint_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        config_path = checkpoint_path.parent / "config.json"
        summary_path = checkpoint_path.parent / "training_summary.txt"
        if config_path.exists():
            try:
                metadata["config"] = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                metadata["config"] = str(config_path)
        if summary_path.exists():
            try:
                metadata["summary"] = summary_path.read_text(encoding="utf-8")
            except Exception:
                metadata["summary"] = str(summary_path)
        return metadata

    @staticmethod
    def _format_training_best(metadata: dict[str, Any]) -> str:
        config = metadata.get("config")
        if not isinstance(config, dict):
            return "unknown"
        best_step = config.get("best_step", "unknown")
        best_val = config.get("best_val_loss", "unknown")
        return f"step={best_step}, val_loss={best_val}"

    def _network_spec(self, metadata: dict[str, Any]) -> dict[str, Any]:
        config = metadata.get("config")
        if isinstance(config, dict):
            return {
                "activation": config.get("activation", self.config.activation),
                "hidden_layers": config.get("hidden_layers", self.config.hidden_layers),
                "neurons": config.get("neurons", self.config.neurons),
            }
        return {
            "activation": self.config.activation,
            "hidden_layers": self.config.hidden_layers,
            "neurons": self.config.neurons,
        }

    def _input_normalization_spec(self, metadata: dict[str, Any]) -> dict[str, Any]:
        config = metadata.get("config")
        if isinstance(config, dict):
            mode = str(config.get("input_normalization", "none"))
            x_range = config.get("input_x_range")
            t_range = config.get("input_t_range")
            x_scale = config.get("input_x_scale")
            t_scale = config.get("input_t_scale")
            if (
                mode == "unit_box"
                and isinstance(x_range, (list, tuple))
                and len(x_range) == 2
                and isinstance(t_range, (list, tuple))
                and len(t_range) == 2
            ):
                x0, x1 = float(x_range[0]), float(x_range[1])
                t0, t1 = float(t_range[0]), float(t_range[1])
                x_scale_val = float(x_scale) if x_scale is not None else float(max(x1 - x0, np.finfo(float).eps))
                t_scale_val = float(t_scale) if t_scale is not None else float(max(t1 - t0, np.finfo(float).eps))
                return {
                    "mode": "unit_box",
                    "x_range": (x0, x1),
                    "t_range": (t0, t1),
                    "x_scale": x_scale_val,
                    "t_scale": t_scale_val,
                }
        return {
            "mode": "none",
            "x_range": (0.0, 1.0),
            "t_range": (0.0, 1.0),
            "x_scale": 1.0,
            "t_scale": 1.0,
        }

    @staticmethod
    def _output_activation_spec(metadata: dict[str, Any]) -> str:
        config = metadata.get("config")
        if isinstance(config, dict):
            activation = str(config.get("output_activation", "identity")).lower()
            if activation in {"identity", "softplus"}:
                return activation
        return "identity"

    def _apply_output_activation(self, torch: Any, values: Any, metadata: dict[str, Any]) -> Any:
        activation = self._output_activation_spec(metadata)
        if activation == "identity":
            return values
        if activation == "softplus":
            return torch.nn.functional.softplus(values)
        raise ValueError(f"Unsupported output activation: {activation}")

    def _output_normalization_spec(self, metadata: dict[str, Any]) -> dict[str, Any]:
        config = metadata.get("config")
        if isinstance(config, dict):
            mode = str(config.get("output_normalization", "none"))
            if mode == "unit_interval":
                shift = float(config.get("target_c_shift", 0.0))
                scale = float(config.get("target_c_scale", 1.0))
                return {
                    "mode": "unit_interval",
                    "shift": shift,
                    "scale": float(max(scale, np.finfo(float).eps)),
                }
        return {
            "mode": "none",
            "shift": 0.0,
            "scale": 1.0,
        }

    def _fourier_spatial_bounds(self, metadata: dict[str, Any]) -> tuple[float, float]:
        config = metadata.get("config")
        if isinstance(config, dict):
            x_range = config.get("x_range")
            if (
                isinstance(x_range, (list, tuple))
                and len(x_range) == 2
                and x_range[1] > x_range[0]
            ):
                return float(x_range[0]), float(x_range[1])
        return float(self.config.spatial_fractional_lower_bound), float(self.config.x_max)
