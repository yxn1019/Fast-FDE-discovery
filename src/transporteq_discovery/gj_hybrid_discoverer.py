"""G-J operator discovery with Taylor-style order linearization for tsfade_fft."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special as sp
from scipy.optimize import least_squares

from transporteq_discovery.fractional_discoverer import (
    AlphaScanResult,
    FractionalDiscoveryConfig,
    FractionalDiscoveryResult,
    FractionalPDEDiscoverer,
)
from transporteq_discovery.eqgpt_adapter import EqGPTGenerator, EqGPTSample
from transporteq_discovery.generated_structures import FRACTIONAL_TOKEN, GeneratedStructure
from transporteq_discovery.models import SparseModel


@dataclass(frozen=True)
class GJBaseField:
    time: np.ndarray
    position: np.ndarray
    H: np.ndarray
    Hx: np.ndarray
    Hxx: np.ndarray
    Hxxx: np.ndarray
    database: np.ndarray
    checkpoint_path: str
    training_metadata: dict[str, Any]


class GJHybridDiscoverer:
    """Discover tsfade_fft with G-J operators and local order linearization."""

    def __init__(self, config: FractionalDiscoveryConfig):
        self.config = config
        self._fde = FractionalPDEDiscoverer(config)

    def discover(self) -> FractionalDiscoveryResult:
        started = time.perf_counter()
        torch, net, training_metadata = self._load_network()
        operator_field = self._build_base_field(torch, net, training_metadata)
        fit_field = self._fit_window_field(operator_field)
        if self.config.order_update_mode == "iterative":
            if self.config.candidate_search == "generated":
                alpha_scan = self._iterate_generated_order_updates(torch, net, operator_field, fit_field)
            else:
                alpha_scan = []
                for beta_start in self._iter_start_beta_values():
                    alpha_scan.extend(
                        self._iterate_order_updates(
                            torch,
                            net,
                            operator_field,
                            fit_field,
                            iter_start_beta_override=beta_start,
                        )
                    )
            best = self._select_iterative_final(alpha_scan)
        elif self.config.order_update_mode == "grid":
            alpha_scan = self._scan_candidates(torch, net, operator_field, fit_field)
            best = self._select_best(alpha_scan)
        else:
            raise ValueError("order_update_mode must be 'grid' or 'iterative'")
        selected_model = best.model
        if self.config.refit_mode == "nonlinear":
            selected_model = self._nonlinear_refit_model(torch, net, operator_field, fit_field, best.model)
        elapsed = time.perf_counter() - started
        final_metadata = dict(selected_model.metadata)
        generated_summaries = self._generated_candidate_summaries(alpha_scan)
        if generated_summaries:
            final_metadata["generated_candidate_summaries"] = tuple(generated_summaries)
        final_metadata["end_to_end_seconds"] = elapsed
        final_model = SparseModel(
            target_name=selected_model.target_name,
            term_names=selected_model.term_names,
            coefficients=selected_model.coefficients.copy(),
            tolerance=selected_model.tolerance,
            ridge_lambda=selected_model.ridge_lambda,
            mse=selected_model.mse,
            residual_norm=selected_model.residual_norm,
            information_criterion=selected_model.information_criterion,
            support_frequency=selected_model.support_frequency.copy(),
            sample_count=selected_model.sample_count,
            metadata=final_metadata,
        )
        initial = next(
            (item.model for item in alpha_scan if np.isclose(item.alpha0, 0.8) and np.isclose(item.model.metadata.get("beta0"), 1.8)),
            alpha_scan[0].model,
        )
        return FractionalDiscoveryResult(
            alpha=float(final_metadata["alpha"]),
            model=final_model,
            initial_model=initial,
            alpha_scan=tuple(alpha_scan),
            used_remainder_terms=("alpha_correction",) if "alpha_correction" in best.model.metadata.get("internal_augmented_terms", ()) else (),
            ga_triggered=False,
        )

    def _fit_window_field(self, field: GJBaseField) -> GJBaseField:
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
        if not np.any(x_mask) or not np.any(t_mask):
            raise ValueError("Configured fit window is empty for the G-J operator grid.")
        nt, nx = field.H.shape
        database_grid = field.database.reshape(nt, nx, -1)
        return GJBaseField(
            time=field.time[t_mask].copy(),
            position=field.position[x_mask].copy(),
            H=field.H[np.ix_(t_mask, x_mask)].copy(),
            Hx=field.Hx[np.ix_(t_mask, x_mask)].copy(),
            Hxx=field.Hxx[np.ix_(t_mask, x_mask)].copy(),
            Hxxx=field.Hxxx[np.ix_(t_mask, x_mask)].copy(),
            database=database_grid[np.ix_(t_mask, x_mask)].reshape(-1, database_grid.shape[-1]).copy(),
            checkpoint_path=field.checkpoint_path,
            training_metadata=field.training_metadata,
        )

    @staticmethod
    def _restrict_values(values: np.ndarray, source: GJBaseField, target: GJBaseField) -> np.ndarray:
        t_indices = [int(np.argmin(np.abs(source.time - time))) for time in target.time]
        x_indices = [int(np.argmin(np.abs(source.position - position))) for position in target.position]
        if not np.allclose(source.time[t_indices], target.time) or not np.allclose(
            source.position[x_indices], target.position
        ):
            raise ValueError("target fit window is not aligned with the G-J operator grid")
        return np.asarray(values)[np.ix_(t_indices, x_indices)]

    def _load_network(self) -> tuple[Any, Any, dict[str, Any]]:
        torch = self._fde._torch()
        checkpoint_path = self.config.checkpoint_path
        metadata = self._fde._load_training_metadata(checkpoint_path)
        network_spec = self._fde._network_spec(metadata)
        net = self._fde._build_network(
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
        return torch, net, metadata

    def _build_base_field(self, torch: Any, net: Any, training_metadata: dict[str, Any]) -> GJBaseField:
        input_norm = self._fde._input_normalization_spec(training_metadata)
        output_norm = self._fde._output_normalization_spec(training_metadata)
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
        prediction = self._fde._apply_output_activation(torch, net(database), training_metadata)
        if output_norm["mode"] == "unit_interval":
            prediction = prediction * float(output_norm["scale"]) + float(output_norm["shift"])
        grad = torch.autograd.grad(outputs=prediction.sum(), inputs=database, create_graph=True)[0]
        hx = grad[:, 0:1]
        hxx = torch.autograd.grad(outputs=hx.sum(), inputs=database, create_graph=True)[0][:, 0:1]
        hxxx = torch.autograd.grad(outputs=hxx.sum(), inputs=database, create_graph=True)[0][:, 0:1]
        nt = int(t.numel())
        nx = int(x.numel())
        return GJBaseField(
            time=t.detach().cpu().numpy().astype(float),
            position=x.detach().cpu().numpy().astype(float),
            H=prediction.detach().cpu().numpy().reshape(nt, nx),
            Hx=(hx.detach().cpu().numpy().reshape(nt, nx) / float(x_scale)).astype(float),
            Hxx=(hxx.detach().cpu().numpy().reshape(nt, nx) / float(x_scale**2)).astype(float),
            Hxxx=(hxxx.detach().cpu().numpy().reshape(nt, nx) / float(x_scale**3)).astype(float),
            database=database.detach().cpu().numpy(),
            checkpoint_path=str(self.config.checkpoint_path),
            training_metadata=training_metadata,
        )

    def _iterate_order_updates(
        self,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        *,
        generated_structure: GeneratedStructure | None = None,
        generated_structure_rank: int | None = None,
        generated_structure_count: int | None = None,
        alpha_cache: dict[float, np.ndarray] | None = None,
        beta_cache: dict[float, np.ndarray] | None = None,
        iter_start_beta_override: float | None = None,
    ) -> list[AlphaScanResult]:
        if self.config.candidate_search != "fixed" and generated_structure is None:
            raise ValueError("iterative generated search requires an explicit generated RHS structure")
        if alpha_cache is None:
            alpha_cache = {}
        if beta_cache is None:
            beta_cache = {}
        alpha = self._clip_iter_alpha(float(self.config.iter_start_alpha))
        beta_start = self.config.iter_start_beta if iter_start_beta_override is None else iter_start_beta_override
        beta = self._clip_iter_beta(float(beta_start))
        max_iters = max(1, int(self.config.iter_max_iters))
        tol = max(0.0, float(self.config.iter_order_tol))
        damping = float(self.config.iter_damping)
        max_step_alpha = abs(float(self.config.iter_max_step_alpha))
        max_step_beta = abs(float(self.config.iter_max_step_beta))
        results: list[AlphaScanResult] = []
        trace: list[dict[str, Any]] = []
        converged = False
        stop_reason = "max_iterations"
        best_iter_objective = float("inf")

        for iteration in range(max_iters):
            self._prepare_iter_alpha_cache(torch, net, operator_field, alpha_cache, alpha)
            self._prepare_iter_beta_cache(torch, net, operator_field, beta_cache, beta)
            result = self._candidate_result(
                torch,
                net,
                operator_field,
                fit_field,
                alpha_cache,
                beta_cache,
                alpha,
                beta,
                generated_structure=generated_structure,
                generated_structure_rank=generated_structure_rank,
                generated_structure_count=generated_structure_count,
                clip_order_corrections=True,
            )
            metadata = result.model.metadata
            current_objective = float(metadata.get("objective", result.objective))
            raw_delta_alpha = float(metadata.get("delta_alpha", result.delta_alpha) or 0.0)
            raw_delta_beta = float(metadata.get("spatial_delta_beta") or 0.0)
            unbounded_delta_alpha = float(metadata.get("delta_alpha_raw", raw_delta_alpha) or 0.0)
            unbounded_delta_beta = float(metadata.get("spatial_delta_beta_raw", raw_delta_beta) or 0.0)
            support = tuple(str(name) for name in result.model.support_names)
            has_fractional_support = any(name.startswith("D_x^") for name in support)
            beta_update_skipped_reason = None
            delta_beta_for_update = raw_delta_beta
            if not has_fractional_support:
                delta_beta_for_update = 0.0
                beta_update_skipped_reason = "no active spatial fractional support in this iteration"
            if not np.isfinite(raw_delta_alpha) or not np.isfinite(delta_beta_for_update):
                stop_reason = "nonfinite_correction"
                delta_alpha_step = 0.0
                delta_beta_step = 0.0
            else:
                delta_alpha_step = float(np.clip(damping * raw_delta_alpha, -max_step_alpha, max_step_alpha))
                delta_beta_step = float(np.clip(damping * delta_beta_for_update, -max_step_beta, max_step_beta))
            alpha_next = self._clip_iter_alpha(alpha + delta_alpha_step)
            beta_next = self._clip_iter_beta(beta + delta_beta_step)
            if not np.isfinite(raw_delta_alpha) or not np.isfinite(delta_beta_for_update):
                alpha_next = alpha
                beta_next = beta

            row = {
                "iter": iteration,
                "alpha0": float(alpha),
                "beta0": float(beta),
                "delta_alpha_raw": unbounded_delta_alpha,
                "delta_beta_raw": unbounded_delta_beta,
                "delta_alpha": raw_delta_alpha,
                "delta_beta": raw_delta_beta,
                "delta_beta_used": delta_beta_for_update,
                "alpha_step": delta_alpha_step,
                "beta_step": delta_beta_step,
                "alpha_next": float(alpha_next),
                "beta_next": float(beta_next),
                "support": "+".join(support) if support else "none",
                "objective": current_objective,
                "regularized_objective": current_objective,
                "raw_mse": float(result.model.mse),
                "mse": float(result.model.mse),
                "condition": float(metadata.get("physical_refit_condition", metadata.get("condition_number", np.nan))),
                "validation_residual_norm": float(metadata.get("validation_residual_norm", np.nan)),
                "complexity_penalty": float(metadata.get("complexity_penalty", np.nan)),
                "active_lamb_sum": float(metadata.get("active_lamb_sum", np.nan)),
                "physical_refit_condition_penalty": float(
                    metadata.get("physical_refit_selection_condition_penalty", np.nan)
                ),
                "active_count_for_loss": int(metadata.get("active_count_for_loss", 0)),
                "tolerance": float(metadata.get("tolerance", np.nan)),
                "spatial_beta_status": metadata.get("spatial_beta_status"),
                "beta_update_skipped_reason": beta_update_skipped_reason,
            }
            if generated_structure is not None:
                row.update(
                    {
                        "generated_structure_rank": generated_structure_rank,
                        "generated_structure": generated_structure.text,
                        "generated_raw_structure": generated_structure.raw_text,
                        "generated_prior_score": float(generated_structure.prior_score),
                        "generated_complexity_score": float(generated_structure.complexity_score),
                    }
                )
            trace.append(row)
            metadata.update(
                {
                    "order_update_mode": "iterative",
                    "iteration_index": iteration,
                    "iter_alpha0": float(alpha),
                    "iter_beta0": float(beta),
                    "iter_alpha_next": float(alpha_next),
                    "iter_beta_next": float(beta_next),
                    "iter_delta_alpha_raw": metadata.get("delta_alpha_raw", raw_delta_alpha),
                    "iter_delta_beta_raw": metadata.get("spatial_delta_beta_raw", raw_delta_beta),
                    "iter_delta_beta_used": delta_beta_for_update,
                    "iter_alpha_step": delta_alpha_step,
                    "iter_beta_step": delta_beta_step,
                    "iter_beta_update_skipped_reason": beta_update_skipped_reason,
                }
            )
            results.append(result)

            if stop_reason == "nonfinite_correction":
                break
            if (
                self.config.iter_stop_on_non_decreasing_objective
                and len(results) > 1
                and not current_objective < best_iter_objective - float(self.config.iter_objective_min_delta)
            ):
                stop_reason = "objective_not_decreased"
                break
            best_iter_objective = min(best_iter_objective, current_objective)
            if abs(raw_delta_alpha) <= tol and abs(delta_beta_for_update) <= tol:
                converged = True
                stop_reason = "correction_below_tolerance"
                break
            if np.isclose(alpha_next, alpha) and np.isclose(beta_next, beta):
                stop_reason = "clipped_update_stalled"
                break
            alpha = alpha_next
            beta = beta_next

        valid_indices = [idx for idx, item in enumerate(results) if self._fde._order_result_is_valid(item)]
        final_index = valid_indices[-1] if valid_indices else len(results) - 1
        best_pool = valid_indices or list(range(len(results)))
        best_index = min(
            best_pool,
            key=lambda idx: float(results[idx].model.metadata.get("objective", results[idx].objective)),
        )
        fractional_indices = [
            idx
            for idx in best_pool
            if any(str(name).startswith("D_x^") for name in results[idx].model.support_names)
        ]
        best_fractional_index = (
            min(
                fractional_indices,
                key=lambda idx: float(
                    results[idx].model.metadata.get("objective", results[idx].objective)
                ),
            )
            if fractional_indices
            else None
        )
        for item in results:
            item.model.metadata.update(
                {
                    "order_update_mode": "iterative",
                    "iteration_count": len(results),
                    "iteration_converged": converged,
                    "iteration_stop_reason": stop_reason,
                    "iteration_final_index": final_index,
                    "iteration_best_objective_index": best_index,
                    "iteration_best_fractional_support_index": best_fractional_index,
                    "iteration_trace": tuple(trace),
                    "iter_start_alpha": float(self.config.iter_start_alpha),
                    "iter_start_beta": float(beta_start),
                    "iter_start_beta_values": self.config.iter_start_beta_values,
                    "iter_order_tol": tol,
                    "iter_damping": damping,
                    "iter_max_step_alpha": max_step_alpha,
                    "iter_max_step_beta": max_step_beta,
                    "iter_delta_alpha_bounds": self.config.iter_delta_alpha_bounds,
                    "iter_delta_beta_bounds": self.config.iter_delta_beta_bounds,
                    "iter_stop_on_non_decreasing_objective": self.config.iter_stop_on_non_decreasing_objective,
                    "iter_objective_min_delta": self.config.iter_objective_min_delta,
                    "iter_selection_mode": self.config.iter_selection_mode,
                }
            )
        return results

    def _iterate_generated_order_updates(
        self,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
    ) -> list[AlphaScanResult]:
        if self.config.eqgpt_model_checkpoint is None:
            raise ValueError(
                "candidate_search=generated now uses the original EqGPT sampler; "
                "provide --eqgpt-model-checkpoint."
            )
        generator = EqGPTGenerator(
            checkpoint_path=self.config.eqgpt_model_checkpoint,
            dictionary_path=self.config.eqgpt_dictionary_path,
            seed=int(self.config.generated_seed),
            random_exploration_probability=float(self.config.eqgpt_random_exploration),
        )
        alpha_cache: dict[float, np.ndarray] = {}
        beta_cache: dict[float, np.ndarray] = {}
        results: list[AlphaScanResult] = []
        cycles = max(1, int(self.config.eqgpt_optimize_epochs))
        top_k = max(1, int(self.config.generated_top_structures))
        for epoch in range(cycles):
            samples = generator.sample_structures(
                count=int(self.config.generated_candidates),
                max_terms=int(self.config.generated_max_terms),
            )
            evaluated = self._evaluate_eqgpt_samples(
                samples=samples,
                epoch=epoch,
                torch=torch,
                net=net,
                operator_field=operator_field,
                fit_field=fit_field,
                alpha_cache=alpha_cache,
                beta_cache=beta_cache,
            )
            evaluated.sort(key=self._generated_structure_sort_key)
            for rank, item in enumerate(evaluated, start=1):
                item.model.metadata["generated_structure_rank"] = rank
                item.model.metadata["eqgpt_reward_rank"] = rank
                item.model.metadata["eqgpt_selection_mode"] = self.config.eqgpt_selection_mode
                item.model.metadata["generated_structure_count"] = len(evaluated)
            results.extend(evaluated)
            top_sequences = [
                tuple(item.model.metadata.get("eqgpt_token_ids", ()))
                for item in evaluated[:top_k]
                if item.model.metadata.get("eqgpt_token_ids")
            ]
            if epoch + 1 < cycles and top_sequences:
                generator.fine_tune(
                    top_sequences,
                    epochs=int(self.config.eqgpt_finetune_epochs),
                    learning_rate=float(self.config.eqgpt_learning_rate),
                )
        return results

    def _evaluate_eqgpt_samples(
        self,
        *,
        samples: tuple[EqGPTSample, ...],
        epoch: int,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        alpha_cache: dict[float, np.ndarray],
        beta_cache: dict[float, np.ndarray],
    ) -> list[AlphaScanResult]:
        unique: dict[tuple[str, ...], EqGPTSample] = {}
        for sample in samples:
            if sample.structure is None:
                continue
            unique.setdefault(sample.structure.terms, sample)
        evaluated: list[AlphaScanResult] = []
        for sample_index, sample in enumerate(unique.values(), start=1):
            structure = sample.structure
            if structure is None:
                continue
            structure_results: list[AlphaScanResult] = []
            for beta_start in self._iter_start_beta_values():
                structure_results.extend(
                    self._iterate_order_updates(
                        torch,
                        net,
                        operator_field,
                        fit_field,
                        generated_structure=structure,
                        generated_structure_rank=sample_index,
                        generated_structure_count=len(unique),
                        alpha_cache=alpha_cache,
                        beta_cache=beta_cache,
                        iter_start_beta_override=beta_start,
                    )
                )
            if not structure_results:
                continue
            best_for_reward = self._best_reward_candidate(structure_results)
            reward, r2 = self._eqgpt_reward(best_for_reward)
            for item in structure_results:
                item.model.metadata.update(
                    {
                        "eqgpt_checkpoint": str(self.config.eqgpt_model_checkpoint),
                        "eqgpt_dictionary": str(self.config.eqgpt_dictionary_path),
                        "eqgpt_epoch": int(epoch),
                        "eqgpt_sample_index": int(sample_index),
                        "eqgpt_sample_count": int(len(samples)),
                        "eqgpt_unique_structure_count": int(len(unique)),
                        "eqgpt_optimize_epochs": int(self.config.eqgpt_optimize_epochs),
                        "eqgpt_finetune_epochs": int(self.config.eqgpt_finetune_epochs),
                        "eqgpt_learning_rate": float(self.config.eqgpt_learning_rate),
                        "eqgpt_reward_sparsity_alpha": float(self.config.eqgpt_reward_sparsity_alpha),
                        "eqgpt_selection_mode": str(self.config.eqgpt_selection_mode),
                        "eqgpt_random_exploration": float(self.config.eqgpt_random_exploration),
                        "eqgpt_reward": float(reward),
                        "eqgpt_r2": float(r2),
                        "eqgpt_token_ids": tuple(int(token) for token in sample.token_ids),
                        "eqgpt_source_equation": sample.equation,
                    }
                )
            evaluated.append(best_for_reward)
        return evaluated

    def _generated_structure_sort_key(self, item: AlphaScanResult) -> tuple[float, float]:
        mode = str(self.config.eqgpt_selection_mode)
        objective = float(item.model.metadata.get("objective", item.objective))
        reward = float(item.model.metadata.get("eqgpt_reward", 0.0))
        if mode == "eqgpt_reward":
            return (-reward, objective)
        if mode == "stridge_objective":
            return (objective, -reward)
        raise ValueError("eqgpt_selection_mode must be 'stridge_objective' or 'eqgpt_reward'")

    def _best_reward_candidate(self, candidates: list[AlphaScanResult]) -> AlphaScanResult:
        valid = [item for item in candidates if self._fde._order_result_is_valid(item)] or candidates
        return min(valid, key=lambda item: float(item.model.metadata.get("objective", item.objective)))

    def _eqgpt_reward(self, item: AlphaScanResult) -> tuple[float, float]:
        r2 = float(item.model.metadata.get("target_r2", 0.0))
        if not np.isfinite(r2):
            r2 = 0.0
        active_count = int(item.model.metadata.get("active_physical_count", 0))
        active_count = max(1, active_count)
        reward = (1.0 - float(self.config.eqgpt_reward_sparsity_alpha) * np.log10(active_count)) * r2
        if not np.isfinite(reward):
            reward = 0.0
        return float(max(0.0, reward)), float(r2)

    def _scan_candidates(
        self,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
    ) -> list[AlphaScanResult]:
        alpha_cache: dict[float, np.ndarray] = {}
        beta_cache: dict[float, np.ndarray] = {}
        alpha0_grid = tuple(float(value) for value in self.config.alpha0_grid if 0.0 < float(value) < 1.0)
        alpha_points = set(alpha0_grid)
        for alpha0 in alpha0_grid:
            step = self._alpha_step(alpha0)
            alpha_points.add(round(alpha0 - step, 10))
            alpha_points.add(round(alpha0 + step, 10))
        for alpha in sorted(value for value in alpha_points if 0.0 < value < 1.0):
            alpha_cache[round(alpha, 10)] = self._compute_halpha(torch, net, operator_field, alpha)

        beta_points = set(float(value) for value in self.config.beta_reference_orders)
        for beta0 in self.config.beta_reference_orders:
            beta0 = float(beta0)
            step = self._beta_step(beta0)
            beta_points.add(round(beta0 - step, 10))
            if not np.isclose(beta0, 2.0):
                beta_points.add(round(beta0 + step, 10))
        for beta in sorted(value for value in beta_points if 1.0 < value <= 2.0):
            beta_cache[round(beta, 10)] = self._compute_hbeta(torch, net, operator_field, beta)

        results: list[AlphaScanResult] = []
        if self.config.candidate_search == "generated":
            structures = generate_rhs_structures(
                max_count=int(self.config.generated_candidates),
                max_terms=int(self.config.generated_max_terms),
                seed=int(self.config.generated_seed),
                eqgpt_dictionary_path=self.config.eqgpt_dictionary_path,
            )
            structures = structures[: max(1, int(self.config.generated_top_structures))]
            for rank, structure in enumerate(structures, start=1):
                for alpha0 in alpha0_grid:
                    if structure.has_fractional:
                        for beta0 in self.config.beta_reference_orders:
                            results.append(
                                self._candidate_result(
                                    torch,
                                    net,
                                    operator_field,
                                    fit_field,
                                    alpha_cache,
                                    beta_cache,
                                    float(alpha0),
                                    float(beta0),
                                    generated_structure=structure,
                                    generated_structure_rank=rank,
                                    generated_structure_count=len(structures),
                                )
                            )
                    else:
                        beta0 = float(self.config.beta_reference_orders[0])
                        results.append(
                            self._candidate_result(
                                torch,
                                net,
                                operator_field,
                                fit_field,
                                alpha_cache,
                                beta_cache,
                                float(alpha0),
                                beta0,
                                generated_structure=structure,
                                generated_structure_rank=rank,
                                generated_structure_count=len(structures),
                            )
                        )
            return results
        if self.config.candidate_search != "fixed":
            raise ValueError("candidate_search must be 'fixed' or 'generated'")
        for alpha0 in alpha0_grid:
            for beta0 in self.config.beta_reference_orders:
                results.append(
                    self._candidate_result(
                        torch,
                        net,
                        operator_field,
                        fit_field,
                        alpha_cache,
                        beta_cache,
                        float(alpha0),
                        float(beta0),
                    )
                )
        return results

    def _candidate_result(
        self,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        alpha_cache: dict[float, np.ndarray],
        beta_cache: dict[float, np.ndarray],
        alpha0: float,
        beta0: float,
        generated_structure: GeneratedStructure | None = None,
        generated_structure_rank: int | None = None,
        generated_structure_count: int | None = None,
        clip_order_corrections: bool = False,
    ) -> AlphaScanResult:
        target0 = self._restrict_values(alpha_cache[round(alpha0, 10)], operator_field, fit_field)
        hbeta0 = self._restrict_values(beta_cache[round(beta0, 10)], operator_field, fit_field)
        base_terms, available_order = self._build_physical_terms(fit_field, hbeta0, beta0)

        alpha_derivative = self._restrict_values(self._alpha_derivative(alpha_cache, alpha0), operator_field, fit_field)
        alpha_correction = -alpha_derivative
        spatial_main_name = f"D_x^{beta0:.7g} H"
        has_fractional_structure = True
        if generated_structure is None:
            base_order = list(available_order)
            if self.config.allowed_physical_terms is not None:
                allowed_terms = set()
                for term in self.config.allowed_physical_terms:
                    if term in {FRACTIONAL_TOKEN, "D_x^beta H", "D_x^beta c"}:
                        allowed_terms.add(spatial_main_name)
                    else:
                        allowed_terms.add(term)
                base_order = [name for name in base_order if name in allowed_terms]
            has_fractional_structure = any(name.startswith("D_x^") for name in base_order)
        else:
            base_order = []
            for abstract_name in generated_structure.terms:
                concrete_name = spatial_main_name if abstract_name == FRACTIONAL_TOKEN else abstract_name
                if concrete_name in base_terms and concrete_name not in base_order:
                    base_order.append(concrete_name)
            has_fractional_structure = any(name.startswith("D_x^") for name in base_order)
        spatial_corr_name = f"spatial_correction_beta0={beta0:.1f}" if has_fractional_structure else None
        extra_columns = [alpha_correction.reshape(-1)]
        extra_names = ["alpha_correction"]
        if has_fractional_structure:
            beta_derivative = self._restrict_values(self._beta_derivative(beta_cache, beta0), operator_field, fit_field)
            extra_columns.append(beta_derivative.reshape(-1))
            extra_names.append(str(spatial_corr_name))
        term_names = list(base_order) + extra_names
        matrix = np.column_stack([base_terms[name].reshape(-1) for name in base_order] + extra_columns)
        target = target0.reshape(-1, 1)
        coefficients, objective, tolerance, l0_penalty, details = self._fde._fit_fde_stridge(
            matrix,
            target,
            term_names=tuple(term_names),
        )
        coefficients, objective, details, grouped_spatial_correction = self._couple_spatial_correction_group(
            matrix=matrix,
            target=target,
            term_names=term_names,
            coefficients=coefficients,
            objective=float(objective),
            l0_penalty=float(l0_penalty),
            details=details,
            spatial_main_name=spatial_main_name,
            spatial_corr_name=spatial_corr_name,
        )
        coef_map = {name: float(coef) for name, coef in zip(term_names, coefficients.reshape(-1))}
        spatial_correction_pruned_without_main = False
        spatial_corr_index = term_names.index(spatial_corr_name) if spatial_corr_name in term_names else None
        spatial_main_index = term_names.index(spatial_main_name) if spatial_main_name in term_names else None
        main_coef_initial = (
            float(coefficients.reshape(-1)[spatial_main_index]) if spatial_main_index is not None else 0.0
        )
        corr_coef_initial = (
            float(coefficients.reshape(-1)[spatial_corr_index]) if spatial_corr_index is not None else 0.0
        )
        if (
            spatial_corr_index is not None
            and np.isclose(main_coef_initial, 0.0)
            and not np.isclose(corr_coef_initial, 0.0)
        ):
            coefficients = coefficients.copy()
            coefficients.reshape(-1)[spatial_corr_index] = 0.0
            coef_map[spatial_corr_name] = 0.0
            spatial_correction_pruned_without_main = True
            candidate_indices = tuple(range(matrix.shape[1]))
            rescored = self._fde._score_coefficients(
                matrix,
                target,
                coefficients,
                float(l0_penalty),
                candidate_indices,
                tuple(term_names),
            )
            details = dict(details)
            details.update(rescored)
            condition = self._fde._condition_from_l0_penalty(float(l0_penalty), matrix)
            details["active_lamb_sum"] = self._fde._active_lamb_sum(
                coefficients,
                tuple(term_names),
                candidate_indices,
            )
            details["condition_number"] = float(condition)
            objective = float(details["validation_residual_norm"] + details["complexity_penalty"])
        delta_alpha = coef_map.get("alpha_correction", 0.0)
        raw_delta_alpha = float(delta_alpha)
        if abs(delta_alpha) < self.config.alpha_correction_tol:
            delta_alpha = 0.0
            coef_map["alpha_correction"] = 0.0
        if clip_order_corrections:
            delta_alpha = float(np.clip(delta_alpha, *self.config.iter_delta_alpha_bounds))
            alpha_est = self._clip_iter_alpha(alpha0 + delta_alpha)
            delta_alpha = float(alpha_est - alpha0)
        else:
            alpha_est = float(alpha0 + delta_alpha)

        main_coef = coef_map.get(spatial_main_name, 0.0)
        corr_coef = coef_map.get(spatial_corr_name, 0.0)
        delta_beta = 0.0
        raw_delta_beta = 0.0
        beta_est = float(beta0)
        if not np.isclose(main_coef, 0.0):
            delta_beta = corr_coef / main_coef if spatial_corr_name is not None and not np.isclose(corr_coef, 0.0) else 0.0
            raw_delta_beta = float(delta_beta)
            if abs(delta_beta) < self.config.beta_correction_tol:
                delta_beta = 0.0
                if spatial_corr_name is not None:
                    coef_map[spatial_corr_name] = 0.0
            if clip_order_corrections:
                delta_beta = float(np.clip(delta_beta, *self.config.iter_delta_beta_bounds))
                beta_est = float(np.clip(beta0 + delta_beta, *self.config.beta_bounds))
                delta_beta = float(beta_est - beta0)
            else:
                beta_est = float(beta0 + delta_beta)

        augmented_residual = target - matrix @ coefficients
        augmented_mse = float(np.mean(augmented_residual**2))
        excluded_terms = {"alpha_correction"}
        if spatial_corr_name is not None:
            excluded_terms.add(spatial_corr_name)
        active_physical = [
            name
            for name in term_names
            if name not in excluded_terms and not np.isclose(coef_map.get(name, 0.0), 0.0)
        ]

        metadata = self._candidate_metadata(
            operator_field=operator_field,
            fit_field=fit_field,
            term_names=tuple(term_names),
            matrix=matrix,
            alpha0=alpha0,
            beta0=beta0,
            alpha_est=alpha_est,
            beta_est=beta_est,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            raw_delta_alpha=raw_delta_alpha,
            raw_delta_beta=raw_delta_beta,
            clip_order_corrections=clip_order_corrections,
            objective=float(objective),
            tolerance=float(tolerance),
            l0_penalty=float(l0_penalty),
            details=details,
            active_physical=tuple(active_physical),
            augmented_coefficients=tuple(float(v) for v in coefficients.reshape(-1)),
        )
        target_centered = target.reshape(-1) - float(np.mean(target))
        target_tss = float(np.sum(target_centered**2))
        augmented_sse = float(np.sum(augmented_residual.reshape(-1) ** 2))
        target_r2 = 0.0 if target_tss <= 0.0 else 1.0 - augmented_sse / target_tss
        metadata.update(
            {
                "target_sse": augmented_sse,
                "target_tss": target_tss,
                "target_r2": float(target_r2),
                "active_physical_count": int(len(active_physical)),
            }
        )
        metadata["spatial_correction_group_repaired"] = bool(grouped_spatial_correction)
        metadata["spatial_correction_pruned_without_main"] = bool(spatial_correction_pruned_without_main)
        self._add_generated_metadata(
            metadata,
            generated_structure=generated_structure,
            generated_structure_rank=generated_structure_rank,
            generated_structure_count=generated_structure_count,
            has_fractional_structure=has_fractional_structure,
        )
        generated_structure_penalty = self._generated_structure_objective_penalty(generated_structure)
        endpoint_penalty = self._spatial_fractional_endpoint_penalty(
            beta_est=beta_est,
            has_fractional_structure=has_fractional_structure,
        )
        selection_objective = float(objective + generated_structure_penalty + endpoint_penalty)
        metadata.update(
            {
                "augmented_stridge_objective_raw": float(objective),
                "generated_structure_objective_penalty": generated_structure_penalty,
                "spatial_fractional_endpoint_penalty": endpoint_penalty,
                "objective": selection_objective,
                "selection_loss_formula": (
                    "validation_residual_norm + cond(Theta)*sum(active_term_lamb) "
                    "+ generated_structure_objective_penalty + spatial_fractional_endpoint_penalty"
                ),
            }
        )

        unpaired_spatial_correction = (
            spatial_corr_name is not None
            and not np.isclose(corr_coef, 0.0)
            and np.isclose(main_coef, 0.0)
        )
        if unpaired_spatial_correction:
            metadata.update(
                {
                    "invalid_order": True,
                    "invalid_order_reason": "spatial beta correction is active without the D_x^beta H main column",
                    "spatial_beta_model_valid": False,
                    "spatial_beta_status": "invalid_unpaired_correction",
                    "physical_refit_selection_objective": float("inf"),
                    "objective": float("inf"),
                }
            )
            model = SparseModel(
                target_name="D_t^alpha H",
                term_names=tuple(term_names),
                coefficients=coefficients.reshape(-1),
                tolerance=float(tolerance),
                ridge_lambda=self.config.ridge_lambda,
                mse=augmented_mse,
                residual_norm=self._fde._norm2(augmented_residual),
                information_criterion=float("inf"),
                support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
                sample_count=int(matrix.shape[0]),
                metadata=metadata,
            )
            return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=float("inf"), model=model)

        if not active_physical:
            metadata["physical_refit_selection_objective"] = float("inf")
            metadata["objective"] = float("inf")
            model = SparseModel(
                target_name="D_t^alpha H",
                term_names=tuple(term_names),
                coefficients=coefficients.reshape(-1),
                tolerance=float(tolerance),
                ridge_lambda=self.config.ridge_lambda,
                mse=augmented_mse,
                residual_norm=self._fde._norm2(augmented_residual),
                information_criterion=float("inf"),
                support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
                sample_count=int(matrix.shape[0]),
                metadata=metadata,
            )
            return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=float("inf"), model=model)

        if not (0.0 < alpha_est < 1.0) or not (1.0 < beta_est <= 2.0):
            metadata.update(
                {
                    "invalid_order": True,
                    "invalid_order_reason": "Taylor-updated alpha/beta is outside the G-J operator domain",
                    "physical_refit_selection_objective": float("inf"),
                    "objective": float("inf"),
                }
            )
            model = SparseModel(
                target_name="D_t^alpha H",
                term_names=tuple(term_names),
                coefficients=coefficients.reshape(-1),
                tolerance=float(tolerance),
                ridge_lambda=self.config.ridge_lambda,
                mse=augmented_mse,
                residual_norm=self._fde._norm2(augmented_residual),
                information_criterion=float("inf"),
                support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
                sample_count=int(matrix.shape[0]),
                metadata=metadata,
            )
            return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=float("inf"), model=model)

        if self.config.selection_objective == "physical-stridge":
            return self._physical_stridge_candidate(
                torch=torch,
                net=net,
                operator_field=operator_field,
                fit_field=fit_field,
                alpha_cache=alpha_cache,
                beta_cache=beta_cache,
                alpha0=alpha0,
                beta0=beta0,
                alpha_est=alpha_est,
                beta_est=beta_est,
                delta_alpha=delta_alpha,
                base_order=base_order,
                metadata=metadata,
                generated_structure_penalty=generated_structure_penalty,
            )

        if self.config.refit_mode == "none":
            metadata.update(
                {
                    "physical_refit": False,
                    "physical_refit_note": "no final physical refit; selected Taylor-augmented STRidge coefficients are reported",
                    "refit_mode": "none",
                    "physical_refit_selection_objective": selection_objective,
                    "physical_refit_selection_residual_norm": float(details["validation_residual_norm"]),
                    "physical_refit_selection_condition_penalty": float(details["complexity_penalty"]),
                    "selection_objective_mode": self.config.selection_objective,
                    "alpha": alpha_est,
                    "end_to_end_seconds": None,
                }
            )
            no_refit_model = self._no_refit_physical_model(
                active_physical=active_physical,
                coef_map=coef_map,
                metadata=metadata,
                beta_est=beta_est,
                augmented_residual=augmented_residual,
                tolerance=float(tolerance),
                physical_objective=selection_objective,
                sample_count=int(matrix.shape[0]),
            )
            return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=selection_objective, model=no_refit_model)

        halpha_est = self._restrict_values(
            self._get_or_compute_halpha(torch, net, operator_field, alpha_cache, alpha_est),
            operator_field,
            fit_field,
        )
        hbeta_est = self._restrict_values(
            self._get_or_compute_hbeta(torch, net, operator_field, beta_cache, beta_est),
            operator_field,
            fit_field,
        )
        refit_terms, _ = self._build_physical_terms(fit_field, hbeta_est, beta_est)
        refit_support_names: list[str] = []
        for name in active_physical:
            if name.startswith("D_x^"):
                refit_support_names.append(next(candidate for candidate in refit_terms if candidate.startswith("D_x^")))
            else:
                refit_support_names.append(name)
        refit_matrix = np.column_stack([refit_terms[name].reshape(-1) for name in refit_support_names])
        refit_target = halpha_est.reshape(-1, 1)
        refit_coef = self._fde._lstsq(refit_matrix, refit_target)
        refit_residual = refit_target - refit_matrix @ refit_coef
        refit_validation = self._fde._validation_residual_norm(refit_matrix, refit_target, refit_coef)
        refit_condition = self._fde._finite_condition(refit_matrix)
        refit_condition_penalty = float(self.config.sparsity_lamb * refit_condition * np.count_nonzero(refit_coef))
        physical_objective = float(refit_validation + refit_condition_penalty + generated_structure_penalty)
        metadata.update(
            {
                "physical_refit": True,
                "physical_refit_note": "coefficients refit on G-J operator columns after local alpha/beta Taylor update",
                "refit_mode": "linear",
                "physical_refit_terms": tuple(refit_support_names),
                "physical_refit_spatial_fractional_mode": "gj_hybrid_taylor",
                "physical_refit_space_derivative_mode": "autodiff",
                "physical_refit_time_operator_mode": "gj_caputo_taylor",
                "physical_refit_mse": float(np.mean(refit_residual**2)),
                "physical_refit_residual_norm": self._fde._norm2(refit_residual),
                "physical_refit_validation_residual_norm": refit_validation,
                "physical_refit_condition": refit_condition,
                "physical_refit_condition_penalty": refit_condition_penalty,
                "generated_structure_objective_penalty": generated_structure_penalty,
                "physical_refit_selection_objective": physical_objective,
                "physical_refit_selection_residual_norm": refit_validation,
                "physical_refit_selection_condition_penalty": refit_condition_penalty,
                "selection_objective_mode": self.config.selection_objective,
                "alpha": alpha_est,
                "end_to_end_seconds": None,
            }
        )
        model = SparseModel(
            target_name="D_t^alpha H",
            term_names=tuple(refit_support_names),
            coefficients=refit_coef.reshape(-1),
            tolerance=float(tolerance),
            ridge_lambda=self.config.ridge_lambda,
            mse=float(np.mean(refit_residual**2)),
            residual_norm=self._fde._norm2(refit_residual),
            information_criterion=physical_objective,
            support_frequency=(np.abs(refit_coef.reshape(-1)) > 0.0).astype(float),
            sample_count=int(refit_matrix.shape[0]),
            metadata=metadata,
        )
        return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=physical_objective, model=model)

    def _physical_stridge_candidate(
        self,
        *,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        alpha_cache: dict[float, np.ndarray],
        beta_cache: dict[float, np.ndarray],
        alpha0: float,
        beta0: float,
        alpha_est: float,
        beta_est: float,
        delta_alpha: float,
        base_order: list[str],
        metadata: dict[str, Any],
        generated_structure_penalty: float,
    ) -> AlphaScanResult:
        halpha_est = self._restrict_values(
            self._get_or_compute_halpha(torch, net, operator_field, alpha_cache, alpha_est),
            operator_field,
            fit_field,
        )
        hbeta_est = self._restrict_values(
            self._get_or_compute_hbeta(torch, net, operator_field, beta_cache, beta_est),
            operator_field,
            fit_field,
        )
        physical_terms, _ = self._build_physical_terms(fit_field, hbeta_est, beta_est)
        fractional_name = next((name for name in physical_terms if name.startswith("D_x^")), None)
        physical_order: list[str] = []
        for name in base_order:
            physical_name = fractional_name if name.startswith("D_x^") else name
            if physical_name is None or physical_name not in physical_terms:
                continue
            if physical_name not in physical_order:
                physical_order.append(physical_name)
        if not physical_order:
            metadata.update(
                {
                    "physical_stridge": True,
                    "physical_stridge_note": "no physical candidates remained after order update",
                    "physical_refit_selection_objective": float("inf"),
                    "objective": float("inf"),
                }
            )
            model = SparseModel(
                target_name="D_t^alpha H",
                term_names=(),
                coefficients=np.asarray([], dtype=float),
                tolerance=float("nan"),
                ridge_lambda=self.config.ridge_lambda,
                mse=float("inf"),
                residual_norm=float("inf"),
                information_criterion=float("inf"),
                support_frequency=np.asarray([], dtype=float),
                sample_count=int(halpha_est.size),
                metadata=metadata,
            )
            return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=float("inf"), model=model)

        physical_matrix = np.column_stack([physical_terms[name].reshape(-1) for name in physical_order])
        physical_target = halpha_est.reshape(-1, 1)
        coefficients, objective, tolerance, l0_penalty, details = self._fde._fit_fde_stridge(
            physical_matrix,
            physical_target,
            term_names=tuple(physical_order),
        )
        residual = physical_target - physical_matrix @ coefficients
        physical_objective = float(objective)
        active_physical_order = tuple(
            name
            for name, coefficient in zip(physical_order, coefficients.reshape(-1))
            if not np.isclose(float(coefficient), 0.0)
        )
        metadata.update(
            {
                "physical_stridge": True,
                "physical_stridge_note": (
                    "order-correction columns are used only to update alpha/beta; "
                    "support and objective are recomputed on the fixed-order physical library"
                ),
                "physical_refit": False,
                "refit_mode": self.config.refit_mode,
                "physical_stridge_library_terms": tuple(physical_order),
                "physical_refit_terms": active_physical_order,
                "physical_refit_selection_objective": physical_objective,
                "physical_refit_selection_residual_norm": float(details["validation_residual_norm"]),
                "physical_refit_selection_condition_penalty": float(details["complexity_penalty"]),
                "physical_stridge_l0_penalty": float(l0_penalty),
                "physical_stridge_tolerance": float(tolerance),
                "physical_stridge_active_count": int(details["active_count"]),
                "physical_stridge_active_lamb_sum": float(details["active_lamb_sum"]),
                "physical_stridge_condition_number": float(details["condition_number"]),
                "generated_structure_objective_penalty": generated_structure_penalty,
                "selection_objective_mode": self.config.selection_objective,
                "selection_loss_formula": "physical validation_residual_norm + cond(physical_Theta)*sum(active_physical_lamb)",
                "objective": physical_objective,
                "alpha": alpha_est,
                "end_to_end_seconds": None,
            }
        )
        model = SparseModel(
            target_name="D_t^alpha H",
            term_names=tuple(physical_order),
            coefficients=coefficients.reshape(-1),
            tolerance=float(tolerance),
            ridge_lambda=self.config.ridge_lambda,
            mse=float(np.mean(residual**2)),
            residual_norm=self._fde._norm2(residual),
            information_criterion=physical_objective,
            support_frequency=(np.abs(coefficients.reshape(-1)) > 0.0).astype(float),
            sample_count=int(physical_matrix.shape[0]),
            metadata=metadata,
        )
        return AlphaScanResult(alpha0=alpha0, alpha=alpha_est, delta_alpha=delta_alpha, objective=physical_objective, model=model)

    def _couple_spatial_correction_group(
        self,
        *,
        matrix: np.ndarray,
        target: np.ndarray,
        term_names: list[str],
        coefficients: np.ndarray,
        objective: float,
        l0_penalty: float,
        details: dict[str, float | int],
        spatial_main_name: str,
        spatial_corr_name: str | None,
    ) -> tuple[np.ndarray, float, dict[str, float | int], bool]:
        del matrix, target, l0_penalty, spatial_main_name, spatial_corr_name
        details = dict(details)
        details["spatial_correction_group_repaired"] = 0
        details["spatial_correction_group_policy"] = "no_repair"
        return coefficients, objective, details, False

    @staticmethod
    def _generated_structure_objective_penalty(generated_structure: GeneratedStructure | None) -> float:
        if generated_structure is None:
            return 0.0
        return float(5.0e-2 * generated_structure.complexity_score)

    def _spatial_fractional_endpoint_penalty(self, *, beta_est: float, has_fractional_structure: bool) -> float:
        if not has_fractional_structure:
            return 0.0
        upper = float(self.config.beta_bounds[1])
        if np.isclose(float(beta_est), upper, atol=1.0e-10, rtol=0.0):
            return 1.0e-1
        return 0.0

    def _no_refit_physical_model(
        self,
        *,
        active_physical: list[str],
        coef_map: dict[str, float],
        metadata: dict[str, Any],
        beta_est: float,
        augmented_residual: np.ndarray,
        tolerance: float,
        physical_objective: float,
        sample_count: int,
    ) -> SparseModel:
        term_names: list[str] = []
        coefficients: list[float] = []
        for name in active_physical:
            coef = float(coef_map.get(name, 0.0))
            if name.startswith("D_x^"):
                term_names.append(f"D_x^{float(beta_est):.7g} H")
            else:
                term_names.append(name)
            coefficients.append(coef)
        coeff_array = np.asarray(coefficients, dtype=float)
        no_refit_metadata = dict(metadata)
        no_refit_metadata.update(
            {
                "physical_refit": False,
                "physical_refit_note": "reported coefficients are the selected Taylor-augmented STRidge coefficients; no final physical least-squares refit was applied",
                "physical_refit_terms": tuple(term_names),
                "refit_mode": "none",
                "reported_without_physical_refit": True,
                "physical_refit_selection_objective": physical_objective,
                "physical_refit_selection_residual_norm": metadata.get("physical_refit_selection_residual_norm"),
                "physical_refit_selection_condition_penalty": metadata.get("physical_refit_selection_condition_penalty"),
            }
        )
        return SparseModel(
            target_name="D_t^alpha H",
            term_names=tuple(term_names),
            coefficients=coeff_array,
            tolerance=tolerance,
            ridge_lambda=self.config.ridge_lambda,
            mse=float(np.mean(augmented_residual**2)),
            residual_norm=self._fde._norm2(augmented_residual),
            information_criterion=physical_objective,
            support_frequency=(np.abs(coeff_array) > 0.0).astype(float),
            sample_count=sample_count,
            metadata=no_refit_metadata,
        )

    def _nonlinear_refit_model(
        self,
        torch: Any,
        net: Any,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        model: SparseModel,
    ) -> SparseModel:
        active = [
            str(name)
            for name, coef in zip(model.term_names, np.asarray(model.coefficients, dtype=float).reshape(-1))
            if not np.isclose(float(coef), 0.0)
        ]
        has_hx = "Hx" in active
        frac_terms = [name for name in active if name.startswith("D_x^")]
        metadata = dict(model.metadata)
        metadata["refit_mode"] = "nonlinear"
        if not (has_hx and len(frac_terms) == 1 and len(active) == 2):
            metadata.update(
                {
                    "nonlinear_refit_success": False,
                    "nonlinear_refit_skipped": True,
                    "nonlinear_refit_message": "nonlinear refit requires exactly Hx plus one D_x^beta term",
                }
            )
            return SparseModel(
                target_name=model.target_name,
                term_names=model.term_names,
                coefficients=model.coefficients.copy(),
                tolerance=model.tolerance,
                ridge_lambda=model.ridge_lambda,
                mse=model.mse,
                residual_norm=model.residual_norm,
                information_criterion=model.information_criterion,
                support_frequency=model.support_frequency.copy(),
                sample_count=model.sample_count,
                metadata=metadata,
            )

        alpha0 = float(metadata.get("alpha", 0.8))
        beta0 = float(metadata.get("spatial_beta", 1.8))
        alpha_cache: dict[float, np.ndarray] = {}
        beta_cache: dict[float, np.ndarray] = {}
        initial_coef = np.array(
            [
                [self._coef_at(model.term_names, model.coefficients, "Hx") or 0.0],
                [self._coef_at(model.term_names, model.coefficients, "D_x") or 0.0],
            ],
            dtype=float,
        )
        last: dict[str, Any] = {"alpha": alpha0, "beta": beta0, "coef": initial_coef}

        def columns(alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
            a_key = round(float(alpha), 10)
            b_key = round(float(beta), 10)
            if a_key not in alpha_cache:
                alpha_cache[a_key] = self._compute_halpha(torch, net, operator_field, float(alpha))
            if b_key not in beta_cache:
                beta_cache[b_key] = self._compute_hbeta(torch, net, operator_field, float(beta))
            hbeta = self._restrict_values(beta_cache[b_key], operator_field, fit_field)
            halpha = self._restrict_values(alpha_cache[a_key], operator_field, fit_field)
            matrix = np.column_stack((fit_field.Hx.reshape(-1), hbeta.reshape(-1)))
            target = halpha.reshape(-1, 1)
            return matrix, target

        def residual_for_orders(params: np.ndarray) -> np.ndarray:
            alpha = float(params[0])
            beta = float(params[1])
            matrix, target = columns(alpha, beta)
            coef = self._fde._lstsq(matrix, target)
            residual = target - matrix @ coef
            scale = max(float(np.linalg.norm(target.reshape(-1))), np.finfo(float).eps)
            last.update({"alpha": alpha, "beta": beta, "coef": coef})
            return residual.reshape(-1) / scale

        opt = None
        status = "completed"
        message = ""
        try:
            opt = least_squares(
                residual_for_orders,
                x0=np.array([np.clip(alpha0, 0.011, 0.999), np.clip(beta0, 1.011, 1.999)], dtype=float),
                bounds=([0.01, 1.01], [0.99999, 1.99999]),
                xtol=1.0e-5,
                ftol=1.0e-5,
                gtol=1.0e-5,
                max_nfev=80,
            )
            residual_for_orders(opt.x)
        except Exception as exc:  # pragma: no cover - diagnostic failure path
            status = "failed"
            message = repr(exc)

        alpha = float(last["alpha"])
        beta = float(last["beta"])
        coef = np.asarray(last["coef"], dtype=float).reshape(-1, 1)
        matrix, target = columns(alpha, beta)
        residual = target - matrix @ coef
        term_names = ("Hx", f"D_x^{beta:.7g} H")
        metadata.update(
            {
                "alpha": alpha,
                "spatial_beta": beta,
                "spatial_delta_beta": beta - float(metadata.get("beta0", beta)),
                "spatial_beta_status": "nonlinear_refit",
                "spatial_beta_in_bounds": bool(self.config.beta_bounds[0] <= beta <= self.config.beta_bounds[1]),
                "spatial_beta_model_valid": bool(self.config.beta_bounds[0] <= beta <= self.config.beta_bounds[1]),
                "physical_refit": True,
                "physical_refit_note": "nonlinear order refit over alpha/beta with inner linear least squares for Hx and D_x^beta H",
                "physical_refit_terms": term_names,
                "physical_refit_mse": float(np.mean(residual**2)),
                "physical_refit_residual_norm": self._fde._norm2(residual),
                "physical_refit_validation_residual_norm": self._fde._validation_residual_norm(matrix, target, coef),
                "physical_refit_condition": self._fde._finite_condition(matrix),
                "nonlinear_refit_success": None if opt is None else bool(opt.success),
                "nonlinear_refit_cost": None if opt is None else float(opt.cost),
                "nonlinear_refit_nfev": None if opt is None else int(opt.nfev),
                "nonlinear_refit_status": status,
                "nonlinear_refit_message": message,
            }
        )
        return SparseModel(
            target_name="D_t^alpha H",
            term_names=term_names,
            coefficients=coef.reshape(-1),
            tolerance=model.tolerance,
            ridge_lambda=model.ridge_lambda,
            mse=float(np.mean(residual**2)),
            residual_norm=self._fde._norm2(residual),
            information_criterion=float(np.mean(residual**2)),
            support_frequency=(np.abs(coef.reshape(-1)) > 0.0).astype(float),
            sample_count=int(matrix.shape[0]),
            metadata=metadata,
        )

    @staticmethod
    def _coef_at(term_names: tuple[str, ...], coefficients: np.ndarray, prefix: str) -> float | None:
        for name, coef in zip(term_names, np.asarray(coefficients, dtype=float).reshape(-1)):
            if str(name) == prefix or str(name).startswith(prefix):
                return float(coef)
        return None

    def _candidate_metadata(
        self,
        *,
        operator_field: GJBaseField,
        fit_field: GJBaseField,
        term_names: tuple[str, ...],
        matrix: np.ndarray,
        alpha0: float,
        beta0: float,
        alpha_est: float,
        beta_est: float,
        delta_alpha: float,
        delta_beta: float,
        raw_delta_alpha: float,
        raw_delta_beta: float,
        clip_order_corrections: bool,
        objective: float,
        tolerance: float,
        l0_penalty: float,
        details: dict[str, float | int],
        active_physical: tuple[str, ...],
        augmented_coefficients: tuple[float, ...],
    ) -> dict[str, Any]:
        training_spec = self._fde._network_spec(operator_field.training_metadata)
        metadata = {
            "route": "gj_hybrid_taylor",
            "case": self.config.case_name,
            "candidate_search": self.config.candidate_search,
            "generated_structure": None,
            "generated_raw_structure": None,
            "generated_normalized_from": (),
            "generated_structure_has_fractional": None,
            "generated_structure_rank": None,
            "generated_prior_score": None,
            "generated_complexity_score": None,
            "generated_structure_objective_penalty": 0.0,
            "generated_structure_count": None,
            "eqgpt_checkpoint": None,
            "eqgpt_dictionary": None,
            "eqgpt_epoch": None,
            "eqgpt_sample_index": None,
            "eqgpt_sample_count": None,
            "eqgpt_unique_structure_count": None,
            "eqgpt_optimize_epochs": None,
            "eqgpt_finetune_epochs": None,
            "eqgpt_learning_rate": None,
            "eqgpt_reward_sparsity_alpha": None,
            "eqgpt_random_exploration": None,
            "eqgpt_reward": None,
            "eqgpt_r2": None,
            "eqgpt_reward_rank": None,
            "eqgpt_source_equation": None,
            "eqgpt_token_ids": (),
            "checkpoint": operator_field.checkpoint_path,
            "training_metadata": operator_field.training_metadata,
            "training_best": self._fde._format_training_best(operator_field.training_metadata),
            "activation": training_spec["activation"],
            "hidden_layers": training_spec["hidden_layers"],
            "neurons": training_spec["neurons"],
            "model_file": self.config.model_file,
            "trained_point": self.config.trained_point,
            "noise_level": self.config.noise_level,
            "grid_shape": fit_field.H.shape,
            "x_range": (float(fit_field.position[0]), float(fit_field.position[-1])),
            "t_range": (float(fit_field.time[0]), float(fit_field.time[-1])),
            "operator_grid_shape": operator_field.H.shape,
            "operator_x_range": (float(operator_field.position[0]), float(operator_field.position[-1])),
            "operator_t_range": (float(operator_field.time[0]), float(operator_field.time[-1])),
            "fit_grid_shape": fit_field.H.shape,
            "fit_x_range": (float(fit_field.position[0]), float(fit_field.position[-1])),
            "fit_t_range": (float(fit_field.time[0]), float(fit_field.time[-1])),
            "fit_window_config": (
                self.config.fit_x_min,
                self.config.fit_x_max,
                self.config.fit_t_min,
                self.config.fit_t_max,
            ),
            "initial_condition_time": float(operator_field.time[0]),
            "initial_condition_note": "G-J operators are evaluated on the full reconstruction domain and then restricted to the fit window for STRidge",
            "term_names": term_names,
            "matrix_shape": matrix.shape,
            "alpha0_count": len(tuple(value for value in self.config.alpha0_grid if 0.0 < float(value) < 1.0)),
            "beta0_count": len(self.config.beta_reference_orders),
            "alpha0": float(alpha0),
            "beta0": float(beta0),
            "delta_alpha": float(delta_alpha),
            "delta_alpha_raw": float(raw_delta_alpha),
            "delta_alpha_clipped": bool(not np.isclose(delta_alpha, raw_delta_alpha)),
            "alpha": float(alpha_est),
            "alpha_correction_mode": "gj_operator_numeric_taylor",
            "alpha_correction_coefficient": float(delta_alpha),
            "alpha_correction_tol": self.config.alpha_correction_tol,
            "beta_correction_tol": self.config.beta_correction_tol,
            "delta_alpha_prune_threshold": self.config.delta_alpha_prune_threshold,
            "delta_alpha_bounds": self.config.delta_alpha_bounds,
            "delta_beta_prune_threshold": self.config.delta_beta_prune_threshold,
            "ridge_lambda": self.config.ridge_lambda,
            "d_tol": self.config.d_tol,
            "maxit": self.config.maxit,
            "STR_iters": self.config.str_iters,
            "normalize": self.config.normalize,
            "split": self.config.split,
            "sparsity_lamb": self.config.sparsity_lamb,
            "fractional_correction_sparsity_lamb": self._fde._fractional_correction_sparsity_lamb(),
            "fractional_correction_tol_scale": self.config.fractional_correction_tol_scale,
            "objective": objective,
            "fde_stridge_loss": objective,
            "l0_penalty": l0_penalty,
            "validation_residual_norm": float(details["validation_residual_norm"]),
            "active_count_for_loss": int(details["active_count"]),
            "complexity_penalty": float(details["complexity_penalty"]),
            "active_lamb_sum": float(details.get("active_lamb_sum", float("nan"))),
            "condition_number": float(details.get("condition_number", self._fde._finite_condition(matrix))),
            "loss_formula": "validation_residual_norm + cond(Theta)*sum(active_term_lamb)",
            "tolerance": tolerance,
            "selection_objective_mode": self.config.selection_objective,
            "order_radius_mode": self.config.order_radius_mode,
            "order_radius_penalty": 0.0,
            "space_derivative_mode": "autodiff",
            "spatial_fractional_mode": "gj_hybrid_taylor",
            "time_operator_mode": "gj_caputo_taylor",
            "laplace_out_of_bounds_fraction": 0.0,
            "s_range": None,
            "s_points": 0,
            "laguerre_nodes": 0,
            "spatial_fractional_enabled": True,
            "spatial_correction_active": not np.isclose(delta_beta, 0.0),
            "spatial_beta_status": "estimated",
            "spatial_beta": float(beta_est),
            "spatial_delta_beta": float(delta_beta),
            "spatial_delta_beta_raw": float(raw_delta_beta),
            "spatial_delta_beta_clipped": bool(not np.isclose(delta_beta, raw_delta_beta)),
            "spatial_beta_correction": float(delta_beta),
            "spatial_beta_in_bounds": bool(self.config.beta_bounds[0] <= float(beta_est) <= self.config.beta_bounds[1]),
            "spatial_beta_bounds": self.config.beta_bounds,
            "spatial_beta_model_valid": bool(self.config.beta_bounds[0] <= float(beta_est) <= self.config.beta_bounds[1]),
            "spatial_log_quad_points": self.config.spatial_log_quad_points,
            "spatial_log_quad_method": "jacobi_gauss",
            "spatial_correction_method": "gj_hybrid_taylor",
            "spatial_boundary_mode": self.config.spatial_boundary_mode,
            "spatial_boundary_value": self.config.spatial_boundary_value,
            "spatial_operator_note": "time and spatial fractional operators both use Jacobi-Gauss quadrature with autodiff derivatives",
            "internal_augmented_terms": term_names,
            "internal_augmented_coefficients": augmented_coefficients,
            "selected_by": None,
            "physical_refit": False,
            "physical_refit_note": None,
            "refit_mode": self.config.refit_mode,
            "order_update_mode": self.config.order_update_mode,
            "iteration_count": None,
            "iteration_converged": None,
            "iteration_stop_reason": None,
            "iteration_final_index": None,
            "iteration_best_objective_index": None,
            "iteration_best_fractional_support_index": None,
            "iteration_selected_index": None,
            "iteration_trace": (),
            "iteration_selected": False,
            "iter_clip_order_corrections": clip_order_corrections,
            "iter_start_alpha": None,
            "iter_start_beta": None,
            "iter_order_tol": None,
            "iter_damping": None,
            "iter_max_step_alpha": None,
            "iter_max_step_beta": None,
            "iter_delta_alpha_bounds": self.config.iter_delta_alpha_bounds,
            "iter_delta_beta_bounds": self.config.iter_delta_beta_bounds,
            "iter_stop_on_non_decreasing_objective": self.config.iter_stop_on_non_decreasing_objective,
            "iter_objective_min_delta": self.config.iter_objective_min_delta,
            "physical_refit_terms": tuple(active_physical),
            "physical_refit_selection_objective": float("inf"),
            "physical_refit_selection_residual_norm": float("inf"),
            "physical_refit_selection_condition_penalty": float("inf"),
            "end_to_end_seconds": None,
        }
        return metadata

    @staticmethod
    def _add_generated_metadata(
        metadata: dict[str, Any],
        *,
        generated_structure: GeneratedStructure | None,
        generated_structure_rank: int | None,
        generated_structure_count: int | None,
        has_fractional_structure: bool,
    ) -> None:
        if generated_structure is None:
            return
        metadata.update(
            {
                "candidate_search": "generated",
                "generated_structure": generated_structure.text,
                "generated_raw_structure": generated_structure.raw_text,
                "generated_normalized_from": generated_structure.normalized_from,
                "generated_structure_has_fractional": bool(has_fractional_structure),
                "generated_structure_rank": generated_structure_rank,
                "generated_prior_score": float(generated_structure.prior_score),
                "generated_complexity_score": float(generated_structure.complexity_score),
                "generated_rank_score": float(generated_structure.rank_score),
                "generated_structure_count": generated_structure_count,
                "generated_source": generated_structure.source,
                "generated_source_equation": generated_structure.source_equation,
                "eqgpt_token_ids": tuple(int(token) for token in generated_structure.token_ids),
            }
        )
        if not has_fractional_structure:
            metadata.update(
                {
                    "spatial_fractional_enabled": False,
                    "spatial_correction_active": False,
                    "spatial_beta_status": "not_applicable",
                    "spatial_beta": None,
                    "spatial_delta_beta": None,
                    "spatial_beta_correction": None,
                    "spatial_beta_in_bounds": None,
                    "spatial_beta_model_valid": True,
                }
            )

    def _iter_start_beta_values(self) -> tuple[float, ...]:
        values = self.config.iter_start_beta_values
        if values is None:
            return (float(self.config.iter_start_beta),)
        cleaned: list[float] = []
        for value in values:
            beta = self._clip_iter_beta(float(value))
            if not any(np.isclose(beta, existing) for existing in cleaned):
                cleaned.append(beta)
        return tuple(cleaned) or (float(self.config.iter_start_beta),)

    def _generated_candidate_summaries(self, scan: list[AlphaScanResult]) -> list[dict[str, Any]]:
        if self.config.candidate_search != "generated":
            return []
        grouped: dict[tuple[int, int], list[AlphaScanResult]] = {}
        for item in scan:
            rank = item.model.metadata.get("generated_structure_rank")
            if rank is not None:
                epoch = int(item.model.metadata.get("eqgpt_epoch") or 0)
                grouped.setdefault((epoch, int(rank)), []).append(item)
        summaries: list[dict[str, Any]] = []
        for epoch, rank in sorted(grouped):
            candidates = grouped[(epoch, rank)]
            valid = [item for item in candidates if self._fde._order_result_is_valid(item)] or candidates
            best = min(valid, key=lambda item: float(item.model.metadata.get("objective", item.objective)))
            metadata = best.model.metadata
            support = tuple(str(name) for name in best.model.support_names)
            coef_hx = self._coef_at(best.model.term_names, best.model.coefficients, "Hx")
            coef_dbeta = next(
                (
                    float(coef)
                    for name, coef in zip(best.model.term_names, best.model.coefficients.reshape(-1))
                    if str(name).startswith("D_x^")
                ),
                None,
            )
            summaries.append(
                {
                    "rank": rank,
                    "epoch": epoch,
                    "structure": metadata.get("generated_structure"),
                    "raw_structure": metadata.get("generated_raw_structure"),
                    "normalized_from": metadata.get("generated_normalized_from"),
                    "prior_score": metadata.get("generated_prior_score"),
                    "complexity_score": metadata.get("generated_complexity_score"),
                    "structure_penalty": metadata.get("generated_structure_objective_penalty"),
                    "eqgpt_reward": metadata.get("eqgpt_reward"),
                    "eqgpt_r2": metadata.get("eqgpt_r2"),
                    "eqgpt_source_equation": metadata.get("eqgpt_source_equation"),
                    "endpoint_penalty": metadata.get("spatial_fractional_endpoint_penalty"),
                    "selected": bool(metadata.get("iteration_selected", False)),
                    "alpha": float(best.alpha),
                    "beta": metadata.get("spatial_beta"),
                    "coef_hx": coef_hx,
                    "coef_dbeta": coef_dbeta,
                    "objective": metadata.get("objective", best.objective),
                    "augmented_objective": metadata.get("augmented_stridge_objective_raw"),
                    "validation_residual_norm": metadata.get("validation_residual_norm"),
                    "condition_number": metadata.get("condition_number"),
                    "support": "+".join(support) if support else "none",
                    "iteration_index": metadata.get("iteration_index"),
                    "iter_start_beta": metadata.get("iter_start_beta"),
                }
            )
        return summaries

    def _select_best(self, scan: list[AlphaScanResult]) -> AlphaScanResult:
        for item in scan:
            item.model.metadata["iteration_selected"] = False
            item.model.metadata["selected_by"] = f"minimum_valid_{self.config.selection_objective}_objective"
        valid = [item for item in scan if self._fde._order_result_is_valid(item)]
        if not valid:
            raise RuntimeError("No G-J hybrid candidate satisfies the configured alpha/beta bounds.")
        if self.config.selection_objective == "augmented":
            best = min(valid, key=lambda item: float(item.model.metadata.get("objective", item.objective)))
        elif self.config.selection_objective in {"physical-refit", "physical-stridge"}:
            best = min(valid, key=lambda item: float(item.model.metadata.get("physical_refit_selection_objective", item.objective)))
        else:
            raise ValueError("selection_objective must be 'augmented', 'physical-refit', or 'physical-stridge'")
        best.model.metadata["iteration_selected"] = True
        return best

    def _select_iterative_final(self, scan: list[AlphaScanResult]) -> AlphaScanResult:
        if not scan:
            raise RuntimeError("Iterative order update produced no candidates.")
        for item in scan:
            item.model.metadata["iteration_selected"] = False
            item.model.metadata["iter_selection_mode"] = self.config.iter_selection_mode
        valid_indices = [idx for idx, item in enumerate(scan) if self._fde._order_result_is_valid(item)]
        generated_has_reward = self.config.candidate_search == "generated" and any(
            item.model.metadata.get("eqgpt_reward") is not None for item in scan
        )
        if generated_has_reward and self.config.eqgpt_selection_mode == "eqgpt_reward":
            for item in scan:
                item.model.metadata["selected_by"] = "maximum_valid_eqgpt_reward"
            search_indices = valid_indices or list(range(len(scan)))
            selected_index = max(
                search_indices,
                key=lambda idx: (
                    float(scan[idx].model.metadata.get("eqgpt_reward", 0.0)),
                    -float(scan[idx].model.metadata.get("objective", scan[idx].objective)),
                ),
            )
        if self.config.iter_selection_mode == "best_objective":
            if not generated_has_reward or self.config.eqgpt_selection_mode == "stridge_objective":
                for item in scan:
                    if generated_has_reward:
                        selected_by = "minimum_valid_generated_stridge_objective"
                    elif self.config.selection_objective == "physical-stridge":
                        selected_by = "minimum_valid_iterative_physical_stridge_objective"
                    elif self.config.selection_objective == "physical-refit":
                        selected_by = "minimum_valid_iterative_physical_refit_objective"
                    else:
                        selected_by = "minimum_valid_iterative_augmented_objective"
                    item.model.metadata["selected_by"] = (
                        selected_by
                    )
                if valid_indices:
                    selected_index = min(
                        valid_indices,
                        key=lambda idx: float(
                            scan[idx].model.metadata.get(
                                "objective",
                                scan[idx].objective,
                            )
                        ),
                    )
                else:
                    selected_index = len(scan) - 1
                    scan[selected_index].model.metadata["selected_by"] = "last_iterative_update_no_valid_candidate"
        elif self.config.iter_selection_mode == "final":
            selected_index = int(scan[-1].model.metadata.get("iteration_final_index", len(scan) - 1))
            selected_index = max(0, min(selected_index, len(scan) - 1))
            for item in scan:
                item.model.metadata["selected_by"] = "final_iterative_update"
        else:
            raise ValueError("iter_selection_mode must be 'best_objective' or 'final'")
        selected = scan[selected_index]
        selected.model.metadata["iteration_selected"] = True
        selected.model.metadata["iteration_selected_index"] = selected_index
        return selected

    def _compute_halpha(self, torch: Any, net: Any, field: GJBaseField, alpha: float) -> np.ndarray:
        input_norm = self._fde._input_normalization_spec(field.training_metadata)
        output_norm = self._fde._output_normalization_spec(field.training_metadata)
        xx_phys, tt_phys = np.meshgrid(field.position, field.time, indexing="xy")
        x_flat = xx_phys.reshape(-1, 1)
        t_flat = tt_phys.reshape(-1, 1)
        total = x_flat.shape[0]
        tau, weights = sp.roots_jacobi(int(self.config.laguerre_nodes), 0.0, -alpha)
        database_tf = torch.zeros((total * int(self.config.laguerre_nodes), 2), dtype=torch.float32)
        transformed_t = t_flat - t_flat / 2.0 * (tau.reshape(1, -1) + 1.0)
        repeated_x = np.repeat(x_flat, int(self.config.laguerre_nodes), axis=1)
        if input_norm["mode"] == "unit_box":
            x_net = (repeated_x - float(input_norm["x_range"][0])) / float(input_norm["x_scale"])
            t_net = (transformed_t - float(input_norm["t_range"][0])) / float(input_norm["t_scale"])
            t_scale = float(input_norm["t_scale"])
        else:
            x_net = repeated_x
            t_net = transformed_t
            t_scale = 1.0
        database_tf[:, 0] = torch.from_numpy(x_net.reshape(-1).astype(np.float32))
        database_tf[:, 1] = torch.from_numpy(t_net.reshape(-1).astype(np.float32))
        database_tf.requires_grad_(True)
        h_tf = self._fde._apply_output_activation(torch, net(database_tf), field.training_metadata)
        if output_norm["mode"] == "unit_interval":
            h_tf = h_tf * float(output_norm["scale"]) + float(output_norm["shift"])
        ht_tf = torch.autograd.grad(outputs=h_tf.sum(), inputs=database_tf, create_graph=True)[0][:, 1:2]
        values = (
            ht_tf.detach().cpu().numpy().reshape(-1, int(self.config.laguerre_nodes)) / float(t_scale)
        )
        return self._legacy_fractional_integral(values, alpha, weights, t_flat).reshape(field.H.shape)

    def _compute_hbeta(self, torch: Any, net: Any, field: GJBaseField, beta: float) -> np.ndarray:
        # Jacobi parameters require 1 - beta > -1 (strictly beta < 2); treat the integer limit by Hxx.
        if beta >= 2.0 - 1.0e-9 or np.isclose(beta, 2.0):
            return field.Hxx.copy()
        input_norm = self._fde._input_normalization_spec(field.training_metadata)
        output_norm = self._fde._output_normalization_spec(field.training_metadata)
        xx_phys, tt_phys = np.meshgrid(field.position, field.time, indexing="xy")
        x_flat = xx_phys.reshape(-1, 1)
        t_flat = tt_phys.reshape(-1, 1)
        total = x_flat.shape[0]
        tau, weights = sp.roots_jacobi(int(self.config.laguerre_nodes), 0.0, 1.0 - beta)
        database_sf = torch.zeros((total * int(self.config.laguerre_nodes), 2), dtype=torch.float32)
        transformed_x = x_flat - x_flat / 2.0 * (tau.reshape(1, -1) + 1.0)
        repeated_t = np.repeat(t_flat, int(self.config.laguerre_nodes), axis=1)
        if input_norm["mode"] == "unit_box":
            x_net = (transformed_x - float(input_norm["x_range"][0])) / float(input_norm["x_scale"])
            t_net = (repeated_t - float(input_norm["t_range"][0])) / float(input_norm["t_scale"])
            x_scale = float(input_norm["x_scale"])
        else:
            x_net = transformed_x
            t_net = repeated_t
            x_scale = 1.0
        database_sf[:, 0] = torch.from_numpy(x_net.reshape(-1).astype(np.float32))
        database_sf[:, 1] = torch.from_numpy(t_net.reshape(-1).astype(np.float32))
        database_sf.requires_grad_(True)
        h_sf = self._fde._apply_output_activation(torch, net(database_sf), field.training_metadata)
        if output_norm["mode"] == "unit_interval":
            h_sf = h_sf * float(output_norm["scale"]) + float(output_norm["shift"])
        hx_sf = torch.autograd.grad(outputs=h_sf.sum(), inputs=database_sf, create_graph=True)[0][:, 0:1]
        hxx_sf = torch.autograd.grad(outputs=hx_sf.sum(), inputs=database_sf, create_graph=True)[0][:, 0:1]
        values = (
            hxx_sf.detach().cpu().numpy().reshape(-1, int(self.config.laguerre_nodes))
            / float(x_scale**2)
        )
        return self._legacy_fractional_integral(values, beta, weights, x_flat).reshape(field.H.shape)

    @staticmethod
    def _legacy_fractional_integral(values: np.ndarray, order: float, weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
        gamma_term = sp.gamma(2.0 - order) if order > 1.0 else sp.gamma(1.0 - order)
        exponent = 2.0 - order if order > 1.0 else 1.0 - order
        return np.matmul(values, weights.reshape(-1, 1)) / gamma_term * (scale / 2.0) ** exponent

    def _alpha_derivative(self, alpha_cache: dict[float, np.ndarray], alpha0: float) -> np.ndarray:
        step = self._alpha_step(alpha0)
        upper = alpha_cache[round(alpha0 + step, 10)]
        lower = alpha_cache[round(alpha0 - step, 10)]
        return (upper - lower) / (2.0 * step)

    def _beta_derivative(self, beta_cache: dict[float, np.ndarray], beta0: float) -> np.ndarray:
        step = self._beta_step(beta0)
        if np.isclose(beta0, 2.0):
            lower = beta_cache[round(beta0 - step, 10)]
            return (beta_cache[round(beta0, 10)] - lower) / step
        upper = beta_cache[round(beta0 + step, 10)]
        lower = beta_cache[round(beta0 - step, 10)]
        return (upper - lower) / (2.0 * step)

    def _get_or_compute_halpha(
        self,
        torch: Any,
        net: Any,
        field: GJBaseField,
        cache: dict[float, np.ndarray],
        alpha: float,
    ) -> np.ndarray:
        key = round(float(alpha), 10)
        if key not in cache:
            cache[key] = self._compute_halpha(torch, net, field, float(alpha))
        return cache[key]

    def _get_or_compute_hbeta(
        self,
        torch: Any,
        net: Any,
        field: GJBaseField,
        cache: dict[float, np.ndarray],
        beta: float,
    ) -> np.ndarray:
        key = round(float(beta), 10)
        if key not in cache:
            cache[key] = self._compute_hbeta(torch, net, field, float(beta))
        return cache[key]

    def _prepare_iter_alpha_cache(
        self,
        torch: Any,
        net: Any,
        field: GJBaseField,
        cache: dict[float, np.ndarray],
        alpha0: float,
    ) -> None:
        step = self._alpha_step(alpha0)
        for alpha in (alpha0 - step, alpha0, alpha0 + step):
            if 0.0 < float(alpha) < 1.0:
                self._get_or_compute_halpha(torch, net, field, cache, float(alpha))

    def _prepare_iter_beta_cache(
        self,
        torch: Any,
        net: Any,
        field: GJBaseField,
        cache: dict[float, np.ndarray],
        beta0: float,
    ) -> None:
        step = self._beta_step(beta0)
        values = [beta0 - step, beta0] if np.isclose(beta0, 2.0) else [beta0 - step, beta0, beta0 + step]
        for beta in values:
            if 1.0 < float(beta) <= 2.0:
                self._get_or_compute_hbeta(torch, net, field, cache, float(beta))

    @staticmethod
    def _clip_iter_alpha(alpha: float) -> float:
        return float(np.clip(float(alpha), 0.01, 0.999))

    @staticmethod
    def _clip_iter_beta(beta: float) -> float:
        return float(np.clip(float(beta), 1.01, 2.0))

    @staticmethod
    def _alpha_step(alpha0: float) -> float:
        room_left = max(float(alpha0), 0.0)
        room_right = max(1.0 - float(alpha0), 0.0)
        step = min(0.05, 0.45 * room_left, 0.45 * room_right)
        if step <= 0.0:
            raise ValueError(f"alpha step is not positive for alpha0={alpha0}")
        return max(step, 1.0e-8)

    @staticmethod
    def _beta_step(beta0: float, default: float = 2.0e-2) -> float:
        max_step = 0.45 * (beta0 - 1.0)
        if beta0 < 2.0:
            max_step = min(max_step, 0.45 * (2.0 - beta0))
        step = min(float(default), max_step)
        if step <= 0.0:
            raise ValueError(f"beta step is not positive for beta0={beta0}")
        return step

    @staticmethod
    def _build_physical_terms(field: GJBaseField, hbeta: np.ndarray, beta: float) -> tuple[dict[str, np.ndarray], list[str]]:
        h = field.H
        hx = field.Hx
        hxx = field.Hxx
        hxxx = field.Hxxx
        frac_name = f"D_x^{beta:.7g} H"
        terms: dict[str, np.ndarray] = {
            "1": np.ones_like(h),
            "H": h,
            "Hx": hx,
            frac_name: hbeta,
            "Hxxx": hxxx,
            "H^2": h * h,
            "H*Hx": h * hx,
            "H*Hxx": h * hxx,
            "H*Hxxx": h * hxxx,
            "H^2*Hx": h * h * hx,
            "H^2*Hxx": h * h * hxx,
            "H^2*Hxxx": h * h * hxxx,
        }
        order = ["1", "H", "Hx", frac_name, "Hxxx", "H^2", "H*Hx", "H*Hxx", "H*Hxxx", "H^2*Hx", "H^2*Hxx", "H^2*Hxxx"]
        return terms, order
