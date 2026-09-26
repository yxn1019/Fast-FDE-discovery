from __future__ import annotations

from typing import Any

import numpy as np

from .fractional_discoverer import FractionalDiscoveryResult


def _prettify_term(term: str) -> str:
    return {
        "u": "H",
        "u_x": "Hx",
        "u_xx": "Hxx",
        "u_xxx": "Hxxx",
        "u*u_x": "H*Hx",
        "u*u_xx": "H*Hxx",
        "u^2*u_x": "H^2*Hx",
        "u^2*u_xx": "H^2*Hxx",
    }.get(term, term)


def _format_number(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) >= 100 or (abs(value) > 0 and abs(value) < 1e-4):
        return f"{value:.3e}"
    return f"{value:.{precision}g}"


def _relative_error(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None or truth == 0:
        return None
    return abs(float(value) - float(truth)) / abs(float(truth))

def _render_equation(
    result: FractionalDiscoveryResult,
    *,
    include_corrections: bool = True,
    numeric_orders: bool = False,
) -> str:
    parts: list[str] = []
    metadata = result.model.metadata
    display_beta = metadata.get("spatial_beta")
    for coefficient, name in zip(result.model.coefficients, result.model.term_names):
        if np.isclose(coefficient, 0.0):
            continue
        if not include_corrections and (
            name == "alpha_correction" or name.startswith("spatial_correction_beta0=")
        ):
            continue
        if (
            not include_corrections
            and name.startswith("spatial_main_beta0=")
            and isinstance(display_beta, (int, float))
        ):
            term = f"D_x^{display_beta:.6g} H"
        else:
            term = _prettify_term(name)
        connector = " " if term.startswith("D_x^") else "*"
        parts.append(f"{coefficient:.4g}{connector}{term}")
    rhs = " + ".join(parts) if parts else "0"
    lhs_order = _format_number(result.alpha, 8) if numeric_orders else "alpha"
    return f"D_t^{lhs_order} H = {rhs}"


def _render_augmented_equation(result: FractionalDiscoveryResult) -> str:
    metadata = result.model.metadata
    terms = metadata.get("internal_augmented_terms")
    coefficients = metadata.get("internal_augmented_coefficients")
    if not terms or not coefficients:
        return _render_equation(result, include_corrections=True, numeric_orders=True)
    parts: list[str] = []
    for coefficient, name in zip(coefficients, terms):
        if np.isclose(coefficient, 0.0):
            continue
        term = _prettify_term(str(name))
        connector = " " if term.startswith("D_x^") else "*"
        parts.append(f"{float(coefficient):.4g}{connector}{term}")
    rhs = " + ".join(parts) if parts else "0"
    return f"D_t^{_format_number(result.alpha, 8)} H = {rhs}"


def format_console_result(result: FractionalDiscoveryResult) -> str:
    metadata = result.model.metadata
    summary = f"alpha = {result.alpha:.8g}"
    if metadata["spatial_fractional_enabled"]:
        beta_value = metadata["spatial_beta"]
        beta_text = f"{beta_value:.8g}" if isinstance(beta_value, (int, float)) else str(beta_value)
        summary += f", beta = {beta_text} (status={metadata['spatial_beta_status']})"
    center_line = (
        f"center orders: alpha0 = {_format_number(metadata.get('alpha0'))}, "
        f"beta0 = {_format_number(metadata.get('beta0'))}"
    )
    order_update_line = f"order update: mode={metadata.get('order_update_mode', 'grid')}"
    if metadata.get("order_update_mode") == "iterative":
        order_update_line += (
            f", iterations={metadata.get('iteration_count')}, "
            f"converged={metadata.get('iteration_converged')}, "
            f"stop={metadata.get('iteration_stop_reason')}, "
            f"selected={metadata.get('iteration_selected_index')}"
        )
    return "\n".join(
        [
            f"example: {metadata.get('example', 'unknown')}",
            f"checkpoint: {metadata['checkpoint']}",
            f"candidate equation: {_render_equation(result, include_corrections=False, numeric_orders=True)}",
            center_line,
            order_update_line,
            summary,
        ]
    )


def format_detailed_result(result: FractionalDiscoveryResult) -> str:
    metadata = result.model.metadata
    true_alpha = metadata.get("true_alpha")
    true_beta = metadata.get("true_beta")
    reference_lines: list[str] = []
    if metadata.get("gamma_shape") is not None:
        reference_lines.append(f"gamma_shape: {metadata.get('gamma_shape')}")
    if metadata.get("gamma_scale") is not None:
        reference_lines.append(f"gamma_scale: {metadata.get('gamma_scale')}")
    if metadata.get("mechanistic_reference_note") is not None:
        reference_lines.append(f"mechanistic_reference_note: {metadata.get('mechanistic_reference_note')}")
    truth_lines: list[str] = []
    if true_alpha is not None:
        truth_lines.extend(
            [
                f"true_alpha: {true_alpha}",
                f"alpha_relative_error: {_relative_error(result.alpha, true_alpha)}",
            ]
        )
    if true_beta is not None:
        truth_lines.extend(
            [
                f"true_beta: {true_beta}",
                f"beta_relative_error: {_relative_error(metadata.get('spatial_beta'), true_beta)}",
            ]
        )
    active_lines = [
        f"  {name}: {coefficient:.10g}"
        for name, coefficient in zip(result.model.term_names, result.model.coefficients)
        if not np.isclose(coefficient, 0.0)
    ]
    scan_title = "alpha0/beta0 scan:"
    scan_header = (
        "  alpha0,beta0,delta_alpha,beta_correction,alpha,beta,"
        "alpha_rel_error,beta_rel_error,mse,objective,status,active"
    )
    scan_lines = [
        (
            "  "
            f"{_format_number(item.model.metadata.get('alpha0'))},"
            f"{_format_number(item.model.metadata.get('beta0'))},"
            f"{_format_number(item.model.metadata.get('delta_alpha'))},"
            f"{_format_number(item.model.metadata.get('spatial_beta_correction'))},"
            f"{_format_number(item.alpha)},"
            f"{_format_number(item.model.metadata.get('spatial_beta'))},"
            f"{_relative_error(item.alpha, true_alpha)},"
            f"{_relative_error(item.model.metadata.get('spatial_beta'), true_beta)},"
            f"{_format_number(item.model.mse)},"
            f"{_format_number(item.model.metadata.get('objective'))},"
            f"{item.model.metadata.get('spatial_beta_status')},"
            f"{' + '.join(item.model.support_names) if item.model.support_names else 'none'}"
        )
        for item in result.alpha_scan
    ]
    iter_trace = metadata.get("iteration_trace") or ()
    iter_trace_lines = [
        (
            "  "
            f"{row.get('iter')},"
            f"{_format_number(row.get('alpha0'))},"
            f"{_format_number(row.get('beta0'))},"
            f"{_format_number(row.get('delta_alpha_raw'))},"
            f"{_format_number(row.get('delta_beta_raw'))},"
            f"{_format_number(row.get('delta_alpha'))},"
            f"{_format_number(row.get('delta_beta'))},"
            f"{_format_number(row.get('delta_beta_used'))},"
            f"{_format_number(row.get('alpha_next'))},"
            f"{_format_number(row.get('beta_next'))},"
            f"{_format_number(row.get('objective'))},"
            f"{_format_number(row.get('regularized_objective'))},"
            f"{_format_number(row.get('raw_mse'))},"
            f"{_format_number(row.get('mse'))},"
            f"{_format_number(row.get('condition'))},"
            f"{_format_number(row.get('validation_residual_norm'))},"
            f"{_format_number(row.get('complexity_penalty'))},"
            f"{_format_number(row.get('active_lamb_sum'))},"
            f"{_format_number(row.get('physical_refit_condition_penalty'))},"
            f"{row.get('active_count_for_loss')},"
            f"{_format_number(row.get('tolerance'))},"
            f"{row.get('spatial_beta_status')},"
            f"{row.get('support')}"
        )
        for row in iter_trace
        if isinstance(row, dict)
    ]
    iter_trace_block: list[str] = []
    if metadata.get("order_update_mode") == "iterative":
        iter_trace_block = [
            "iterative order update trace:",
            "  iter,alpha0,beta0,delta_alpha_raw,delta_beta_raw,delta_alpha,delta_beta,delta_beta_used,alpha_next,beta_next,objective,regularized_objective,raw_mse,mse,condition,validation_residual_norm,complexity_penalty,active_lamb_sum,physical_refit_condition_penalty,active_count_for_loss,tolerance,status,support",
            *(iter_trace_lines or ["  none"]),
        ]
    generated_summaries = metadata.get("generated_candidate_summaries") or ()
    generated_summary_lines = [
        (
            "  "
            f"{row.get('epoch')},"
            f"{row.get('rank')}," 
            f"{row.get('selected')}," 
            f"{row.get('structure')}," 
            f"{row.get('raw_structure')}," 
            f"{row.get('eqgpt_source_equation')},"
            f"{_format_number(row.get('eqgpt_reward'))},"
            f"{_format_number(row.get('eqgpt_r2'))},"
            f"{_format_number(row.get('complexity_score'))},"
            f"{_format_number(row.get('structure_penalty'))},"
            f"{_format_number(row.get('endpoint_penalty'))},"
            f"{_format_number(row.get('iter_start_beta'))},"
            f"{_format_number(row.get('alpha'))},"
            f"{_format_number(row.get('beta'))},"
            f"{_format_number(row.get('coef_hx'))},"
            f"{_format_number(row.get('coef_dbeta'))},"
            f"{_format_number(row.get('objective'))},"
            f"{_format_number(row.get('augmented_objective'))},"
            f"{_format_number(row.get('validation_residual_norm'))},"
            f"{_format_number(row.get('condition_number'))},"
            f"{row.get('support')}"
        )
        for row in generated_summaries
        if isinstance(row, dict)
    ]
    generated_summary_block: list[str] = []
    if metadata.get("candidate_search") == "generated":
        generated_summary_block = [
            "generated candidate summary:",
            "  epoch,rank,selected,structure,raw_structure,eqgpt_source_equation,eqgpt_reward,eqgpt_r2,complexity_score,structure_penalty,endpoint_penalty,iter_start_beta,alpha,beta,coef_hx,coef_dbeta,objective,augmented_objective,validation_residual_norm,condition,support",
            *(generated_summary_lines or ["  none"]),
        ]
    return "\n".join(
        [
            f"case: {metadata['case']}",
            f"route: {metadata['route']}",
            f"example: {metadata.get('example')}",
            f"paper_example: {metadata.get('paper_example')}",
            f"preset_example: {metadata.get('preset_example')}",
            f"example_source: {metadata.get('example_source')}",
            f"example_description: {metadata.get('example_description')}",
            f"selected_by: {metadata.get('selected_by')}",
            f"checkpoint: {metadata['checkpoint']}",
            f"training metadata: {metadata.get('training_metadata')}",
            f"model_file: {metadata['model_file']}",
            f"trained_point: {metadata['trained_point']}",
            f"noise_level: {metadata['noise_level']}",
            f"activation: {metadata['activation']}",
            f"hidden_layers: {metadata['hidden_layers']}",
            f"neurons: {metadata['neurons']}",
            *reference_lines,
            f"NN grid shape: {metadata['grid_shape']}",
            f"x_range: {metadata['x_range']}",
            f"t_range: {metadata['t_range']}",
            f"operator_grid_shape: {metadata.get('operator_grid_shape')}",
            f"operator_x_range: {metadata.get('operator_x_range')}",
            f"operator_t_range: {metadata.get('operator_t_range')}",
            f"fit_grid_shape: {metadata.get('fit_grid_shape')}",
            f"fit_x_range: {metadata.get('fit_x_range')}",
            f"fit_t_range: {metadata.get('fit_t_range')}",
            f"fit_window_config: {metadata.get('fit_window_config')}",
            f"initial_condition_time: {metadata['initial_condition_time']}",
            f"initial_condition_note: {metadata['initial_condition_note']}",
            f"library terms: {', '.join(metadata['term_names'])}",
            f"candidate_search: {metadata.get('candidate_search')}",
            f"generated_structure: {metadata.get('generated_structure')}",
            f"generated_raw_structure: {metadata.get('generated_raw_structure')}",
            f"generated_normalized_from: {metadata.get('generated_normalized_from')}",
            f"generated_structure_has_fractional: {metadata.get('generated_structure_has_fractional')}",
            f"generated_structure_rank: {metadata.get('generated_structure_rank')}",
            f"generated_complexity_score: {metadata.get('generated_complexity_score')}",
            f"generated_structure_objective_penalty: {metadata.get('generated_structure_objective_penalty')}",
            f"generated_structure_count: {metadata.get('generated_structure_count')}",
            f"eqgpt_checkpoint: {metadata.get('eqgpt_checkpoint')}",
            f"eqgpt_dictionary: {metadata.get('eqgpt_dictionary')}",
            f"eqgpt_epoch: {metadata.get('eqgpt_epoch')}",
            f"eqgpt_sample_count: {metadata.get('eqgpt_sample_count')}",
            f"eqgpt_unique_structure_count: {metadata.get('eqgpt_unique_structure_count')}",
            f"eqgpt_optimize_epochs: {metadata.get('eqgpt_optimize_epochs')}",
            f"eqgpt_finetune_epochs: {metadata.get('eqgpt_finetune_epochs')}",
            f"eqgpt_learning_rate: {metadata.get('eqgpt_learning_rate')}",
            f"eqgpt_reward_sparsity_alpha: {metadata.get('eqgpt_reward_sparsity_alpha')}",
            f"eqgpt_selection_mode: {metadata.get('eqgpt_selection_mode')}",
            f"eqgpt_random_exploration: {metadata.get('eqgpt_random_exploration')}",
            f"eqgpt_reward: {metadata.get('eqgpt_reward')}",
            f"eqgpt_r2: {metadata.get('eqgpt_r2')}",
            f"eqgpt_source_equation: {metadata.get('eqgpt_source_equation')}",
            *generated_summary_block,
            f"matrix_shape: {metadata['matrix_shape']}",
            f"alpha0_count: {metadata['alpha0_count']}",
            f"best alpha0: {metadata['alpha0']}",
            f"best beta0: {metadata.get('beta0')}",
            f"delta alpha: {metadata['delta_alpha']}",
            f"delta alpha raw: {metadata.get('delta_alpha_raw')}",
            f"alpha: {result.alpha}",
            *truth_lines,
            f"s_range: {metadata['s_range']}",
            f"s_points: {metadata['s_points']}",
            f"laguerre_nodes: {metadata['laguerre_nodes']}",
            f"space_derivative_mode: {metadata.get('space_derivative_mode')}",
            f"spatial_fractional_mode: {metadata.get('spatial_fractional_mode')}",
            f"time_operator_mode: {metadata.get('time_operator_mode')}",
            f"laplace_out_of_bounds_fraction: {metadata['laplace_out_of_bounds_fraction']}",
            f"ridge_lambda: {metadata['ridge_lambda']}",
            f"d_tol: {metadata['d_tol']}",
            f"maxit: {metadata['maxit']}",
            f"STR_iters: {metadata['STR_iters']}",
            f"normalize: {metadata['normalize']}",
            f"split: {metadata['split']}",
            f"sparsity_lamb: {metadata['sparsity_lamb']}",
            f"fractional_correction_sparsity_lamb: {metadata['fractional_correction_sparsity_lamb']}",
            f"fractional_correction_tol_scale: {metadata['fractional_correction_tol_scale']}",
            f"alpha_correction_tol: {metadata.get('alpha_correction_tol')}",
            f"beta_correction_tol: {metadata.get('beta_correction_tol')}",
            f"delta_alpha_prune_threshold: {metadata['delta_alpha_prune_threshold']}",
            f"delta_alpha_bounds: {metadata['delta_alpha_bounds']}",
            f"delta_beta_prune_threshold: {metadata['delta_beta_prune_threshold']}",
            f"loss_formula: {metadata['loss_formula']}",
            f"validation_residual_norm: {metadata['validation_residual_norm']}",
            f"physical_refit: {metadata.get('physical_refit', False)}",
            f"physical_refit_note: {metadata.get('physical_refit_note')}",
            f"physical_refit_terms: {metadata.get('physical_refit_terms')}",
            f"physical_stridge_library_terms: {metadata.get('physical_stridge_library_terms')}",
            f"active_count_for_loss: {metadata['active_count_for_loss']}",
            f"complexity_penalty: {metadata['complexity_penalty']}",
            f"l0_penalty: {metadata['l0_penalty']}",
            f"best tolerance: {metadata['tolerance']}",
            f"augmented_stridge_objective_raw: {metadata.get('augmented_stridge_objective_raw')}",
            f"selection_loss_formula: {metadata.get('selection_loss_formula')}",
            f"objective: {metadata['objective']}",
            f"selection_objective_mode: {metadata.get('selection_objective_mode')}",
            f"refit_mode: {metadata.get('refit_mode')}",
            f"order_update_mode: {metadata.get('order_update_mode')}",
            f"iteration_count: {metadata.get('iteration_count')}",
            f"iteration_converged: {metadata.get('iteration_converged')}",
            f"iteration_stop_reason: {metadata.get('iteration_stop_reason')}",
            f"iteration_final_index: {metadata.get('iteration_final_index')}",
            f"iteration_best_objective_index: {metadata.get('iteration_best_objective_index')}",
            f"iteration_best_fractional_support_index: {metadata.get('iteration_best_fractional_support_index')}",
            f"iteration_selected_index: {metadata.get('iteration_selected_index')}",
            f"iteration_selected: {metadata.get('iteration_selected')}",
            f"iter_selection_mode: {metadata.get('iter_selection_mode')}",
            f"iter_start_alpha: {metadata.get('iter_start_alpha')}",
            f"iter_start_beta: {metadata.get('iter_start_beta')}",
            f"iter_start_beta_values: {metadata.get('iter_start_beta_values')}",
            f"iter_order_tol: {metadata.get('iter_order_tol')}",
            f"iter_damping: {metadata.get('iter_damping')}",
            f"iter_max_step_alpha: {metadata.get('iter_max_step_alpha')}",
            f"iter_max_step_beta: {metadata.get('iter_max_step_beta')}",
            f"iter_delta_alpha_bounds: {metadata.get('iter_delta_alpha_bounds')}",
            f"iter_delta_beta_bounds: {metadata.get('iter_delta_beta_bounds')}",
            f"iter_stop_on_non_decreasing_objective: {metadata.get('iter_stop_on_non_decreasing_objective')}",
            f"iter_objective_min_delta: {metadata.get('iter_objective_min_delta')}",
            f"physical_refit_selection_objective: {metadata.get('physical_refit_selection_objective')}",
            f"physical_refit_selection_residual_norm: {metadata.get('physical_refit_selection_residual_norm')}",
            f"nonlinear_refit_success: {metadata.get('nonlinear_refit_success')}",
            f"nonlinear_refit_cost: {metadata.get('nonlinear_refit_cost')}",
            f"nonlinear_refit_nfev: {metadata.get('nonlinear_refit_nfev')}",
            f"nonlinear_refit_status: {metadata.get('nonlinear_refit_status')}",
            f"nonlinear_refit_message: {metadata.get('nonlinear_refit_message')}",
            f"order_radius_mode: {metadata.get('order_radius_mode')}",
            f"order_radius_penalty: {metadata.get('order_radius_penalty')}",
            f"mse: {result.model.mse}",
            f"residual_norm: {result.model.residual_norm}",
            f"used alpha correction: {'alpha_correction' in result.model.support_names}",
            f"spatial_fractional_enabled: {metadata['spatial_fractional_enabled']}",
            f"spatial_correction_active: {metadata['spatial_correction_active']}",
            f"spatial_beta_status: {metadata['spatial_beta_status']}",
            f"spatial_beta: {metadata['spatial_beta']}",
            f"spatial_delta_beta: {metadata['spatial_delta_beta']}",
            f"spatial_delta_beta_raw: {metadata.get('spatial_delta_beta_raw')}",
            f"spatial_beta_correction: {metadata['spatial_beta_correction']}",
            f"spatial_beta_in_bounds: {metadata['spatial_beta_in_bounds']}",
            f"spatial_beta_bounds: {metadata['spatial_beta_bounds']}",
            f"spatial_beta_model_valid: {metadata['spatial_beta_model_valid']}",
            f"spatial_log_quad_points: {metadata['spatial_log_quad_points']}",
            f"spatial_log_quad_method: {metadata['spatial_log_quad_method']}",
            f"spatial_correction_method: {metadata['spatial_correction_method']}",
            f"spatial_correction_pruned_without_main: {metadata.get('spatial_correction_pruned_without_main')}",
            f"physical_refit_spatial_fractional_mode: {metadata.get('physical_refit_spatial_fractional_mode')}",
            f"physical_refit_space_derivative_mode: {metadata.get('physical_refit_space_derivative_mode')}",
            f"physical_refit_time_operator_mode: {metadata.get('physical_refit_time_operator_mode')}",
            f"spatial_boundary_mode: {metadata['spatial_boundary_mode']}",
            f"spatial_boundary_value: {metadata['spatial_boundary_value']}",
            f"spatial_operator_note: {metadata['spatial_operator_note']}",
            f"end_to_end_seconds: {metadata.get('end_to_end_seconds')}",
            f"timing_surrogate_load_seconds: {metadata.get('timing_surrogate_load_seconds')}",
            f"timing_operator_field_seconds: {metadata.get('timing_operator_field_seconds')}",
            f"timing_order_search_seconds: {metadata.get('timing_order_search_seconds')}",
            f"timing_fractional_operator_seconds: {metadata.get('timing_fractional_operator_seconds')}",
            f"timing_sparse_regression_seconds: {metadata.get('timing_sparse_regression_seconds')}",
            f"timing_order_update_overhead_seconds: {metadata.get('timing_order_update_overhead_seconds')}",
            f"candidate equation: {_render_equation(result, include_corrections=False, numeric_orders=True)}",
            f"augmented regression equation: {_render_augmented_equation(result)}",
            "active terms:",
            *(active_lines or ["  none"]),
            *iter_trace_block,
            scan_title,
            scan_header,
            *scan_lines,
        ]
    )
