"""Run a fair DE-legacy benchmark on the normalized tsfade paper surrogate.

The original previous-project script hard-codes an old checkpoint layout.  This
adapter keeps the legacy idea--DE over (alpha, beta) with STRidge support
selection at every trial order--but loads the current paper normalized tanh
checkpoints and uses the same fit windows as the proposed method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_case_params import MODEL_PRESETS, PAPER_EXAMPLES, make_config
from src.transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer


TRUE_ALPHA = 0.78
TRUE_BETA = 1.83
TRUE_HX = -1.0
TRUE_DBETA = 0.5

DEFAULT_CASES = {
    0.0: ("clean", "tsfade_clean"),
    5.0: ("5%", "tsfade_noise5"),
    25.0: ("25%", "tsfade_noise25"),
}


@dataclass
class LegacyFitResult:
    coefficients: np.ndarray
    objective: float
    tolerance: float
    condition: float
    validation_residual_norm: float
    mse: float
    support: tuple[str, ...]
    coef_hx: float | None
    coef_dbeta: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-levels", default="0,5,25", help="comma-separated noise levels from 0,5,25")
    parser.add_argument(
        "--stridge-mode",
        choices=("legacy", "same_hyperparams", "same_stridge_core"),
        default="legacy",
        help=(
            "STRidge fairness mode: legacy keeps old fSTRidge-style settings; "
            "same_hyperparams uses paper-example lamb/d_tol in the legacy wrapper; "
            "same_stridge_core calls the mainline _fit_fde_stridge wrapper without correction columns"
        ),
    )
    parser.add_argument("--maxiter", type=int, default=15, help="differential_evolution maxiter")
    parser.add_argument("--popsize", type=int, default=15, help="differential_evolution population multiplier")
    parser.add_argument("--tol", type=float, default=1.0e-4, help="differential_evolution tolerance")
    parser.add_argument("--seed", type=int, default=525, help="random seed for DE and STRidge train/test split")
    parser.add_argument("--lamb", type=float, default=1.0e-3, help="legacy l0 penalty multiplier")
    parser.add_argument(
        "--core-lamb-override",
        type=float,
        default=None,
        help=(
            "diagnostic override for the mainline STRidge sparsity lamb when "
            "--stridge-mode same_stridge_core is used"
        ),
    )
    parser.add_argument(
        "--core-d-tol-override",
        type=float,
        default=None,
        help="diagnostic override for the mainline STRidge d_tol in same_stridge_core mode",
    )
    parser.add_argument("--ridge-lam", type=float, default=2.0, help="legacy STRidge ridge lambda")
    parser.add_argument("--d-tol", type=float, default=0.1, help="legacy tolerance increment")
    parser.add_argument("--maxit", type=int, default=25, help="legacy tolerance-search iterations")
    parser.add_argument("--str-iters", type=int, default=10, help="legacy STRidge pruning iterations")
    parser.add_argument("--split", type=float, default=0.8, help="legacy train split for objective")
    parser.add_argument("--normalize", type=int, default=0, help="legacy STRidge normalization flag")
    parser.add_argument("--quiet", action="store_true", help="suppress per-trial DE logging")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "legacy_de_stridge_alignment_diagnostic",
    )
    parser.add_argument("--no-trace-csv", action="store_true", help="only write per-noise trace JSON")
    return parser.parse_args()


def parse_noise_levels(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in DEFAULT_CASES:
            raise ValueError(f"unsupported noise level {value}; choose from 0,5,25")
        values.append(value)
    if not values:
        raise ValueError("at least one noise level is required")
    return values


def finite_condition(matrix: np.ndarray) -> float:
    condition = float(np.linalg.cond(np.asarray(matrix, dtype=float)))
    return condition if math.isfinite(condition) else 1.0e12


def norm2(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(array * array)))


def legacy_stridge(X0: np.ndarray, y: np.ndarray, lam: float, maxit: int, tol: float, normalize: int) -> np.ndarray:
    n, d = X0.shape
    if normalize != 0:
        X = np.zeros((n, d), dtype=float)
        Mreg = np.zeros((d, 1), dtype=float)
        for i in range(d):
            col_norm = np.linalg.norm(X0[:, i], normalize)
            Mreg[i] = 1.0 / max(float(col_norm), np.finfo(float).eps)
            X[:, i] = (Mreg[i] * X0[:, i : i + 1]).reshape(-1)
    else:
        X = X0
        Mreg = np.ones((d, 1), dtype=float)

    if lam != 0:
        w = np.linalg.lstsq(X.T @ X + lam * np.eye(d), X.T @ y, rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where(np.abs(w.reshape(-1)) > tol)[0].tolist()

    for _ in range(maxit):
        smallinds = np.where(np.abs(w.reshape(-1)) < tol)[0]
        new_biginds = [i for i in range(d) if i not in set(smallinds.tolist())]
        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)
        if len(new_biginds) == 0:
            if not biginds:
                return np.multiply(Mreg, w) if normalize != 0 else w
            break
        biginds = new_biginds
        w[smallinds] = 0.0
        if lam != 0:
            w[biginds] = np.linalg.lstsq(
                X[:, biginds].T @ X[:, biginds] + lam * np.eye(len(biginds)),
                X[:, biginds].T @ y,
                rcond=None,
            )[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if biginds:
        w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]
    return np.multiply(Mreg, w) if normalize != 0 else w


def legacy_fstridge_fit(
    matrix: np.ndarray,
    target: np.ndarray,
    term_names: tuple[str, ...],
    *,
    seed: int,
    ridge_lam: float,
    d_tol: float,
    maxit: int,
    str_iters: int,
    normalize: int,
    split: float,
    lamb: float,
) -> LegacyFitResult:
    rng = np.random.default_rng(seed)
    n_rows = matrix.shape[0]
    train_count = int(n_rows * split)
    train = rng.choice(n_rows, train_count, replace=False)
    test_mask = np.ones(n_rows, dtype=bool)
    test_mask[train] = False
    test = np.flatnonzero(test_mask)
    train_matrix = matrix[train, :]
    test_matrix = matrix[test, :]
    train_target = target[train, :]
    test_target = target[test, :]

    condition = finite_condition(matrix)
    l0_penalty = float(lamb * condition)
    tol = float(d_tol)
    best_tol = float(d_tol)
    best_weights = np.linalg.lstsq(train_matrix, train_target, rcond=None)[0]
    best_error = norm2(test_target - test_matrix @ best_weights) + l0_penalty * int(np.count_nonzero(best_weights))

    current_d_tol = float(d_tol)
    for iteration in range(maxit):
        weights = legacy_stridge(matrix, target, ridge_lam, str_iters, tol, normalize=normalize)
        error = norm2(test_target - test_matrix @ weights) + l0_penalty * int(np.count_nonzero(weights))
        if error <= best_error:
            best_error = float(error)
            best_weights = weights
            best_tol = float(tol)
            tol += current_d_tol
        else:
            tol = max(0.0, tol - 2.0 * current_d_tol)
            current_d_tol = 2.0 * current_d_tol / max(maxit - iteration, 1)
            tol += current_d_tol

    residual = target - matrix @ best_weights
    active_indices = [idx for idx, value in enumerate(best_weights.reshape(-1)) if abs(float(value)) > 1.0e-12]
    support = tuple(term_names[idx] for idx in active_indices)
    coef_hx = None
    coef_dbeta = None
    for name, value in zip(term_names, best_weights.reshape(-1)):
        if name == "Hx":
            coef_hx = float(value)
        if name.startswith("D_x^"):
            coef_dbeta = float(value)
    return LegacyFitResult(
        coefficients=best_weights.reshape(-1),
        objective=float(best_error),
        tolerance=float(best_tol),
        condition=float(condition),
        validation_residual_norm=norm2(residual),
        mse=float(np.mean(residual**2)),
        support=support,
        coef_hx=coef_hx,
        coef_dbeta=coef_dbeta,
    )


def mainline_stridge_fit(
    discoverer: GJHybridDiscoverer,
    matrix: np.ndarray,
    target: np.ndarray,
    term_names: tuple[str, ...],
) -> LegacyFitResult:
    coefficients, objective, tolerance, l0_penalty, details = discoverer._fde._fit_fde_stridge(
        matrix,
        target,
        protected_indices=(),
        term_names=term_names,
    )
    coefficients = coefficients.reshape(-1)
    residual = target - matrix @ coefficients.reshape(-1, 1)
    active_indices = [idx for idx, value in enumerate(coefficients) if abs(float(value)) > 1.0e-12]
    support = tuple(term_names[idx] for idx in active_indices)
    coef_hx = None
    coef_dbeta = None
    for name, value in zip(term_names, coefficients):
        if name == "Hx":
            coef_hx = float(value)
        if name.startswith("D_x^"):
            coef_dbeta = float(value)
    condition = float(details.get("condition_number", finite_condition(matrix)))
    return LegacyFitResult(
        coefficients=coefficients,
        objective=float(objective),
        tolerance=float(tolerance),
        condition=condition,
        validation_residual_norm=float(details["validation_residual_norm"]),
        mse=float(np.mean(residual**2)),
        support=support,
        coef_hx=coef_hx,
        coef_dbeta=coef_dbeta,
    )


def config_for_paper_example(paper_example: str):
    paper = PAPER_EXAMPLES[paper_example]
    preset = MODEL_PRESETS[paper["example"]]
    kwargs: dict[str, Any] = {
        "case_name": paper["case"],
        "checkpoint_file": preset["checkpoint_file"],
        "model_alpha_tag": preset["alpha_tag"],
        "trained_point": int(preset["trained_point"]),
        "noise_level": float(preset["noise_level"]),
        "activation": str(preset["activation"]),
        "hidden_layers": int(preset["hidden_layers"]),
        "neurons": int(preset["neurons"]),
        "enable_spatial_fractional": bool(preset["enable_spatial_fractional"]),
        "order_update_mode": "grid",
        "refit_mode": "none",
    }
    for key in ("fit_x_min", "fit_x_max", "fit_t_min", "fit_t_max", "sparsity_lamb", "d_tol"):
        if key in paper:
            kwargs[key] = paper[key]
    return make_config(**kwargs)


def fit_values_for_orders(
    discoverer: GJHybridDiscoverer,
    torch: Any,
    net: Any,
    operator_field: Any,
    fit_field: Any,
    alpha_cache: dict[float, np.ndarray],
    beta_cache: dict[float, np.ndarray],
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    alpha_key = round(float(alpha), 10)
    beta_key = round(float(beta), 10)
    if alpha_key not in alpha_cache:
        alpha_cache[alpha_key] = discoverer._compute_halpha(torch, net, operator_field, float(alpha))
    if beta_key not in beta_cache:
        beta_cache[beta_key] = discoverer._compute_hbeta(torch, net, operator_field, float(beta))
    halpha = discoverer._restrict_values(alpha_cache[alpha_key], operator_field, fit_field)
    hbeta = discoverer._restrict_values(beta_cache[beta_key], operator_field, fit_field)
    terms, order = discoverer._build_physical_terms(fit_field, hbeta, float(beta))
    matrix = np.column_stack([terms[name].reshape(-1) for name in order])
    target = halpha.reshape(-1, 1)
    return matrix, target, tuple(order)


def support_is_correct(support: tuple[str, ...], beta: float, coef_dbeta: float | None) -> bool:
    frac_terms = [name for name in support if name.startswith("D_x^")]
    return (
        len(support) == 2
        and "Hx" in support
        and len(frac_terms) == 1
        and 1.0 < float(beta) < 2.0
        and coef_dbeta is not None
    )


def equation_from_record(alpha: float, beta: float, coef_hx: float | None, coef_dbeta: float | None, support: tuple[str, ...]) -> str:
    pieces: list[str] = []
    if coef_hx is not None and "Hx" in support:
        pieces.append(f"{coef_hx:.4g}*Hx")
    frac_name = next((name for name in support if name.startswith("D_x^")), None)
    if coef_dbeta is not None and frac_name is not None:
        pieces.append(f"{coef_dbeta:.4g} D_x^{beta:.6g} H")
    for name in support:
        if name == "Hx" or name.startswith("D_x^"):
            continue
        pieces.append(name)
    rhs = " + ".join(pieces) if pieces else "0"
    return f"D_t^{alpha:.8g} H = {rhs}"


def relative_error(value: float | None, truth: float) -> float | None:
    if value is None:
        return None
    return abs(float(value) - truth) / max(abs(truth), np.finfo(float).eps)


def run_case(noise_level: float, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    noise_label, paper_example = DEFAULT_CASES[noise_level]
    config = config_for_paper_example(paper_example)
    if args.stridge_mode == "same_stridge_core" and args.core_lamb_override is not None:
        config = replace(config, sparsity_lamb=float(args.core_lamb_override))
    if args.stridge_mode == "same_stridge_core" and args.core_d_tol_override is not None:
        config = replace(config, d_tol=float(args.core_d_tol_override))
    discoverer = GJHybridDiscoverer(config)
    torch, net, metadata = discoverer._load_network()
    operator_field = discoverer._build_base_field(torch, net, metadata)
    fit_field = discoverer._fit_window_field(operator_field)

    alpha_cache: dict[float, np.ndarray] = {}
    beta_cache: dict[float, np.ndarray] = {}
    trace: list[dict[str, Any]] = []
    eval_counter = 0
    if args.stridge_mode == "legacy":
        effective_lamb = float(args.lamb)
        effective_ridge_lam = float(args.ridge_lam)
        effective_d_tol = float(args.d_tol)
        effective_maxit = int(args.maxit)
        effective_str_iters = int(args.str_iters)
        effective_normalize = int(args.normalize)
        effective_split = float(args.split)
    else:
        effective_lamb = float(config.sparsity_lamb)
        effective_ridge_lam = float(config.ridge_lambda)
        effective_d_tol = float(config.d_tol)
        effective_maxit = int(config.maxit)
        effective_str_iters = int(config.str_iters)
        effective_normalize = int(config.normalize)
        effective_split = float(config.split)

    def evaluate(alpha: float, beta: float) -> LegacyFitResult:
        matrix, target, term_names = fit_values_for_orders(
            discoverer,
            torch,
            net,
            operator_field,
            fit_field,
            alpha_cache,
            beta_cache,
            alpha,
            beta,
        )
        if args.stridge_mode == "same_stridge_core":
            return mainline_stridge_fit(discoverer, matrix, target, term_names)
        return legacy_fstridge_fit(
            matrix,
            target,
            term_names,
            seed=int(args.seed),
            ridge_lam=effective_ridge_lam,
            d_tol=effective_d_tol,
            maxit=effective_maxit,
            str_iters=effective_str_iters,
            normalize=effective_normalize,
            split=effective_split,
            lamb=effective_lamb,
        )

    def objective(params: np.ndarray) -> float:
        nonlocal eval_counter
        alpha = float(params[0])
        beta = float(params[1])
        started = time.perf_counter()
        try:
            fit = evaluate(alpha, beta)
            status = "ok"
            message = ""
            loss = fit.objective
        except Exception as exc:  # pragma: no cover - recorded for long batch robustness
            fit = None
            status = "failed"
            message = str(exc)
            loss = 1.0e30
        elapsed = time.perf_counter() - started
        eval_counter += 1
        row: dict[str, Any] = {
            "eval": eval_counter,
            "alpha": alpha,
            "beta": beta,
            "loss": float(loss),
            "status": status,
            "message": message,
            "seconds": elapsed,
        }
        if fit is not None:
            row.update(
                {
                    "support": list(fit.support),
                    "coef_hx": fit.coef_hx,
                    "coef_dbeta": fit.coef_dbeta,
                    "tolerance": fit.tolerance,
                    "condition": fit.condition,
                    "mse": fit.mse,
                    "validation_residual_norm": fit.validation_residual_norm,
                    "coefficients": fit.coefficients.tolist(),
                    "stridge_mode": args.stridge_mode,
                    "lamb": effective_lamb,
                    "d_tol": effective_d_tol,
                }
            )
        trace.append(row)
        if not args.quiet:
            print(
                f"{noise_label} eval={eval_counter} loss={loss:.6g} alpha={alpha:.5f} beta={beta:.5f} "
                f"support={row.get('support')}",
                flush=True,
            )
        return float(loss)

    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=[(0.01, 0.99999), (1.01, 1.99999)],
        maxiter=int(args.maxiter),
        tol=float(args.tol),
        popsize=int(args.popsize),
        polish=False,
        seed=int(args.seed),
        updating="immediate",
        workers=1,
        disp=not args.quiet,
    )
    runtime = time.perf_counter() - started
    alpha_opt = float(result.x[0])
    beta_opt = float(result.x[1])
    final_fit = evaluate(alpha_opt, beta_opt)
    support_correct = support_is_correct(final_fit.support, beta_opt, final_fit.coef_dbeta)
    equation = equation_from_record(alpha_opt, beta_opt, final_fit.coef_hx, final_fit.coef_dbeta, final_fit.support)
    record = {
        "noise_level": noise_level,
        "noise_label": noise_label,
        "method": f"legacy_de_normalized_tanh_adapter_{args.stridge_mode}",
        "stridge_mode": args.stridge_mode,
        "paper_example": paper_example,
        "checkpoint": str(config.checkpoint_path),
        "fit_x_min": config.fit_x_min,
        "fit_x_max": config.fit_x_max,
        "fit_t_min": config.fit_t_min,
        "fit_t_max": config.fit_t_max,
        "operator_x_min": config.x_min,
        "operator_x_max": config.x_max,
        "operator_t_min": config.t_min,
        "operator_t_max": config.t_max,
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "tol": float(args.tol),
        "de_success": bool(result.success),
        "de_message": str(result.message),
        "de_fun": float(result.fun),
        "de_nit": int(result.nit),
        "de_nfev": int(result.nfev),
        "alpha": alpha_opt,
        "beta": beta_opt,
        "coef_hx": final_fit.coef_hx,
        "coef_dbeta": final_fit.coef_dbeta,
        "support": list(final_fit.support),
        "support_correct": bool(support_correct),
        "equation": equation,
        "runtime_seconds": runtime,
        "alpha_error": relative_error(alpha_opt, TRUE_ALPHA),
        "beta_error": relative_error(beta_opt, TRUE_BETA) if final_fit.coef_dbeta is not None else None,
        "coef_hx_error": relative_error(final_fit.coef_hx, TRUE_HX),
        "coef_dbeta_error": relative_error(final_fit.coef_dbeta, TRUE_DBETA),
        "legacy_lamb": effective_lamb,
        "legacy_ridge_lam": effective_ridge_lam,
        "legacy_d_tol": effective_d_tol,
        "legacy_maxit": effective_maxit,
        "legacy_str_iters": effective_str_iters,
        "legacy_split": effective_split,
        "legacy_normalize": effective_normalize,
        "lamb": effective_lamb,
        "d_tol": effective_d_tol,
        "ridge_lambda": effective_ridge_lam,
        "split": effective_split,
        "objective": final_fit.objective,
        "mse": final_fit.mse,
        "validation_residual_norm": final_fit.validation_residual_norm,
        "condition": final_fit.condition,
        "tolerance": final_fit.tolerance,
    }
    return record, trace


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_outputs(records: list[dict[str, Any]], traces: dict[str, list[dict[str, Any]]], out_dir: Path, write_trace_csv: bool) -> None:
    candidates = [
        out_dir,
        Path(tempfile.gettempdir()) / "linear_fractional_results" / out_dir.name,
    ]
    last_error: Exception | None = None
    for candidate_dir in candidates:
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            probe = candidate_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            out_dir = candidate_dir
            break
        except OSError as exc:
            last_error = exc
    else:
        raise PermissionError(f"No writable legacy output directory found: {last_error}") from last_error

    mode = str(records[0].get("stridge_mode", "legacy")) if records else "legacy"
    csv_path = out_dir / f"legacy_de_normalized_tsfade_alpha078_beta183_{mode}.csv"
    json_path = out_dir / f"legacy_de_normalized_tsfade_alpha078_beta183_{mode}.json"
    try:
        with csv_path.open("w", newline="", encoding="utf-8"):
            pass
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "linear_fractional_results" / out_dir.name
        if out_dir.resolve() != fallback.resolve():
            return write_outputs(records, traces, fallback, write_trace_csv)
        raise
    fieldnames = [
        "noise_level",
        "noise_label",
        "method",
        "stridge_mode",
        "paper_example",
        "checkpoint",
        "fit_x_min",
        "fit_x_max",
        "fit_t_min",
        "fit_t_max",
        "maxiter",
        "popsize",
        "de_success",
        "de_message",
        "de_fun",
        "de_nit",
        "de_nfev",
        "alpha",
        "beta",
        "coef_hx",
        "coef_dbeta",
        "support",
        "support_correct",
        "equation",
        "runtime_seconds",
        "alpha_error",
        "beta_error",
        "coef_hx_error",
        "coef_dbeta_error",
        "legacy_lamb",
        "legacy_d_tol",
        "lamb",
        "d_tol",
        "ridge_lambda",
        "split",
        "objective",
        "mse",
        "condition",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            serial = dict(row)
            serial["support"] = ";".join(str(item) for item in row.get("support", []))
            writer.writerow({field: serial.get(field) for field in fieldnames})
    json_path.write_text(json.dumps({"records": records, "traces": traces}, indent=2, default=json_default), encoding="utf-8")

    for noise_label, trace in traces.items():
        safe_label = noise_label.replace("%", "pct").replace(" ", "_")
        trace_json = out_dir / f"legacy_de_normalized_tsfade_alpha078_beta183_{mode}_trace_{safe_label}.json"
        trace_json.write_text(json.dumps(trace, indent=2, default=json_default), encoding="utf-8")
        if write_trace_csv and trace:
            trace_csv = out_dir / f"legacy_de_normalized_tsfade_alpha078_beta183_{mode}_trace_{safe_label}.csv"
            keys = sorted({key for row in trace for key in row})
            with trace_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in trace:
                    serial = dict(row)
                    if isinstance(serial.get("support"), list):
                        serial["support"] = ";".join(str(item) for item in serial["support"])
                    if isinstance(serial.get("coefficients"), list):
                        serial["coefficients"] = ";".join(f"{float(item):.12g}" for item in serial["coefficients"])
                    writer.writerow(serial)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def main() -> int:
    args = parse_args()
    noise_levels = parse_noise_levels(args.noise_levels)
    records: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, Any]]] = {}
    for noise_level in noise_levels:
        noise_label, _ = DEFAULT_CASES[noise_level]
        print(f"running fair normalized DE legacy benchmark: noise={noise_label}", flush=True)
        record, trace = run_case(noise_level, args)
        records.append(record)
        traces[noise_label] = trace
        print(f"  result: {record['equation']} support_correct={record['support_correct']}", flush=True)
    write_outputs(records, traces, args.out_dir, write_trace_csv=not args.no_trace_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
