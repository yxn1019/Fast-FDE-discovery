"""Polish MADE2 FFT model parameters under profile-normalized objectives.

The input is the already discovered no-refit FFT prediction bundle.  This
diagnostic optimizes either only the first-order reaction coefficient or all
parameters of the fixed discovered support.  The objective uses the same
profile-wise normalization as the MADE paper figures: each observed and
predicted profile is divided by its own spatial mass before residuals are
computed.  It intentionally does not modify the paper figures; use the plotting
script with the generated NPZ to preview the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.plot_hydrology_fft_discovery_heatmaps import _fft_prediction  # noqa: E402


DEFAULT_INPUT = ROOT / "figures" / "hydrology_fft_discovery" / "made2_row252" / "made2_row252_prediction.npz"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "hydrology_fft_discovery" / "made2_row252_reaction_refit"


def _safe_profile(field: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(field, dtype=float), 0.0, None)


def _profile_area(x: np.ndarray, profile: np.ndarray) -> float:
    return float(np.trapezoid(_safe_profile(profile), x))


def _own_mass_normalize(x: np.ndarray, field: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(field, dtype=float):
        area = _profile_area(x, row)
        clean = _safe_profile(row)
        if not np.isfinite(area) or area <= np.finfo(float).eps:
            rows.append(np.zeros_like(clean))
        else:
            rows.append(clean / area)
    return np.vstack(rows)


def _relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(finite):
        return float("nan")
    denom = max(float(np.linalg.norm(reference[finite])), np.finfo(float).eps)
    return float(np.linalg.norm(prediction[finite] - reference[finite]) / denom)


def _equation_from_npz(data: Any, reaction_coefficient: float | None = None) -> dict[str, Any]:
    coef_h = float(np.asarray(data["coef_h"]).reshape(())) if reaction_coefficient is None else float(reaction_coefficient)
    beta = float(np.asarray(data["beta"]).reshape(()))
    return {
        "alpha": float(np.asarray(data["alpha"]).reshape(())),
        "beta": beta,
        "constant": float(np.asarray(data["constant"]).reshape(())),
        "h": coef_h,
        "hx": float(np.asarray(data["coef_hx"]).reshape(())),
        "fractional": float(np.asarray(data["coef_fractional"]).reshape(())),
    }


def _equation_from_vector(values: np.ndarray) -> dict[str, Any]:
    alpha, beta, constant, h, hx, fractional = [float(v) for v in values]
    return {
        "alpha": alpha,
        "beta": beta,
        "constant": constant,
        "h": h,
        "hx": hx,
        "fractional": fractional,
    }


def _vector_from_equation(equation: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(equation["alpha"]),
            float(equation["beta"]),
            float(equation["constant"]),
            float(equation["h"]),
            float(equation["hx"]),
            float(equation["fractional"]),
        ],
        dtype=float,
    )


def _format_equation(equation: dict[str, Any]) -> str:
    return (
        f"D_t^{equation['alpha']:.3f} H = "
        f"{equation['constant']:.5g} - {abs(equation['h']):.5g} H "
        f"{equation['hx']:+.5g} Hx {equation['fractional']:+.5g} D_x^{equation['beta']:.3f} H"
    ).replace("+ -", "- ")


def _training_indices(t: np.ndarray, split_time: float) -> np.ndarray:
    mask = np.asarray(t, dtype=float) <= float(split_time) + 1.0e-9
    indices = np.flatnonzero(mask)
    # The initial row is used as the FFT initial condition and is insensitive to
    # the reaction coefficient.  Excluding it keeps the scalar fit focused on
    # propagated profiles.
    return indices[indices != 0]


def _objective(
    coefficient: float,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    surrogate: np.ndarray,
    base_equation: dict[str, Any],
    train_indices: np.ndarray,
) -> float:
    equation = dict(base_equation)
    equation["h"] = float(coefficient)
    prediction = _safe_profile(_fft_prediction(x=x, t=t, surrogate=surrogate, equation=equation))
    # Match the paper-profile diagnostic: each observed and predicted profile is
    # divided by its own spatial mass before the objective is evaluated.
    reference = _own_mass_normalize(x, raw)
    prediction_norm = _own_mass_normalize(x, prediction)
    residual = prediction_norm[train_indices] - reference[train_indices]
    denom = max(float(np.linalg.norm(reference[train_indices])), np.finfo(float).eps)
    return float(np.linalg.norm(residual) / denom)


def _residual_vector(
    vector: np.ndarray,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    surrogate: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    equation = _equation_from_vector(np.asarray(vector, dtype=float))
    prediction = _safe_profile(_fft_prediction(x=x, t=t, surrogate=surrogate, equation=equation))
    reference = _own_mass_normalize(x, raw)
    prediction_norm = _own_mass_normalize(x, prediction)
    residual = prediction_norm[train_indices] - reference[train_indices]
    denom = max(float(np.linalg.norm(reference[train_indices])), np.finfo(float).eps)
    return (residual / denom).reshape(-1)


def _minimize_reaction(
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    surrogate: np.ndarray,
    base_equation: dict[str, Any],
    train_indices: np.ndarray,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda value: _objective(
                value,
                x=x,
                t=t,
                raw=raw,
                surrogate=surrogate,
                base_equation=base_equation,
                train_indices=train_indices,
            ),
            bounds=bounds,
            method="bounded",
            options={"xatol": 1.0e-7},
        )
        return {
            "coef_h": float(result.x),
            "objective": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "method": "scipy_minimize_scalar_bounded",
        }
    except Exception as exc:
        grid = np.linspace(float(bounds[0]), float(bounds[1]), 121)
        values = [
            _objective(
                value,
                x=x,
                t=t,
                raw=raw,
                surrogate=surrogate,
                base_equation=base_equation,
                train_indices=train_indices,
            )
            for value in grid
        ]
        index = int(np.argmin(values))
        return {
            "coef_h": float(grid[index]),
            "objective": float(values[index]),
            "success": False,
            "message": f"scipy unavailable or failed; used grid fallback: {exc}",
            "nfev": int(len(grid)),
            "method": "grid_fallback",
        }


def _minimize_all_parameters(
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    surrogate: np.ndarray,
    base_equation: dict[str, Any],
    train_indices: np.ndarray,
    max_nfev: int,
    bounds_mode: str,
) -> dict[str, Any]:
    initial = _vector_from_equation(base_equation)
    if bounds_mode == "local":
        lower = np.asarray([0.90, 1.70, 0.00, -0.025, -0.045, 0.005], dtype=float)
        upper = np.asarray([0.999, 2.00, 0.08, 0.005, -0.005, 0.055], dtype=float)
    else:
        lower = np.asarray([0.60, 1.05, -0.20, -0.08, -0.12, -0.08], dtype=float)
        upper = np.asarray([0.999, 2.00, 0.20, 0.04, 0.08, 0.12], dtype=float)
    initial = np.minimum(np.maximum(initial, lower + 1.0e-10), upper - 1.0e-10)
    try:
        from scipy.optimize import least_squares

        result = least_squares(
            lambda vector: _residual_vector(
                vector,
                x=x,
                t=t,
                raw=raw,
                surrogate=surrogate,
                train_indices=train_indices,
            ),
            x0=initial,
            bounds=(lower, upper),
            method="trf",
            loss="linear",
            x_scale="jac",
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
            max_nfev=int(max_nfev),
        )
        residual = _residual_vector(
            result.x,
            x=x,
            t=t,
            raw=raw,
            surrogate=surrogate,
            train_indices=train_indices,
        )
        return {
            "parameters": _equation_from_vector(result.x),
            "objective": float(np.linalg.norm(residual)),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "active_mask": [int(v) for v in result.active_mask],
            "method": "scipy_least_squares_profile_normalized_full_support",
            "bounds_mode": str(bounds_mode),
            "bounds": {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            },
        }
    except Exception as exc:
        residual = _residual_vector(
            initial,
            x=x,
            t=t,
            raw=raw,
            surrogate=surrogate,
            train_indices=train_indices,
        )
        return {
            "parameters": _equation_from_vector(initial),
            "objective": float(np.linalg.norm(residual)),
            "success": False,
            "message": f"least_squares failed; returned initial parameters: {exc}",
            "nfev": 0,
            "cost": float(0.5 * np.dot(residual, residual)),
            "optimality": float("nan"),
            "active_mask": [],
            "method": "initial_fallback",
            "bounds_mode": str(bounds_mode),
            "bounds": {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            },
        }
def _slice_metrics(x: np.ndarray, t: np.ndarray, raw: np.ndarray, prediction: np.ndarray, split_time: float) -> list[dict[str, Any]]:
    raw_clip = _safe_profile(raw)
    pred_clip = _safe_profile(prediction)
    raw_norm = _own_mass_normalize(x, raw_clip)
    pred_norm = _own_mass_normalize(x, pred_clip)
    rows = []
    for index, time_value in enumerate(np.asarray(t, dtype=float)):
        rows.append(
            {
                "time": float(time_value),
                "split": "train" if float(time_value) <= float(split_time) + 1.0e-9 else "holdout",
                "relative_l2_raw": _relative_l2(raw_clip[index], pred_clip[index]),
                "relative_l2_own_mass_normalized": _relative_l2(raw_norm[index], pred_norm[index]),
                "raw_mass": _profile_area(x, raw_clip[index]),
                "prediction_mass": _profile_area(x, pred_clip[index]),
                "mass_ratio_prediction_to_raw": _profile_area(x, pred_clip[index])
                / max(_profile_area(x, raw_clip[index]), np.finfo(float).eps),
                "peak_location_error": float(abs(x[int(np.argmax(pred_clip[index]))] - x[int(np.argmax(raw_clip[index]))])),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split and row["time"] > 9.0]
    if not subset:
        return {}
    return {
        "mean_relative_l2_raw": float(np.mean([row["relative_l2_raw"] for row in subset])),
        "mean_relative_l2_own_mass_normalized": float(
            np.mean([row["relative_l2_own_mass_normalized"] for row in subset])
        ),
        "mean_abs_mass_ratio_error": float(
            np.mean([abs(row["mass_ratio_prediction_to_raw"] - 1.0) for row in subset])
        ),
        "mean_peak_location_error": float(np.mean([row["peak_location_error"] for row in subset])),
    }


def _time_index(t: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t, dtype=float) - float(value))))


def _plot_comparison_profiles(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    base_prediction: np.ndarray,
    polished_prediction: np.ndarray,
    times: list[float],
    ncols: int,
    y_upper: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_norm = _own_mass_normalize(x, raw)
    base_norm = _own_mass_normalize(x, base_prediction)
    polished_norm = _own_mass_normalize(x, polished_prediction)
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for panel_index, (ax, requested_time) in enumerate(zip(axes.reshape(-1), times)):
        idx = _time_index(t, requested_time)
        ax.plot(
            x,
            raw_norm[idx],
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            markeredgewidth=0.85,
            color="#1f77b4",
            label="observed",
        )
        ax.plot(x, base_norm[idx], color="#d95f02", lw=1.8, label="direct discovery")
        ax.plot(x, polished_norm[idx], color="#2ca02c", lw=1.8, label="parameter polish")
        ax.set_title(f"t={float(t[idx]):g} d", fontsize=9.5, pad=4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Normalized concentration")
        ax.set_ylim(bottom=0.0, top=y_upper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        if panel_index == 0:
            ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    for ax in axes.reshape(-1)[len(times) :]:
        ax.axis("off")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-time", type=float, default=None)
    parser.add_argument("--lower", type=float, default=-0.05, help="lower bound for the H reaction coefficient")
    parser.add_argument("--upper", type=float, default=0.02, help="upper bound for the H reaction coefficient")
    parser.add_argument(
        "--mode",
        choices=["reaction", "full"],
        default="reaction",
        help="polish only the H reaction coefficient or all fixed-support parameters",
    )
    parser.add_argument("--max-nfev", type=int, default=160, help="maximum function evaluations for full polish")
    parser.add_argument(
        "--full-bounds",
        choices=["local", "broad"],
        default="local",
        help="parameter bounds for full polish; local keeps the solution near the discovered physics",
    )
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    raw = np.asarray(data["raw"], dtype=float)
    surrogate = np.asarray(data["surrogate"], dtype=float)
    split_time = (
        float(args.split_time)
        if args.split_time is not None
        else float(np.asarray(data["split_time"]).reshape(()))
        if "split_time" in data
        else float(t[-2])
    )

    base_equation = _equation_from_npz(data)
    train_indices = _training_indices(t, split_time)
    if args.mode == "full":
        fit = _minimize_all_parameters(
            x=x,
            t=t,
            raw=raw,
            surrogate=surrogate,
            base_equation=base_equation,
            train_indices=train_indices,
            max_nfev=int(args.max_nfev),
            bounds_mode=str(args.full_bounds),
        )
        refit_equation = dict(fit["parameters"])
    else:
        fit = _minimize_reaction(
            x=x,
            t=t,
            raw=raw,
            surrogate=surrogate,
            base_equation=base_equation,
            train_indices=train_indices,
            bounds=(float(args.lower), float(args.upper)),
        )
        refit_equation = _equation_from_npz(data, reaction_coefficient=float(fit["coef_h"]))
    base_prediction = _fft_prediction(x=x, t=t, surrogate=surrogate, equation=base_equation)
    refit_prediction = _fft_prediction(x=x, t=t, surrogate=surrogate, equation=refit_equation)

    base_rows = _slice_metrics(x, t, raw, base_prediction, split_time)
    refit_rows = _slice_metrics(x, t, raw, refit_prediction, split_time)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "made2_row252_parameter_polish_full" if args.mode == "full" else "made2_row252_reaction_refit"
    npz_path = output_dir / f"{prefix}_prediction.npz"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    comparison_train_path = output_dir / f"{prefix}_training_comparison.png"
    comparison_holdout_path = output_dir / f"{prefix}_holdout_comparison.png"

    np.savez_compressed(
        npz_path,
        x=x,
        t=t,
        raw=raw,
        surrogate=surrogate,
        fft_prediction=refit_prediction,
        fft_prediction_display=_safe_profile(refit_prediction),
        alpha=np.array(float(refit_equation["alpha"])),
        beta=np.array(float(refit_equation["beta"])),
        constant=np.array(float(refit_equation["constant"])),
        coef_h=np.array(float(refit_equation["h"])),
        coef_hx=np.array(float(refit_equation["hx"])),
        coef_fractional=np.array(float(refit_equation["fractional"])),
        split_time=np.array(split_time),
        summary_row_line=np.asarray(data["summary_row_line"]) if "summary_row_line" in data else np.array(252),
        run_id=np.array(f"{prefix}_own_mass_normalized"),
    )

    ref_idx = _time_index(t, 126.0)
    raw_norm = _own_mass_normalize(x, raw)
    base_norm = _own_mass_normalize(x, base_prediction)
    refit_norm = _own_mass_normalize(x, refit_prediction)
    y_upper = 1.08 * float(
        np.nanmax([np.nanmax(raw_norm[ref_idx]), np.nanmax(base_norm[ref_idx]), np.nanmax(refit_norm[ref_idx])])
    )
    if not np.isfinite(y_upper) or y_upper <= 0.0:
        y_upper = 1.0
    _plot_comparison_profiles(
        comparison_train_path,
        x=x,
        t=t,
        raw=raw,
        base_prediction=base_prediction,
        polished_prediction=refit_prediction,
        times=[126.0, 202.0, 279.0, 370.0],
        ncols=2,
        y_upper=y_upper,
    )
    _plot_comparison_profiles(
        comparison_holdout_path,
        x=x,
        t=t,
        raw=raw,
        base_prediction=base_prediction,
        polished_prediction=refit_prediction,
        times=[503.0],
        ncols=1,
        y_upper=y_upper,
    )

    metrics = {
        "input_npz": str(args.input_npz),
        "output_npz": str(npz_path),
        "split_time": split_time,
        "train_times_used": [float(t[index]) for index in train_indices],
        "visible_name": "parameter polish",
        "polish_mode": "full_fixed_support" if args.mode == "full" else "reaction_coefficient_only",
        "objective": "own_mass_normalized_profile_l2_train",
        "bounds": [float(args.lower), float(args.upper)] if args.mode == "reaction" else fit.get("bounds"),
        "fit": fit,
        "base_equation": _format_equation(base_equation),
        "refit_equation": _format_equation(refit_equation),
        "base_coefficients": base_equation,
        "refit_coefficients": refit_equation,
        "base_slice_metrics": base_rows,
        "refit_slice_metrics": refit_rows,
        "base_train_summary": _aggregate(base_rows, "train"),
        "refit_train_summary": _aggregate(refit_rows, "train"),
        "base_holdout_summary": _aggregate(base_rows, "holdout"),
        "refit_holdout_summary": _aggregate(refit_rows, "holdout"),
        "comparison_figures": {
            "training": str(comparison_train_path),
            "holdout": str(comparison_holdout_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"base:  {metrics['base_equation']}")
    print(f"polished: {metrics['refit_equation']}")
    print("parameter changes:")
    for key in ["alpha", "beta", "constant", "h", "hx", "fractional"]:
        print(f"  {key}: {base_equation[key]:.10g} -> {refit_equation[key]:.10g}")
    print(f"fit objective: {fit['objective']:.6g}; success={fit['success']}; nfev={fit['nfev']}")
    print(f"output_npz: {npz_path}")
    print(f"metrics: {metrics_path}")
    print(f"comparison_training: {comparison_train_path}")
    print(f"comparison_holdout: {comparison_holdout_path}")


if __name__ == "__main__":
    main()
