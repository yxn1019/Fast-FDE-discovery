"""Plot hydrology raw/NN/FFT prediction heatmaps for a discovered equation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plot_raw_vs_nn_surrogate import reconstruct_nn_on_raw_grid  # noqa: E402
from transporteq_discovery.fractional_discoverer import FractionalDiscoveryConfig  # noqa: E402
from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer  # noqa: E402
from transporteq_discovery.mittag_leffler import mittag_leffler  # noqa: E402


DEFAULT_SUMMARY = (
    ROOT
    / "results"
    / "hydrology_discovery_scan"
    / "coarse_hybrid_log_mse_20260516_160112"
    / "summary.csv"
)
DEFAULT_HYDROLOGY_SUMMARY = ROOT / "data" / "hydrology_experiments" / "hydrology_preparation_summary.json"
DEFAULT_MODEL_ROOT = ROOT / "data" / "models" / "hydrology_experiments"
DEFAULT_OUTPUT_ROOT = ROOT / "figures" / "hydrology_fft_discovery"


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary_row(path: Path, physical_line: int) -> dict[str, str]:
    if physical_line < 2:
        raise ValueError("--summary-row-line must include the CSV header and therefore be >= 2")
    rows = _load_summary_rows(path)
    index = int(physical_line) - 2
    if index < 0 or index >= len(rows):
        raise IndexError(f"CSV physical line {physical_line} is outside {path} with {len(rows) + 1} lines")
    return rows[index]


def _load_summary_row_by_run_id(path: Path, run_id: str) -> dict[str, str]:
    rows = _load_summary_rows(path)
    matches = [row for row in rows if row.get("run_id") == run_id]
    if not matches:
        raise KeyError(f"run_id {run_id!r} not found in {path}")
    if len(matches) > 1:
        raise ValueError(f"run_id {run_id!r} appears {len(matches)} times in {path}")
    return matches[0]


def _loads_json_field(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return default


def _float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} as float: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _fractional_key(active_terms: dict[str, float]) -> str | None:
    for key in active_terms:
        if str(key).startswith("D_x^"):
            return str(key)
    return None


def _parse_equation(row: dict[str, str]) -> dict[str, Any]:
    active_raw = _loads_json_field(row.get("active_terms", "{}"), {})
    if not isinstance(active_raw, dict):
        raise ValueError(f"active_terms is not a JSON object: {active_raw!r}")
    active_terms = {str(key): _float(value, f"active_terms[{key}]") for key, value in active_raw.items()}
    frac_key = _fractional_key(active_terms)
    beta = _float(row.get("beta"), "beta")
    if frac_key is not None:
        match = re.search(r"D_x\^([0-9eE+\-.]+)", frac_key)
        if match:
            beta = float(match.group(1))
    return {
        "run_id": row.get("run_id"),
        "experiment": row.get("experiment"),
        "variant": row.get("variant"),
        "candidate_equation": row.get("candidate_equation"),
        "active_terms": active_terms,
        "support": _loads_json_field(row.get("support", "[]"), []),
        "alpha": _float(row.get("alpha"), "alpha"),
        "beta": beta,
        "constant": float(active_terms.get("1", 0.0)),
        "h": float(active_terms.get("H", 0.0)),
        "hx": float(active_terms.get("Hx", 0.0)),
        "fractional_key": frac_key,
        "fractional": float(active_terms.get(frac_key, 0.0)) if frac_key is not None else 0.0,
        "x_max": _float(row.get("x_max"), "x_max"),
        "fit_t_min": _float(row.get("fit_t_min"), "fit_t_min"),
        "fit_t_max": _float(row.get("fit_t_max"), "fit_t_max"),
        "objective": _float(row.get("objective"), "objective"),
        "validation_residual_norm": _float(row.get("validation_residual_norm"), "validation_residual_norm"),
        "command": row.get("command", ""),
    }


def _command_float(command: str, option: str, fallback: float) -> float:
    match = re.search(rf"{re.escape(option)}\s+([^\s\"]+|\"[^\"]+\")", command)
    if not match:
        return float(fallback)
    value = match.group(1).strip().strip('"')
    try:
        return float(value)
    except ValueError:
        return float(fallback)


def _load_metadata(value: Any) -> dict[str, Any]:
    try:
        return json.loads(str(np.asarray(value).reshape(())))
    except Exception:
        return {}


def _load_hydrology_items(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, dict):
            items[(str(item.get("experiment")), str(item.get("variant")))] = item
    return items


def _model_checkpoint(*, experiment: str, variant: str, hidden_layers: int, neurons: int, loss: str) -> Path:
    label = f"{experiment}_{variant}_tanh_{hidden_layers}x{neurons}_{loss}"
    return DEFAULT_MODEL_ROOT / label / "best.pkl"


def _sanitize_path_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def _resolve_paths(
    *,
    equation: dict[str, Any],
    hydrology_summary: Path,
    full_npz: Path | None,
    checkpoint: Path | None,
    output_dir: Path | None,
    hidden_layers: int,
    neurons: int,
    loss: str,
) -> tuple[Path, Path, Path]:
    experiment = str(equation.get("experiment") or "")
    variant = str(equation.get("variant") or "")
    items = _load_hydrology_items(hydrology_summary)
    item = items.get((experiment, variant))
    if item is None:
        raise KeyError(f"No hydrology prep item for {experiment}:{variant} in {hydrology_summary}")
    resolved_npz = Path(full_npz) if full_npz is not None else Path(str(item["full_npz"]))
    resolved_checkpoint = (
        Path(checkpoint)
        if checkpoint is not None
        else _model_checkpoint(
            experiment=experiment,
            variant=variant,
            hidden_layers=hidden_layers,
            neurons=neurons,
            loss=loss,
        )
    )
    if output_dir is not None:
        resolved_output = Path(output_dir)
    else:
        token = _sanitize_path_token(f"{experiment}_{variant}_{equation.get('run_id') or 'summary_row'}")
        resolved_output = DEFAULT_OUTPUT_ROOT / token
    return resolved_npz, resolved_checkpoint, resolved_output


def _fourier_wavenumbers(x: np.ndarray) -> np.ndarray:
    dx = float(np.median(np.diff(np.asarray(x, dtype=float))))
    return 2.0 * np.pi * np.fft.fftfreq(int(x.size), d=dx)


def _ik_power(k: np.ndarray, order: float) -> np.ndarray:
    ik = 1j * np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k != 0.0, ik ** float(order), 0.0 + 0.0j)


def _fft_prediction(
    *,
    x: np.ndarray,
    t: np.ndarray,
    surrogate: np.ndarray,
    equation: dict[str, Any],
) -> np.ndarray:
    alpha = float(equation["alpha"])
    beta = float(equation["beta"])
    c0 = float(equation["constant"])
    c_h = float(equation["h"])
    c_hx = float(equation["hx"])
    c_beta = float(equation["fractional"])

    k = _fourier_wavenumbers(x)
    lambda_k = c_h + c_hx * (1j * k) + c_beta * _ik_power(k, beta)
    u0 = np.asarray(surrogate[0, :], dtype=float)
    u0_hat = np.fft.fft(u0)
    source_hat = np.zeros_like(u0_hat, dtype=np.complex128)
    source_hat[0] = int(x.size) * c0

    t0 = float(t[0])
    prediction = np.zeros((int(t.size), int(x.size)), dtype=float)
    for index, time_value in enumerate(np.asarray(t, dtype=float)):
        tau = max(float(time_value) - t0, 0.0)
        if tau == 0.0:
            u_hat = u0_hat
        else:
            z = lambda_k * (tau**alpha)
            homogeneous = u0_hat * mittag_leffler(z, alpha, 1.0)
            forced = source_hat * (tau**alpha) * mittag_leffler(z, alpha, alpha + 1.0)
            u_hat = homogeneous + forced
        prediction[index, :] = np.fft.ifft(u_hat).real
    return prediction


def _refit_coefficients(
    *,
    checkpoint: Path,
    x: np.ndarray,
    t: np.ndarray,
    equation: dict[str, Any],
) -> dict[str, Any]:
    """Refit the selected physical support on the dense G-J operator grid."""

    command = str(equation.get("command") or "")
    dx = float(np.median(np.diff(x)))
    t_step_fallback = float(np.median(np.diff(t)))
    config = FractionalDiscoveryConfig(
        checkpoint_file=checkpoint,
        x_min=_command_float(command, "--x-min", float(x[0])),
        x_max=float(equation["x_max"]),
        x_step=_command_float(command, "--x-step", dx),
        t_min=_command_float(command, "--t-min", float(t[0])),
        t_max=_command_float(command, "--t-max", float(t[-2] + t_step_fallback) if t.size > 1 else float(t[-1])),
        t_step=_command_float(command, "--t-step", t_step_fallback),
        fit_x_min=_command_float(command, "--fit-x-min", float(x[0])),
        fit_x_max=float(equation["x_max"]),
        fit_t_min=float(equation["fit_t_min"]),
        fit_t_max=float(equation["fit_t_max"]),
        laguerre_nodes=int(round(_command_float(command, "--laguerre-nodes", 5.0))),
        enable_spatial_fractional=True,
        spatial_fractional_mode="gj_richardson",
        space_derivative_mode="autodiff",
        time_operator_mode="laplace_taylor",
    )
    discoverer = GJHybridDiscoverer(config)
    torch, net, metadata = discoverer._load_network()
    operator_field = discoverer._build_base_field(torch, net, metadata)
    fit_field = discoverer._fit_window_field(operator_field)
    halpha_full = discoverer._compute_halpha(torch, net, operator_field, float(equation["alpha"]))
    hbeta_full = discoverer._compute_hbeta(torch, net, operator_field, float(equation["beta"]))
    target = discoverer._restrict_values(halpha_full, operator_field, fit_field).reshape(-1)
    hbeta = discoverer._restrict_values(hbeta_full, operator_field, fit_field)
    terms, _order = discoverer._build_physical_terms(fit_field, hbeta, float(equation["beta"]))
    support = [str(name) for name in equation.get("support") or []]
    if not support:
        support = [str(name) for name in equation.get("active_terms", {})]
    matrix_columns: list[np.ndarray] = []
    resolved_support: list[str] = []
    for name in support:
        term_name = name
        if name.startswith("D_x^"):
            term_name = next(candidate for candidate in terms if candidate.startswith("D_x^"))
        if term_name not in terms:
            continue
        matrix_columns.append(np.asarray(terms[term_name], dtype=float).reshape(-1))
        resolved_support.append(term_name)
    if not matrix_columns:
        raise ValueError("No selected support terms were available for coefficient refit")
    theta = np.stack(matrix_columns, axis=1)
    coefficients, *_ = np.linalg.lstsq(theta, target, rcond=None)
    residual = target - theta @ coefficients
    refit_terms = dict(equation["active_terms"])
    for name, coefficient in zip(resolved_support, coefficients):
        if name.startswith("D_x^"):
            original = _fractional_key(refit_terms) or name
            refit_terms.pop(original, None)
            refit_terms[name] = float(coefficient)
        else:
            refit_terms[name] = float(coefficient)
    refit_equation = dict(equation)
    refit_equation["active_terms_original"] = dict(equation["active_terms"])
    refit_equation["active_terms"] = refit_terms
    refit_equation["constant"] = float(refit_terms.get("1", 0.0))
    refit_equation["h"] = float(refit_terms.get("H", 0.0))
    refit_equation["hx"] = float(refit_terms.get("Hx", 0.0))
    frac_key = _fractional_key(refit_terms)
    refit_equation["fractional_key"] = frac_key
    refit_equation["fractional"] = float(refit_terms.get(frac_key, 0.0)) if frac_key else 0.0
    refit_equation["refit"] = {
        "mode": "dense_gj_lstsq_fixed_support",
        "support": resolved_support,
        "sample_count": int(theta.shape[0]),
        "condition": float(np.linalg.cond(theta)),
        "residual_norm": float(np.linalg.norm(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }
    refit_equation["candidate_equation"] = _render_equation(refit_equation)
    return refit_equation


def _render_equation(equation: dict[str, Any]) -> str:
    terms = []
    for name, coefficient in equation["active_terms"].items():
        coef = float(coefficient)
        if name.startswith("D_x^"):
            terms.append(f"{coef:.5g} {name}")
        elif name == "1":
            terms.append(f"{coef:.5g}*1")
        else:
            terms.append(f"{coef:.5g}*{name}")
    rhs = " + ".join(terms).replace("+ -", "- ") if terms else "0"
    return f"D_t^{float(equation['alpha']):.7g} H = {rhs}"


def _edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.array([arr[0] - 0.5, arr[0] + 0.5], dtype=float)
    mids = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - 0.5 * (arr[1] - arr[0])
    last = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate([[first], mids, [last]])


def _shared_vmax(fields: list[np.ndarray], quantile: float) -> float:
    values = np.concatenate([np.clip(np.asarray(field).reshape(-1), 0.0, None) for field in fields])
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0
    vmax = float(np.quantile(positive, float(quantile)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(positive))
    return max(vmax, np.finfo(float).eps)


def _positive_floor(fields: list[np.ndarray]) -> float:
    values = np.concatenate([np.asarray(field).reshape(-1) for field in fields])
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0e-12
    return max(float(np.quantile(positive, 0.005)), 1.0e-12)


def _plot_heatmaps(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    surrogate: np.ndarray,
    prediction: np.ndarray,
    split_time: float,
    vmax: float,
    log_scale: bool,
    equation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment = str(equation.get("experiment") or metadata.get("experiment") or "hydrology").upper()
    variant = str(equation.get("variant") or metadata.get("variant") or "").lower()
    case_label = f"{experiment} {variant}".strip()
    time_unit = str(metadata.get("time_unit") or "")
    x_unit = str(metadata.get("x_unit") or "")
    x_label = f"x ({x_unit})" if x_unit else "x"
    t_label = f"t ({time_unit})" if time_unit else "t"
    split_label = f"train/validation split: t={float(split_time):g}"
    if time_unit:
        split_label += f" {time_unit}"
    holdout_time = metadata.get("holdout_time")
    if holdout_time is not None:
        split_label += f"\nholdout: t={float(holdout_time):g}"
        if time_unit:
            split_label += f" {time_unit}"
    panels = [
        (np.clip(raw, 0.0, None), f"{case_label} raw data"),
        (np.clip(surrogate, 0.0, None), f"{case_label} NN surrogate"),
        (np.clip(prediction, 0.0, None), f"{case_label} discovery FFT"),
    ]
    x_edges = _edges(x)
    t_edges = _edges(t)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharex=True, sharey=True, constrained_layout=True)
    image = None
    if log_scale:
        floor = _positive_floor([raw, surrogate, prediction])
        norm = mcolors.LogNorm(vmin=floor, vmax=max(vmax, floor * 10.0))
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    for ax, (field, title) in zip(axes, panels):
        display = np.clip(field, norm.vmin if log_scale else 0.0, None) if log_scale else field
        image = ax.pcolormesh(x_edges, t_edges, display, shading="auto", cmap="viridis", norm=norm)
        ax.axhline(float(split_time), color="white", lw=1.4, ls="--", alpha=0.95)
        ax.text(
            0.98,
            0.98,
            split_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.35, "pad": 3.0},
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
    axes[0].set_ylabel(t_label)
    scale = "log" if log_scale else "linear"
    fig.suptitle(
        f"{case_label} | {equation.get('row_selector', 'summary row')} | run_id={equation.get('run_id')} | {scale}\n"
        f"{equation['candidate_equation']}",
        fontsize=10.5,
    )
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.012)
    cbar.set_label("c")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _plot_validation_plume(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    prediction: np.ndarray,
    metadata: dict[str, Any],
    equation: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    holdout_time = metadata.get("holdout_time")
    if holdout_time is None:
        index = int(t.size - 1)
    else:
        index = int(np.argmin(np.abs(np.asarray(t, dtype=float) - float(holdout_time))))
    time_value = float(t[index])
    pred = np.clip(np.asarray(prediction[index, :], dtype=float), 0.0, None)
    raw_row = np.asarray(raw[index, :], dtype=float)
    raw_display = np.clip(raw_row, 0.0, None)
    experiment = str(equation.get("experiment") or metadata.get("experiment") or "hydrology").upper()
    variant = str(equation.get("variant") or metadata.get("variant") or "").lower()
    case_label = f"{experiment} {variant}".strip()
    time_unit = str(metadata.get("time_unit") or "")
    x_unit = str(metadata.get("x_unit") or "")
    x_label = f"x ({x_unit})" if x_unit else "x"
    title_time = f"t={time_value:g}" + (f" {time_unit}" if time_unit else "")

    finite = np.isfinite(raw_display) & np.isfinite(pred)
    denom = max(float(np.linalg.norm(raw_display[finite])), np.finfo(float).eps) if np.any(finite) else np.nan
    rel_l2 = float(np.linalg.norm(pred[finite] - raw_display[finite]) / denom) if np.any(finite) else np.nan
    peak_error = float(abs(x[int(np.argmax(pred))] - x[int(np.argmax(raw_display))]))

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.plot(x, raw_display, color="#111827", lw=2.0, marker="o", ms=3.0, label="raw data")
    ax.plot(x, pred, color="#2563eb", lw=2.2, label="FFT discovery")
    ax.set_xlabel(x_label)
    ax.set_ylabel("c")
    ax.set_title(f"{case_label} validation plume | {title_time}")
    ax.text(
        0.98,
        0.96,
        f"relL2={rel_l2:.3g}\npeak error={peak_error:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86, "pad": 4.0},
    )
    ax.legend(frameon=False)
    ax.grid(True, color="#e5e7eb", lw=0.8)
    ax.set_ylim(bottom=0.0)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _profile_metrics(x: np.ndarray, raw: np.ndarray, surrogate: np.ndarray, prediction: np.ndarray, t: np.ndarray, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    dx = float(np.median(np.diff(x)))
    holdout_time = metadata.get("holdout_time")
    rows: list[dict[str, Any]] = []
    for index, time_value in enumerate(t):
        raw_row = np.asarray(raw[index, :], dtype=float)
        surrogate_row = np.asarray(surrogate[index, :], dtype=float)
        pred_row = np.asarray(prediction[index, :], dtype=float)
        pred_display = np.clip(pred_row, 0.0, None)
        finite = np.isfinite(raw_row) & np.isfinite(pred_row)
        positive = finite & (raw_row > 0.0)
        denom = max(float(np.linalg.norm(raw_row[finite])), np.finfo(float).eps)
        rel_l2 = float(np.linalg.norm(pred_display[finite] - raw_row[finite]) / denom) if np.any(finite) else float("nan")
        peak = float(np.max(raw_row[positive])) if np.any(positive) else 0.0
        if np.any(positive):
            eps = 1.0e-3 * max(peak, np.finfo(float).eps)
            log_mse = float(np.mean((np.log10(pred_display[positive] + eps) - np.log10(raw_row[positive] + eps)) ** 2))
        else:
            log_mse = float("nan")
        rows.append(
            {
                "time": float(time_value),
                "split": "holdout"
                if holdout_time is not None and np.isclose(float(time_value), float(holdout_time))
                else "train",
                "relative_l2_vs_raw": rel_l2,
                "log_mse_positive_vs_raw": log_mse,
                "raw_mass": float(np.trapezoid(np.clip(raw_row, 0.0, None), x)),
                "surrogate_mass": float(np.trapezoid(np.clip(surrogate_row, 0.0, None), x)),
                "fft_mass": float(np.trapezoid(pred_display, x)),
                "fft_unclipped_mass": float(np.trapezoid(pred_row, x)),
                "mass_error_vs_raw": float(np.trapezoid(pred_display - np.clip(raw_row, 0.0, None), x)),
                "peak_location_error_vs_raw": float(abs(x[int(np.argmax(pred_display))] - x[int(np.argmax(raw_row))])),
                "negative_prediction_fraction": float(np.mean(pred_row < 0.0)),
                "negative_prediction_min": float(np.min(pred_row)),
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hydrology-summary", type=Path, default=DEFAULT_HYDROLOGY_SUMMARY)
    parser.add_argument("--summary-row-line", type=int, default=252)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--full-npz", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-time", type=float, default=None)
    parser.add_argument("--loss", default="hybrid_log_mse")
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--neurons", type=int, default=50)
    parser.add_argument("--vmax-quantile", type=float, default=0.995)
    parser.add_argument("--refit-coefficients", action="store_true")
    parser.add_argument("--linear-only", action="store_true", help="write only the linear heatmap and skip the log heatmap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row = (
        _load_summary_row_by_run_id(args.summary, str(args.run_id))
        if args.run_id
        else _load_summary_row(args.summary, args.summary_row_line)
    )
    equation = _parse_equation(row)
    equation["row_selector"] = (
        f"run_id selector"
        if args.run_id
        else f"summary physical line {int(args.summary_row_line)}"
    )
    full_npz, checkpoint, output_dir = _resolve_paths(
        equation=equation,
        hydrology_summary=args.hydrology_summary,
        full_npz=args.full_npz,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        hidden_layers=int(args.hidden_layers),
        neurons=int(args.neurons),
        loss=str(args.loss),
    )
    file_token = _sanitize_path_token(f"{equation.get('experiment')}_{equation.get('variant')}_{equation.get('run_id')}")
    linear_path = output_dir / f"{file_token}_raw_surrogate_fft_heatmaps.png"
    log_path = output_dir / f"{file_token}_raw_surrogate_fft_heatmaps_log.png"
    validation_plume_path = output_dir / f"{file_token}_validation_plume_raw_vs_fft.png"
    npz_path = output_dir / f"{file_token}_prediction.npz"
    metrics_path = output_dir / f"{file_token}_metrics.json"

    data = np.load(full_npz, allow_pickle=True)
    metadata = _load_metadata(data["metadata"]) if "metadata" in data else {}
    split_time = (
        float(args.split_time)
        if args.split_time is not None
        else float(max(metadata.get("train_times", [equation["fit_t_max"]])))
    )

    if args.dry_run:
        print(f"summary: {args.summary}")
        print(f"summary_row_line: {args.summary_row_line}")
        print(f"run_id: {equation['run_id']}")
        print(f"case: {equation.get('experiment')}:{equation.get('variant')}")
        print(f"equation: {equation['candidate_equation']}")
        print(f"alpha: {equation['alpha']}")
        print(f"beta: {equation['beta']}")
        print(f"coefficients: constant={equation['constant']}, H={equation['h']}, Hx={equation['hx']}, fractional={equation['fractional']}")
        print(f"refit_coefficients: {bool(args.refit_coefficients)}")
        print(f"full_npz: {full_npz}")
        print(f"checkpoint: {checkpoint}")
        print(f"split_time: {split_time}")
        print(f"linear_figure: {linear_path}")
        if not args.linear_only:
            print(f"log_figure: {log_path}")
        print(f"validation_plume: {validation_plume_path}")
        print(f"prediction_npz: {npz_path}")
        print(f"metrics: {metrics_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    raw = np.asarray(data["c"], dtype=float)
    surrogate = reconstruct_nn_on_raw_grid(checkpoint, x, t)
    if args.refit_coefficients:
        equation = _refit_coefficients(checkpoint=checkpoint, x=x, t=t, equation=equation)
    prediction = _fft_prediction(x=x, t=t, surrogate=surrogate, equation=equation)
    display_prediction = np.clip(prediction, 0.0, None)

    if raw.shape != surrogate.shape or raw.shape != prediction.shape:
        raise ValueError(f"shape mismatch: raw={raw.shape}, surrogate={surrogate.shape}, prediction={prediction.shape}")
    initial_error = float(np.max(np.abs(prediction[0, :] - surrogate[0, :])))
    if not np.isfinite(prediction).all():
        raise ValueError("FFT prediction contains NaN or Inf")

    vmax = _shared_vmax([raw, surrogate, display_prediction], float(args.vmax_quantile))
    _plot_heatmaps(
        linear_path,
        x=x,
        t=t,
        raw=raw,
        surrogate=surrogate,
        prediction=prediction,
        split_time=split_time,
        vmax=vmax,
        log_scale=False,
        equation=equation,
        metadata=metadata,
    )
    _plot_heatmaps(
        log_path,
        x=x,
        t=t,
        raw=raw,
        surrogate=surrogate,
        prediction=prediction,
        split_time=split_time,
        vmax=vmax,
        log_scale=True,
        equation=equation,
        metadata=metadata,
    ) if not args.linear_only else None
    _plot_validation_plume(
        validation_plume_path,
        x=x,
        t=t,
        raw=raw,
        prediction=prediction,
        metadata=metadata,
        equation=equation,
    )

    metrics = {
        "summary": str(args.summary),
        "summary_row_line": int(args.summary_row_line),
        "run_id": equation["run_id"],
        "experiment": equation.get("experiment"),
        "variant": equation.get("variant"),
        "equation": equation["candidate_equation"],
        "coefficients": {
            "constant": equation["constant"],
            "H": equation["h"],
            "Hx": equation["hx"],
            "D_x_beta": equation["fractional"],
        },
        "refit": equation.get("refit"),
        "active_terms_original": equation.get("active_terms_original"),
        "alpha": equation["alpha"],
        "beta": equation["beta"],
        "shape": list(raw.shape),
        "full_npz": str(full_npz),
        "checkpoint": str(checkpoint),
        "split_time": split_time,
        "initial_max_abs_error_vs_surrogate": initial_error,
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "shared_vmax_quantile": float(args.vmax_quantile),
        "shared_vmax": vmax,
        "slice_metrics": _profile_metrics(x, raw, surrogate, prediction, t, metadata),
        "linear_figure": str(linear_path),
        "log_figure": None if args.linear_only else str(log_path),
        "validation_plume_figure": str(validation_plume_path),
        "prediction_npz": str(npz_path),
    }

    np.savez_compressed(
        npz_path,
        x=x,
        t=t,
        raw=raw,
        surrogate=surrogate,
        fft_prediction=prediction,
        fft_prediction_display=display_prediction,
        alpha=np.array(float(equation["alpha"])),
        beta=np.array(float(equation["beta"])),
        constant=np.array(float(equation["constant"])),
        coef_h=np.array(float(equation["h"])),
        coef_hx=np.array(float(equation["hx"])),
        coef_fractional=np.array(float(equation["fractional"])),
        split_time=np.array(split_time),
        summary_row_line=np.array(int(args.summary_row_line)),
        run_id=np.array(str(equation["run_id"])),
    )
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    print(f"shape: {raw.shape}")
    print(f"initial_max_abs_error_vs_surrogate: {initial_error:.6e}")
    print(f"linear: {linear_path}")
    if not args.linear_only:
        print(f"log: {log_path}")
    print(f"validation_plume: {validation_plume_path}")
    print(f"prediction_npz: {npz_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
