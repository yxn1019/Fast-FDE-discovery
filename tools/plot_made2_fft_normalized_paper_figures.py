"""Regenerate MADE-site paper figures from the no-refit FFT discovery solution.

The plotted profiles and heatmaps are normalized at each time slice by the
spatial integral of the observed nonnegative concentration profile. The
observations, surrogate, and discovered model therefore share the same
profile-wise reference mass.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "figures" / "hydrology_fft_discovery" / "made2_row252" / "made2_row252_prediction.npz"
DEFAULT_PAPER_FIG_DIR = ROOT / "figures" / "paper"


def _safe_profile(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    return np.clip(values, 0.0, None)


def _normalize_profile(x: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, float]:
    clean = _safe_profile(profile)
    area = float(np.trapezoid(clean, x))
    if not np.isfinite(area) or area <= np.finfo(float).eps:
        return np.zeros_like(clean), area
    return clean / area, area


def _normalize_field_by_own_area(x: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    areas = []
    for row in np.asarray(field, dtype=float):
        normalized, area = _normalize_profile(x, row)
        rows.append(normalized)
        areas.append(area)
    return np.vstack(rows), np.asarray(areas, dtype=float)


def _normalize_by_reference_area(
    x: np.ndarray,
    field: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    reference_areas = []
    for row, reference_row in zip(np.asarray(field, dtype=float), np.asarray(reference, dtype=float)):
        reference_clean = _safe_profile(reference_row)
        reference_area = float(np.trapezoid(reference_clean, x))
        clean = _safe_profile(row)
        if not np.isfinite(reference_area) or reference_area <= np.finfo(float).eps:
            rows.append(np.zeros_like(clean))
        else:
            rows.append(clean / reference_area)
        reference_areas.append(reference_area)
    return np.vstack(rows), np.asarray(reference_areas, dtype=float)


def _normalize_by_constant_area(field: np.ndarray, area: float) -> np.ndarray:
    rows = []
    for row in np.asarray(field, dtype=float):
        clean = _safe_profile(row)
        if not np.isfinite(area) or area <= np.finfo(float).eps:
            rows.append(np.zeros_like(clean))
        else:
            rows.append(clean / float(area))
    return np.vstack(rows)


def _time_index(t: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t, dtype=float) - float(value))))


def _panel_label(time_value: float) -> str:
    return f"t={time_value:g} d"


def _plot_profiles(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw_norm: np.ndarray,
    pred_norm: np.ndarray,
    times: list[float],
    ncols: int,
    model_label: str,
    y_upper: float | None = None,
) -> list[dict[str, float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.1 * ncols, 2.35 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    metrics: list[dict[str, float]] = []
    for ax, requested_time in zip(axes.reshape(-1), times):
        idx = _time_index(t, requested_time)
        time_value = float(t[idx])
        observed = raw_norm[idx]
        predicted = pred_norm[idx]
        finite = np.isfinite(observed) & np.isfinite(predicted)
        denom = max(float(np.linalg.norm(observed[finite])), np.finfo(float).eps)
        rel_l2 = float(np.linalg.norm(predicted[finite] - observed[finite]) / denom)
        metrics.append({"time": time_value, "relative_l2": rel_l2})

        ax.plot(
            x,
            observed,
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            markeredgewidth=0.85,
            color="#1f77b4",
            label="observed",
        )
        ax.plot(x, predicted, color="#d95f02", lw=2.0, label=model_label)
        ax.set_title(_panel_label(time_value), fontsize=9.5, pad=4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Normalized concentration")
        ax.set_ylim(bottom=0.0, top=y_upper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        if len(metrics) == 1:
            ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    for ax in axes.reshape(-1)[len(times) :]:
        ax.axis("off")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return metrics


def _edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.array([arr[0] - 0.5, arr[0] + 0.5])
    mids = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - 0.5 * (arr[1] - arr[0])
    last = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate([[first], mids, [last]])


def _plot_heatmaps(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw_norm: np.ndarray,
    surrogate_norm: np.ndarray,
    pred_norm: np.ndarray,
    split_time: float,
    model_label: str,
    x_max: float | None = None,
    scale: str = "linear",
    log_vmin_quantile: float = 0.01,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("Observed", raw_norm),
        ("NN surrogate", surrogate_norm),
        (model_label, pred_norm),
    ]
    x_plot = np.asarray(x, dtype=float)
    if x_max is not None:
        x_mask = x_plot <= float(x_max)
        if int(np.count_nonzero(x_mask)) < 2:
            raise ValueError(f"heatmap x-window has fewer than two points for x_max={x_max}")
        x_plot = x_plot[x_mask]
        fields = [(title, np.asarray(field)[:, x_mask]) for title, field in fields]

    positive = np.concatenate([field.reshape(-1) for _, field in fields])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    vmax = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    vmax = max(vmax, np.finfo(float).eps)
    norm = None
    vmin = 0.0
    cmap = plt.get_cmap("viridis").copy()
    if scale == "log":
        clipped_quantile = min(max(float(log_vmin_quantile), 0.0), 0.5)
        vmin = float(np.quantile(positive, clipped_quantile)) if positive.size else np.finfo(float).eps
        vmin = max(vmin, np.finfo(float).eps)
        if vmin >= vmax:
            vmin = max(float(np.min(positive)) if positive.size else np.finfo(float).eps, np.finfo(float).eps)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap.set_bad("#f2f2f2")
    elif scale != "linear":
        raise ValueError(f"Unsupported heatmap scale: {scale}")

    x_edges = _edges(x_plot)
    t_edges = _edges(t)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, (title, field) in zip(axes, fields):
        plot_field = np.asarray(field, dtype=float)
        if scale == "log":
            plot_field = np.ma.masked_where(~np.isfinite(plot_field) | (plot_field <= 0.0), plot_field)
        image = ax.pcolormesh(
            x_edges,
            t_edges,
            plot_field,
            shading="auto",
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
        )
        ax.axhline(float(split_time), color="white", lw=1.2, ls="--", alpha=0.95)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_xlabel("x (m)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    axes[0].set_ylabel("t (d)")
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.015)
    cbar_label = "Normalized concentration"
    if scale == "log":
        cbar_label += " (log scale)"
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument(
        "--normalization-mode",
        choices=["own_profile_mass", "observed_profile_mass", "initial_observed_mass"],
        default="own_profile_mass",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix inserted before .png/.json, useful for preview variants.",
    )
    parser.add_argument("--model-label", default="Discovered model")
    parser.add_argument("--heatmap-x-max", type=float, default=None, help="Optional maximum x shown in the heatmap.")
    parser.add_argument("--heatmap-scale", choices=["linear", "log"], default="linear")
    parser.add_argument(
        "--heatmap-log-vmin-quantile",
        type=float,
        default=0.01,
        help="Positive-value quantile used as the lower color bound for log heatmaps.",
    )
    args = parser.parse_args()

    input_npz = Path(args.input_npz)
    paper_fig_dir = Path(args.paper_fig_dir)
    data = np.load(input_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    raw = np.asarray(data["raw"], dtype=float)
    surrogate = np.asarray(data["surrogate"], dtype=float)
    prediction = np.asarray(data["fft_prediction"], dtype=float)
    split_time = float(np.asarray(data["split_time"]).reshape(())) if "split_time" in data else 370.0

    raw_area = np.asarray([float(np.trapezoid(_safe_profile(row), x)) for row in raw], dtype=float)
    if args.normalization_mode == "own_profile_mass":
        raw_norm, raw_area = _normalize_field_by_own_area(x, raw)
        surrogate_norm, surrogate_area = _normalize_field_by_own_area(x, surrogate)
        pred_norm, pred_area = _normalize_field_by_own_area(x, prediction)
        reference_area = None
        normalization_description = "each profile divided by its own spatial area after clipping negative concentrations to zero"
    elif args.normalization_mode == "initial_observed_mass":
        reference_area = float(raw_area[0])
        raw_norm = _normalize_by_constant_area(raw, reference_area)
        surrogate_norm = _normalize_by_constant_area(surrogate, reference_area)
        pred_norm = _normalize_by_constant_area(prediction, reference_area)
        normalization_description = "all profiles divided by the initial observed spatial area"
    else:
        raw_norm, raw_area = _normalize_by_reference_area(x, raw, raw)
        surrogate_norm, _ = _normalize_by_reference_area(x, surrogate, raw)
        pred_norm, _ = _normalize_by_reference_area(x, prediction, raw)
        reference_area = None
        normalization_description = "per-time observed spatial area after clipping negative concentrations to zero"
        _, surrogate_area = _normalize_field_by_own_area(x, surrogate)
        _, pred_area = _normalize_field_by_own_area(x, prediction)
    ref_idx = _time_index(t, 126.0)
    y_upper_126 = 1.08 * float(np.nanmax([np.nanmax(raw_norm[ref_idx]), np.nanmax(pred_norm[ref_idx])]))
    if not np.isfinite(y_upper_126) or y_upper_126 <= 0.0:
        y_upper_126 = None

    suffix = str(args.output_suffix)
    heatmap_path = paper_fig_dir / f"made2_field_heatmaps_linear{suffix}.png"
    training_path = paper_fig_dir / f"made2_field_training_profiles_linear{suffix}.png"
    validation_path = paper_fig_dir / f"made2_field_validation_profile_linear{suffix}.png"
    metrics_path = paper_fig_dir / f"made2_field_normalized_metrics{suffix}.json"

    _plot_heatmaps(
        heatmap_path,
        x=x,
        t=t,
        raw_norm=raw_norm,
        surrogate_norm=surrogate_norm,
        pred_norm=pred_norm,
        split_time=split_time,
        model_label=str(args.model_label),
        x_max=args.heatmap_x_max,
        scale=str(args.heatmap_scale),
        log_vmin_quantile=float(args.heatmap_log_vmin_quantile),
    )
    training_metrics = _plot_profiles(
        training_path,
        x=x,
        t=t,
        raw_norm=raw_norm,
        pred_norm=pred_norm,
        times=[126.0, 202.0, 279.0, 370.0],
        ncols=2,
        model_label=str(args.model_label).lower(),
        y_upper=y_upper_126,
    )
    validation_metrics = _plot_profiles(
        validation_path,
        x=x,
        t=t,
        raw_norm=raw_norm,
        pred_norm=pred_norm,
        times=[503.0],
        ncols=1,
        model_label=str(args.model_label).lower(),
        y_upper=y_upper_126,
    )

    metrics = {
        "source": str(input_npz),
        "normalization_mode": str(args.normalization_mode),
        "normalization": normalization_description,
        "initial_observed_area": reference_area,
        "alpha": float(np.asarray(data["alpha"]).reshape(())),
        "beta": float(np.asarray(data["beta"]).reshape(())),
        "constant": float(np.asarray(data["constant"]).reshape(())),
        "coef_h": float(np.asarray(data["coef_h"]).reshape(())),
        "coef_hx": float(np.asarray(data["coef_hx"]).reshape(())),
        "coef_fractional": float(np.asarray(data["coef_fractional"]).reshape(())),
        "raw_area": raw_area.tolist(),
        "surrogate_area": surrogate_area.tolist(),
        "discovered_area": pred_area.tolist(),
        "profile_y_upper_reference_time": 126.0,
        "profile_y_upper": y_upper_126,
        "heatmap_x_max": args.heatmap_x_max,
        "heatmap_scale": str(args.heatmap_scale),
        "heatmap_log_vmin_quantile": float(args.heatmap_log_vmin_quantile),
        "split_time": split_time,
        "training_profile_metrics": training_metrics,
        "validation_profile_metrics": validation_metrics,
        "figures": {
            "heatmap": str(heatmap_path),
            "training_profiles": str(training_path),
            "validation_profile": str(validation_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"heatmap: {heatmap_path}")
    print(f"training_profiles: {training_path}")
    print(f"validation_profile: {validation_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
