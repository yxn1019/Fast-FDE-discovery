"""Plot hydrology raw-data slices against trained NN surrogate profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_raw_vs_nn_surrogate import reconstruct_nn_on_raw_grid


ROOT = Path(__file__).resolve().parents[1]


def _load_metadata(value: Any) -> dict[str, Any]:
    try:
        return json.loads(str(np.asarray(value).reshape(())))
    except Exception:
        return {}


def _plot_grid(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    raw: np.ndarray,
    nn: np.ndarray,
    metadata: dict[str, Any],
    log_y: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_slices = int(t.size)
    n_cols = 3 if n_slices > 6 else 2
    n_rows = int(np.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    holdout_time = metadata.get("holdout_time")
    time_unit = metadata.get("time_unit", "")
    experiment = metadata.get("experiment", "hydrology")
    variant = metadata.get("variant", "")
    y_positive = raw[raw > 0.0]
    y_floor = float(max(np.nanmin(y_positive) * 0.5, 1.0e-12)) if y_positive.size else 1.0e-12
    y_hi = float(max(np.nanmax(raw), np.nanmax(nn))) if raw.size and nn.size else 1.0
    if not np.isfinite(y_hi) or y_hi <= 0.0:
        y_hi = 1.0

    for index, ax in enumerate(axes.reshape(-1)):
        if index >= n_slices:
            ax.axis("off")
            continue
        raw_row = np.asarray(raw[index, :], dtype=float)
        nn_row = np.asarray(nn[index, :], dtype=float)
        is_holdout = holdout_time is not None and np.isclose(float(t[index]), float(holdout_time))
        title = f"t={float(t[index]):g} {time_unit}".strip()
        if is_holdout:
            title += " (holdout)"
        if log_y:
            raw_mask = raw_row > 0.0
            nn_mask = nn_row > 0.0
            ax.semilogy(
                x[raw_mask],
                raw_row[raw_mask],
                linestyle="None",
                marker="o",
                markersize=3.2,
                markerfacecolor="none",
                markeredgecolor="#1f77b4",
                markeredgewidth=0.9,
                label="raw_data" if index == 0 else None,
            )
            ax.semilogy(
                x[nn_mask],
                nn_row[nn_mask],
                color="#d95f02",
                lw=1.8,
                label="nn_surrogate" if index == 0 else None,
            )
            ax.set_ylim(y_floor, y_hi * 1.8)
        else:
            ax.plot(
                x,
                raw_row,
                linestyle="None",
                marker="o",
                markersize=3.0,
                markerfacecolor="none",
                markeredgecolor="#1f77b4",
                markeredgewidth=0.8,
                label="raw_data" if index == 0 else None,
            )
            ax.plot(x, nn_row, color="#d95f02", lw=1.8, label="nn_surrogate" if index == 0 else None)
            ax.set_ylim(bottom=min(0.0, float(np.nanmin(raw_row)) * 1.1), top=y_hi * 1.12)
        if is_holdout:
            ax.set_facecolor("#FFF7ED")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("c" if not log_y else "c (log)")
        ax.grid(alpha=0.2, which="both")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    scale = "semilogy" if log_y else "linear"
    fig.suptitle(f"{experiment} {variant} raw_data vs nn_surrogate ({scale})", y=1.02, fontsize=13)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _metrics(raw: np.ndarray, nn: np.ndarray, t: np.ndarray, metadata: dict[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    holdout_time = metadata.get("holdout_time")
    for index, time_value in enumerate(t):
        raw_row = np.asarray(raw[index, :], dtype=float)
        nn_row = np.asarray(nn[index, :], dtype=float)
        finite = np.isfinite(raw_row) & np.isfinite(nn_row)
        positive = finite & (raw_row > 0.0)
        err = nn_row[finite] - raw_row[finite]
        raw_norm = max(float(np.linalg.norm(raw_row[finite])), np.finfo(float).eps)
        row: dict[str, float | str] = {
            "time": float(time_value),
            "split": "holdout"
            if holdout_time is not None and np.isclose(float(time_value), float(holdout_time))
            else "train",
            "rmse": float(np.sqrt(np.mean(err**2))) if err.size else float("nan"),
            "relative_l2": float(np.linalg.norm(err) / raw_norm) if err.size else float("nan"),
        }
        if np.any(positive):
            eps = 1.0e-3 * max(float(np.max(raw_row[positive])), np.finfo(float).eps)
            row["log_mse_positive"] = float(
                np.mean((np.log10(np.clip(nn_row[positive], 0.0, None) + eps) - np.log10(raw_row[positive] + eps)) ** 2)
            )
        else:
            row["log_mse_positive"] = float("nan")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.full_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    raw = np.asarray(data["c"], dtype=float)
    metadata = _load_metadata(data["metadata"]) if "metadata" in data else {}
    nn = reconstruct_nn_on_raw_grid(args.checkpoint, x, t)

    linear_path = args.output_prefix.with_name(args.output_prefix.name + "_linear.png")
    log_path = args.output_prefix.with_name(args.output_prefix.name + "_semilogy.png")
    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.json")
    _plot_grid(linear_path, x=x, t=t, raw=raw, nn=nn, metadata=metadata, log_y=False)
    _plot_grid(log_path, x=x, t=t, raw=raw, nn=nn, metadata=metadata, log_y=True)
    metrics = {
        "full_npz": str(args.full_npz),
        "checkpoint": str(args.checkpoint),
        "linear_figure": str(linear_path),
        "semilogy_figure": str(log_path),
        "slices": _metrics(raw, nn, t, metadata),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"linear: {linear_path}")
    print(f"semilogy: {log_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
