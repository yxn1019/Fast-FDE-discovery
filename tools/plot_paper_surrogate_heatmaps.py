"""Create paper-ready raw/NN/error heatmaps for tfade and tsfade surrogates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from plot_raw_vs_nn_surrogate import checkpoint_config, load_raw_field, noisy_observation, reconstruct_nn_on_raw_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(r"D:\OneDrive - HHU\My paper\11Laplace-Taylor discovery\figures"),
    )
    parser.add_argument(
        "--tsfade-summary",
        type=Path,
        default=None,
        help="optional legacy training summary; omitted by default so paper plots use repo-local data",
    )
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def load_mat_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = loadmat(path)
    x = np.asarray(payload["x"], dtype=float).reshape(-1)
    t = np.asarray(payload["t"], dtype=float).reshape(-1)
    key = "c" if "c" in payload else "Exact"
    values = np.asarray(payload[key], dtype=float)
    if values.shape == (x.size, t.size):
        values = values.T
    return x, t, values


def robust_limit(values: list[np.ndarray], percentile: float = 99.0) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in values])
    lo, hi = np.percentile(flat[np.isfinite(flat)], [100.0 - percentile, percentile])
    if np.isclose(lo, hi):
        delta = max(abs(float(hi)), 1.0) * 1.0e-3
        return float(lo - delta), float(hi + delta)
    return float(lo), float(hi)


def error_limit(values: list[np.ndarray], percentile: float = 99.5) -> float:
    flat = np.concatenate([np.abs(np.asarray(item, dtype=float)).reshape(-1) for item in values])
    hi = np.percentile(flat[np.isfinite(flat)], percentile)
    return float(max(hi, np.finfo(float).eps))


def metrics(reference: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - reference
    return {
        "relative_l2": float(np.linalg.norm(err) / max(np.linalg.norm(reference), np.finfo(float).eps)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "p95_abs_error": float(np.percentile(np.abs(err), 95)),
        "p99_abs_error": float(np.percentile(np.abs(err), 99)),
    }


def save_panel(
    *,
    title: str,
    records: list[dict[str, Any]],
    output_base: Path,
    formats: list[str],
    raw_label: str,
) -> list[dict[str, Any]]:
    raw_values = [item["raw"] for item in records]
    nn_values = [item["nn"] for item in records]
    err_values = [item["nn"] - item["reference"] for item in records]
    vmin, vmax = robust_limit(raw_values + nn_values)
    emax = error_limit(err_values)

    fig, axes = plt.subplots(
        len(records),
        3,
        figsize=(10.4, 2.75 * len(records)),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if len(records) == 1:
        axes = np.asarray([axes])

    for row, item in enumerate(records):
        x = item["x"]
        t = item["t"]
        extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
        panels = [
            (item["raw"], raw_label, "viridis", vmin, vmax),
            (item["nn"], "NN surrogate", "viridis", vmin, vmax),
            (np.abs(item["nn"] - item["reference"]), "|NN - exact|", "magma", 0.0, emax),
        ]
        for col, (values, panel_title, cmap, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(panel_title if row == 0 else "")
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(f"{item['label']}\nt")
            else:
                ax.set_ylabel("")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=11)
    written = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)

    summary = []
    for item in records:
        summary.append(
            {
                "label": item["label"],
                "data_file": str(item["data_file"]),
                "checkpoint": str(item["checkpoint"]),
                "metrics_vs_exact": metrics(item["reference"], item["nn"]),
                "metrics_vs_raw": metrics(item["raw"], item["nn"]),
            }
        )
    return summary


def tsfade_records(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_mat = Path(payload["raw_data"]["mat_path"])
    x_exact, t_exact, exact = load_mat_field(raw_mat)
    records = []
    for item in payload["training"]:
        checkpoint = Path(item["checkpoint"])
        noisy_mat = Path(item["noisy_data_mat"])
        x_raw, t_raw, raw = load_mat_field(noisy_mat)
        if not (np.array_equal(x_exact, x_raw) and np.array_equal(t_exact, t_raw)):
            raise ValueError(f"grid mismatch for {noisy_mat}")
        nn = reconstruct_nn_on_raw_grid(checkpoint, x_exact, t_exact)
        noise = float(item["noise_level"])
        label = "clean" if np.isclose(noise, 0.0) else f"{noise:g}% noise"
        records.append(
            {
                "label": label,
                "x": x_exact,
                "t": t_exact,
                "raw": raw,
                "reference": exact,
                "nn": nn,
                "data_file": noisy_mat,
                "checkpoint": checkpoint,
            }
        )
    return records


def tsfade_repo_records() -> list[dict[str, Any]]:
    raw_root = ROOT / "data" / "tsfade_retrained_alpha078_beta183" / "raw_data"
    model_root = ROOT / "data" / "models" / "tsfade_retrained_alpha078_beta183_normalized_tanh"
    reference_mat = raw_root / "tsfade_alpha078_beta183_paper_grid.mat"
    reference_payload = loadmat(reference_mat)
    x_exact = np.asarray(reference_payload["x"], dtype=float).reshape(-1)
    t_exact = np.asarray(reference_payload["t"], dtype=float).reshape(-1)
    exact = np.asarray(reference_payload["Exact"], dtype=float)
    if exact.shape == (x_exact.size, t_exact.size):
        exact = exact.T

    specs = [
        ("clean", 0, raw_root / "tsfade_alpha078_beta183_noise0.mat", model_root / "draft-2000-0" / "best.pkl"),
        ("5% noise", 5, raw_root / "tsfade_alpha078_beta183_noise5.mat", model_root / "draft-2000-5" / "best.pkl"),
        ("25% noise", 25, raw_root / "tsfade_alpha078_beta183_noise25.mat", model_root / "draft-2000-25" / "best.pkl"),
    ]
    records = []
    for label, _noise, noisy_mat, checkpoint in specs:
        payload = loadmat(noisy_mat)
        x_raw = np.asarray(payload["x"], dtype=float).reshape(-1)
        t_raw = np.asarray(payload["t"], dtype=float).reshape(-1)
        raw = np.asarray(payload["c"], dtype=float)
        if raw.shape == (x_raw.size, t_raw.size):
            raw = raw.T
        if not (np.array_equal(x_exact, x_raw) and np.array_equal(t_exact, t_raw)):
            raise ValueError(f"grid mismatch for {noisy_mat}")
        nn = reconstruct_nn_on_raw_grid(checkpoint, x_exact, t_exact)
        records.append(
            {
                "label": label,
                "x": x_exact,
                "t": t_exact,
                "raw": raw,
                "reference": exact,
                "nn": nn,
                "data_file": noisy_mat,
                "checkpoint": checkpoint,
            }
        )
    return records


def tfade_records() -> list[dict[str, Any]]:
    data_file = ROOT / "data" / "periodic_tfade_fft" / "periodic_tfade_fft.mat"
    checkpoints = [
        (
            "clean",
            ROOT
            / "data"
            / "models"
            / "periodic_tfade_fft_sin_5x50_clean_6000_spectral_w300_early5_dx002_periodic_w1_20000"
            / "best.pkl",
        ),
        (
            "10% noise",
            ROOT / "data" / "models" / "periodic_tfade_fft_sin_5x50_noise10_spectral_w300_early5_dx002_periodic_w1" / "best.pkl",
        ),
        (
            "20% noise",
            ROOT / "data" / "models" / "periodic_tfade_fft_sin_5x50_noise20_spectral_w300_early5_dx002_periodic_w1" / "best.pkl",
        ),
    ]
    x, t, exact = load_raw_field(data_file)
    records = []
    for label, checkpoint in checkpoints:
        config = checkpoint_config(checkpoint)
        raw = noisy_observation(exact, config)
        nn = reconstruct_nn_on_raw_grid(checkpoint, x, t)
        records.append(
            {
                "label": label,
                "x": x,
                "t": t,
                "raw": raw,
                "reference": exact,
                "nn": nn,
                "data_file": data_file,
                "checkpoint": checkpoint,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]

    summary = {
        "tsfade": save_panel(
            title="Space-time FADE surrogate reconstruction",
            records=tsfade_records(args.tsfade_summary) if args.tsfade_summary is not None else tsfade_repo_records(),
            output_base=args.figure_dir / "tsfade_retrained_surrogate_heatmaps",
            formats=formats,
            raw_label="raw data",
        ),
        "tfade": save_panel(
            title="Periodic time-FADE full-domain surrogate diagnostics",
            records=tfade_records(),
            output_base=args.figure_dir / "tfade_surrogate_heatmaps",
            formats=formats,
            raw_label="raw data",
        ),
    }
    summary_path = args.figure_dir / "surrogate_heatmap_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    for case in ("tsfade", "tfade"):
        print(f"{case}_rows={len(summary[case])}")
        for row in summary[case]:
            metrics_text = row["metrics_vs_exact"]
            print(
                f"{case} {row['label']}: rel_l2={metrics_text['relative_l2']:.6g}, "
                f"rmse={metrics_text['rmse']:.6g}, max={metrics_text['max_abs_error']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
