"""Plot full-domain DNS and discovered-equation prediction heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dns_laplace_utils import (
    dirac_profile_vectorized,
    load_metadata,
    normalize_mass,
    parse_discovery,
)


ROOT = Path(__file__).resolve().parents[1]


def _prediction_grid(
    x: np.ndarray,
    t: np.ndarray,
    *,
    alpha: float,
    velocity: float,
    x0: float,
    dx: float,
    nil_a: float,
    nil_ns: int,
    nil_nd: int,
) -> np.ndarray:
    pred = np.zeros((t.size, x.size), dtype=float)
    x0_index = int(np.argmin(np.abs(x - float(x0))))
    for i, time_value in enumerate(t):
        if float(time_value) <= 0.0:
            pred[i, x0_index] = 1.0 / dx
            continue
        pred[i, :] = dirac_profile_vectorized(
            x,
            alpha=alpha,
            velocity=velocity,
            x0=x0,
            t=float(time_value),
            mass=1.0,
            nil_a=nil_a,
            nil_ns=nil_ns,
            nil_nd=nil_nd,
        )
    return np.clip(pred, 0.0, None)


def _shared_vmax(dns: np.ndarray, pred: np.ndarray, t: np.ndarray, quantile: float) -> float:
    rows = np.asarray(t, dtype=float) > 0.0
    values = np.concatenate(
        [
            np.clip(np.asarray(dns[rows, :], dtype=float).reshape(-1), 0.0, None),
            np.clip(np.asarray(pred[rows, :], dtype=float).reshape(-1), 0.0, None),
        ]
    )
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 1.0
    vmax = float(np.quantile(positive, float(quantile)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(positive))
    return vmax


def _oos_metrics(
    dns: np.ndarray,
    pred: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    *,
    split_time: float,
    dx: float,
) -> dict[str, float]:
    rows = np.asarray(t, dtype=float) > float(split_time)
    if not np.any(rows):
        return {
            "oos_snapshot_count": 0.0,
            "mean_relative_l2_mass_normalized": float("nan"),
            "mean_log_mse_2pct": float("nan"),
            "mean_peak_location_error": float("nan"),
            "mean_mass_error": float("nan"),
        }
    rel_l2: list[float] = []
    log_mse: list[float] = []
    peak_error: list[float] = []
    mass_error: list[float] = []
    for dns_row, pred_row in zip(dns[rows, :], pred[rows, :]):
        dns_norm, dns_mass = normalize_mass(dns_row, dx)
        pred_norm, pred_mass = normalize_mass(pred_row, dx)
        denom = max(float(np.linalg.norm(dns_norm)), np.finfo(float).eps)
        rel_l2.append(float(np.linalg.norm(pred_norm - dns_norm) / denom))
        peak = float(np.max(dns_norm))
        mask = dns_norm > 0.02 * peak
        if np.any(mask):
            eps = 1.0e-3 * max(peak, np.finfo(float).eps)
            residual = np.log10(pred_norm[mask] + eps) - np.log10(dns_norm[mask] + eps)
            log_mse.append(float(np.mean(residual**2)))
        peak_error.append(float(abs(x[int(np.argmax(pred_norm))] - x[int(np.argmax(dns_norm))])))
        mass_error.append(float(pred_mass - dns_mass))
    return {
        "oos_snapshot_count": float(np.count_nonzero(rows)),
        "mean_relative_l2_mass_normalized": float(np.mean(rel_l2)),
        "mean_log_mse_2pct": float(np.mean(log_mse)) if log_mse else float("nan"),
        "mean_peak_location_error": float(np.mean(peak_error)),
        "mean_mass_error": float(np.mean(mass_error)),
    }


def _plot_heatmaps(
    path: Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    dns: np.ndarray,
    pred: np.ndarray,
    split_time: float,
    vmax: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharex=True, sharey=True, constrained_layout=True)
    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
    panels = [(dns, r"DNS $c(x,t)$"), (pred, r"prediction $c(x,t)$")]
    image = None
    for ax, (field, title) in zip(axes, panels):
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
        )
        ax.axhline(float(split_time), color="white", lw=1.4, ls="--", alpha=0.95)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.94, pad=0.015)
    cbar.set_label(r"$c(x,t)$")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-npz",
        type=Path,
        default=ROOT / "data" / "dns_gamma075_lc1_uniform_kmin1e6" / "dns_gamma075_lc1_uniform_kmin1e6.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "tsfade_fft_dns_gamma075_lc1_uniform_kmin1e6_gj_hybrid_taylor_result.txt",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--velocity", type=float, default=None)
    parser.add_argument("--split-time", type=float, default=80.0)
    parser.add_argument("--nil-a", type=float, default=6.0)
    parser.add_argument("--nil-ns", type=int, default=20)
    parser.add_argument("--nil-nd", type=int, default=19)
    parser.add_argument("--vmax-quantile", type=float, default=0.995)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=ROOT / "data" / "dns_gamma075_lc1_uniform_kmin1e6" / "forecast_full_domain_t80split.npz",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "results" / "dns_gamma075_lc1_uniform_kmin1e6_full_domain_t80split_metrics.txt",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures" / "dns_gamma075_lc1_uniform_kmin1e6_full_domain_t80split_heatmaps.png",
    )
    args = parser.parse_args()

    data = np.load(args.full_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    dns = np.clip(np.asarray(data["c"], dtype=float), 0.0, None)
    dx = float(np.median(np.diff(x)))
    metadata = load_metadata(data["metadata"])
    x0 = float(metadata.get("particle_release_x", 0.0))
    if args.alpha is None or args.velocity is None:
        alpha, velocity, equation = parse_discovery(args.report)
    else:
        alpha = float(args.alpha)
        velocity = float(args.velocity)
        equation = f"D_t^{alpha:.8g} H = {-velocity:.5g}*Hx"

    pred = _prediction_grid(
        x,
        t,
        alpha=alpha,
        velocity=velocity,
        x0=x0,
        dx=dx,
        nil_a=args.nil_a,
        nil_ns=args.nil_ns,
        nil_nd=args.nil_nd,
    )
    vmax = _shared_vmax(dns, pred, t, args.vmax_quantile)
    metrics = _oos_metrics(dns, pred, x, t, split_time=args.split_time, dx=dx)
    _plot_heatmaps(args.figure, x=x, t=t, dns=dns, pred=pred, split_time=args.split_time, vmax=vmax)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        x=x,
        t=t,
        dns=dns,
        prediction=pred,
        alpha=np.array(alpha),
        velocity=np.array(velocity),
        theta=np.array(-velocity),
        x0=np.array(x0),
        split_time=np.array(float(args.split_time)),
        vmax=np.array(vmax),
    )

    lines = [
        "Uniform DNS full-domain prediction heatmap comparison",
        f"full_npz: {args.full_npz}",
        f"report: {args.report}",
        f"equation: {equation}",
        f"alpha: {alpha}",
        f"velocity: {velocity}",
        f"theta: {-velocity}",
        f"x0: {x0}",
        f"x_range: ({float(x[0])}, {float(x[-1])})",
        f"t_range: ({float(t[0])}, {float(t[-1])})",
        f"split_time: {float(args.split_time)}",
        f"fit_last_snapshot_at_or_before_split: {max(float(value) for value in t if value <= float(args.split_time))}",
        f"first_oos_snapshot_after_split: {min(float(value) for value in t if value > float(args.split_time))}",
        "t0_prediction_convention: discrete Dirac bin at nearest x0 with value 1/dx; shared color limit excludes t=0 prediction row through positive-time quantile",
        f"shared_vmax_quantile: {float(args.vmax_quantile)}",
        f"shared_vmax: {vmax}",
    ]
    lines.extend(f"{key}: {value}" for key, value in metrics.items())
    lines.extend(
        [
            f"output_npz: {args.output_npz}",
            f"figure: {args.figure}",
            "",
        ]
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
