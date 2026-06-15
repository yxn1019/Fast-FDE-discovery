"""Plot held-out DNS profile against the discovered-equation prediction."""

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
    profile_stats,
    log_mse,
)


ROOT = Path(__file__).resolve().parents[1]


def profile_metrics(
    label: str,
    y: np.ndarray,
    *,
    x: np.ndarray,
    dx: float,
    dns: np.ndarray,
    dns_norm: np.ndarray,
    mask_2pct: np.ndarray,
    mask_positive: np.ndarray,
    epsilon: float,
) -> dict[str, float | str]:
    y_pos = np.clip(np.asarray(y, dtype=float), 0.0, None)
    y_norm, mass = normalize_mass(y_pos, dx)
    raw_dns = np.clip(np.asarray(dns, dtype=float), 0.0, None)
    stats = profile_stats(y_pos, x, dx, dns_norm, mask_2pct, epsilon)
    return {
        "label": label,
        "mass": mass,
        "raw_mse": float(np.mean((y_pos - raw_dns) ** 2)),
        "raw_rel_l2": float(np.linalg.norm(y_pos - raw_dns) / max(np.linalg.norm(raw_dns), np.finfo(float).eps)),
        "norm_mse_all": float(np.mean((y_norm - dns_norm) ** 2)),
        "norm_mse_positive": float(np.mean((y_norm[mask_positive] - dns_norm[mask_positive]) ** 2)),
        "relative_l2_mass_normalized": float(stats["relative_l2_mass_normalized"]),
        "log_mse_2pct": log_mse(y_norm, dns_norm, mask_2pct, epsilon),
        "log_mse_positive": log_mse(y_norm, dns_norm, mask_positive, epsilon),
        "mean_x": float(stats["mean_x"]),
        "std_x": float(stats["std_x"]),
        "skewness": float(stats["skewness"]),
        "peak_x": float(stats["peak_x"]),
        "peak_value": float(stats["peak_value"]),
    }


def plot_linear(path: Path, *, x: np.ndarray, x0: float, dns_norm: np.ndarray, prediction_norm: np.ndarray) -> None:
    distance = x - float(x0)
    mask = distance > 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(
        distance[mask],
        dns_norm[mask],
        linestyle="None",
        marker="o",
        markersize=3.5,
        markerfacecolor="none",
        markeredgecolor="#1f77b4",
        markeredgewidth=0.9,
        alpha=0.72,
        label="DNS at t=100",
    )
    ax.plot(distance[mask], prediction_norm[mask], color="#d95f02", lw=2.0, ls="--", label="prediction")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$x-x_0$")
    ax.set_ylabel(r"mass-normalized $c(x,t)$")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_loglog(
    path: Path,
    *,
    x: np.ndarray,
    x0: float,
    dns_norm: np.ndarray,
    prediction_norm: np.ndarray,
    y_min: float,
) -> None:
    distance = x - float(x0)
    base_mask = distance > 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    dns_mask = base_mask & (dns_norm > 0.0)
    pred_mask = base_mask & (prediction_norm > 0.0)
    ax.loglog(
        distance[dns_mask],
        dns_norm[dns_mask],
        linestyle="None",
        marker="o",
        markersize=3.5,
        markerfacecolor="none",
        markeredgecolor="#1f77b4",
        markeredgewidth=0.9,
        alpha=0.72,
        label="DNS at t=100",
    )
    ax.loglog(distance[pred_mask], prediction_norm[pred_mask], color="#d95f02", lw=2.0, ls="--", label="prediction")
    y_hi = max(float(np.nanmax(dns_norm)), float(np.nanmax(prediction_norm))) * 1.35
    ax.set_ylim(bottom=float(y_min), top=y_hi)
    ax.set_xlabel(r"$x-x_0$")
    ax.set_ylabel(r"mass-normalized $c(x,t)$")
    ax.legend(frameon=False)
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
    parser.add_argument("--target-time", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--velocity", type=float, default=None)
    parser.add_argument("--nil-a", type=float, default=6.0)
    parser.add_argument("--nil-ns", type=int, default=20)
    parser.add_argument("--nil-nd", type=int, default=19)
    parser.add_argument("--log-y-min", type=float, default=1.0e-3)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=ROOT / "data" / "dns_gamma075_lc1_uniform_kmin1e6" / "forecast_t100_comparison.npz",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "results" / "dns_gamma075_lc1_uniform_kmin1e6_forecast_metrics.txt",
    )
    parser.add_argument(
        "--linear-figure",
        type=Path,
        default=ROOT / "figures" / "dns_gamma075_lc1_uniform_kmin1e6_forecast_t100_linear.png",
    )
    parser.add_argument(
        "--loglog-figure",
        type=Path,
        default=ROOT / "figures" / "dns_gamma075_lc1_uniform_kmin1e6_forecast_t100_loglog.png",
    )
    args = parser.parse_args()

    data = np.load(args.full_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    c = np.asarray(data["c"], dtype=float)
    target_index = int(np.argmin(np.abs(t - float(args.target_time))))
    target_time = float(t[target_index])
    dns = np.asarray(c[target_index, :], dtype=float)
    dx = float(np.median(np.diff(x)))
    metadata = load_metadata(data["metadata"])
    x0 = float(metadata.get("particle_release_x", 0.0))
    dns_norm, dns_mass = normalize_mass(dns, dx)
    peak_norm = float(np.max(dns_norm))
    epsilon = 1.0e-3 * peak_norm
    mask_positive = (x > x0) & (dns_norm > 0.0)
    mask_2pct = (x > x0) & (dns_norm > 0.02 * peak_norm)

    if args.alpha is None or args.velocity is None:
        alpha, velocity, equation = parse_discovery(args.report)
    else:
        alpha = float(args.alpha)
        velocity = float(args.velocity)
        equation = f"D_t^{alpha:.8g} H = {-velocity:.5g}*Hx"

    prediction = dirac_profile_vectorized(
        x,
        alpha=alpha,
        velocity=velocity,
        x0=x0,
        t=target_time,
        mass=1.0,
        nil_a=args.nil_a,
        nil_ns=args.nil_ns,
        nil_nd=args.nil_nd,
    )
    prediction_norm, prediction_mass = normalize_mass(prediction, dx)
    row = profile_metrics(
        "prediction vs DNS",
        prediction,
        x=x,
        dx=dx,
        dns=dns,
        dns_norm=dns_norm,
        mask_2pct=mask_2pct,
        mask_positive=mask_positive,
        epsilon=epsilon,
    )

    plot_linear(args.linear_figure, x=x, x0=x0, dns_norm=dns_norm, prediction_norm=prediction_norm)
    plot_loglog(args.loglog_figure, x=x, x0=x0, dns_norm=dns_norm, prediction_norm=prediction_norm, y_min=args.log_y_min)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        x=x,
        t_target=np.array(target_time),
        dns=dns,
        dns_mass_normalized=dns_norm,
        prediction=prediction,
        prediction_mass_normalized=prediction_norm,
        alpha=np.array(alpha),
        velocity=np.array(velocity),
        theta=np.array(-velocity),
        x0=np.array(x0),
        log_epsilon=np.array(epsilon),
        mask_2pct=mask_2pct,
    )

    lines = [
        "Uniform DNS t=100 prediction comparison",
        f"full_npz: {args.full_npz}",
        f"report: {args.report}",
        f"equation: {equation}",
        f"alpha: {alpha}",
        f"velocity: {velocity}",
        f"theta: {-velocity}",
        f"target_time: {target_time}",
        f"x0: {x0}",
        f"dns_mass: {dns_mass}",
        f"prediction_mass: {prediction_mass}",
        f"log_epsilon: {epsilon}",
        "",
        f"[{row['label']}]",
    ]
    lines.extend(f"{key}: {value}" for key, value in row.items() if key != "label")
    lines.extend(
        [
            "",
            f"output_npz: {args.output_npz}",
            f"linear_figure: {args.linear_figure}",
            f"loglog_figure: {args.loglog_figure}",
            "",
        ]
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
