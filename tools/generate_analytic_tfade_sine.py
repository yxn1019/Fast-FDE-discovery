"""Generate and discover an analytic time-fractional advection-diffusion case.

PDE:
    D_t^alpha c = D c_xx - v c_x, x in (0, pi), c(0,t)=c(pi,t)=0.

Analytic solution:
    c(x,t) = exp(v*x/(2D)) * sin(x) * E_alpha(-lambda*t^alpha),
    lambda = D + v^2/(4D).

The script writes a MATLAB-compatible dataset, residual diagnostics, figures,
and a small raw-data Laplace discovery summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.special import gamma, roots_laguerre

ROOT = Path(__file__).resolve().parents[1]


def mittag_leffler_series(z: np.ndarray, alpha: float, terms: int = 100, tol: float = 1e-15) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z, dtype=float)
    power = np.ones_like(z, dtype=float)
    for k in range(terms):
        if k > 0:
            power = power * z
        term = power / gamma(alpha * k + 1.0)
        result = result + term
        if np.max(np.abs(term)) < tol:
            break
    return result


def finite_differences_x(c: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    cx = np.empty_like(c)
    cxx = np.empty_like(c)
    cx[:, 1:-1] = (c[:, 2:] - c[:, :-2]) / (2.0 * dx)
    cxx[:, 1:-1] = (c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]) / (dx * dx)
    cx[:, 0] = (c[:, 1] - c[:, 0]) / dx
    cx[:, -1] = (c[:, -1] - c[:, -2]) / dx
    cxx[:, 0] = (c[:, 2] - 2.0 * c[:, 1] + c[:, 0]) / (dx * dx)
    cxx[:, -1] = (c[:, -1] - 2.0 * c[:, -2] + c[:, -3]) / (dx * dx)
    return cx, cxx


def caputo_l1(c: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """Uniform-grid L1 approximation, returns same shape with row 0 as NaN."""

    dt = float(t[1] - t[0])
    nt = t.size
    out = np.full_like(c, np.nan)
    prefactor = 1.0 / (gamma(2.0 - alpha) * dt**alpha)
    weights = np.array([(j + 1) ** (1.0 - alpha) - j ** (1.0 - alpha) for j in range(nt)], dtype=float)
    increments = c[1:, :] - c[:-1, :]
    for n in range(1, nt):
        # sum_{k=0}^{n-1} b_k (u_{n-k}-u_{n-k-1})
        out[n, :] = prefactor * np.tensordot(weights[:n], increments[n - 1 :: -1, :], axes=(0, 0))
    return out


def sample_time(values: np.ndarray, tau: np.ndarray, query_tau: np.ndarray) -> np.ndarray:
    flat = values.reshape((values.shape[0], -1))
    sampled = np.empty((query_tau.shape[0], query_tau.shape[1], flat.shape[1]), dtype=float)
    for col in range(flat.shape[1]):
        sampled[:, :, col] = np.interp(query_tau, tau, flat[:, col], left=flat[0, col], right=flat[-1, col])
    return sampled.reshape((query_tau.shape[0], query_tau.shape[1], values.shape[1]))


def laplace_laguerre(values: np.ndarray, t: np.ndarray, s_values: np.ndarray, nodes_count: int) -> tuple[np.ndarray, float]:
    nodes, weights = roots_laguerre(nodes_count)
    tau = t - t[0]
    query_tau = nodes.reshape(-1, 1) / s_values.reshape(1, -1)
    out_of_bounds = (query_tau < tau[0]) | (query_tau > tau[-1])
    sampled = sample_time(values, tau, query_tau)
    transformed = np.einsum("n,nsx->sx", weights, sampled) / s_values.reshape(-1, 1)
    return transformed, float(np.mean(out_of_bounds))


def write_plots(x: np.ndarray, t: np.ndarray, c: np.ndarray, residual: np.ndarray, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, c, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("c")
    fig.tight_layout()
    fig.savefig(output_dir / "analytic_tfade_solution_surface.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(
        np.abs(residual[1:, 1:-1]),
        origin="lower",
        aspect="auto",
        extent=[x[1], x[-2], t[1], t[-1]],
        cmap="magma",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(image, ax=ax, label="|residual|")
    fig.tight_layout()
    fig.savefig(output_dir / "analytic_tfade_residual_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--D", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=-1.0)
    parser.add_argument("--ml-terms", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "analytic_tfade_sine")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, np.pi, args.nx)
    t = np.linspace(0.0, args.t_max, args.nt)
    X, T = np.meshgrid(x, t)
    lam = args.D + args.v**2 / (4.0 * args.D)
    spatial = np.exp((args.v / (2.0 * args.D)) * X) * np.sin(X)
    temporal = mittag_leffler_series(-lam * T**args.alpha, args.alpha, terms=args.ml_terms)
    c = spatial * temporal
    dx = float(x[1] - x[0])
    cx, cxx = finite_differences_x(c, dx)
    caputo = caputo_l1(c, t, args.alpha)
    rhs = args.D * cxx - args.v * cx
    residual = caputo - rhs
    residual_abs = np.abs(residual[1:, 1:-1])

    mat_path = output_dir / "analytic_tfade_sine.mat"
    savemat(
        mat_path,
        {
            "x": x.reshape(1, -1),
            "t": t.reshape(1, -1),
            "Exact": c,
            "C": c,
            "cx": cx,
            "cxx": cxx,
            "caputo_l1": caputo,
            "residual": residual,
            "alpha": float(args.alpha),
            "D": float(args.D),
            "v": float(args.v),
            "lambda": float(lam),
        },
    )
    write_plots(x, t, c, residual, output_dir)

    summary = output_dir / "analytic_tfade_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"alpha_true: {args.alpha}",
                f"D_true: {args.D}",
                f"v_true: {args.v}",
                f"lambda: {lam}",
                f"grid: Nx={args.nx}, Nt={args.nt}, t_max={args.t_max}",
                f"boundary max |c(0,t)|: {np.max(np.abs(c[:, 0])):.12e}",
                f"boundary max |c(pi,t)|: {np.max(np.abs(c[:, -1])):.12e}",
                f"initial max error: {np.max(np.abs(c[0, :] - np.exp((args.v/(2*args.D))*x)*np.sin(x))):.12e}",
                f"L1 residual max abs: {np.nanmax(residual_abs):.12e}",
                f"L1 residual mean abs: {np.nanmean(residual_abs):.12e}",
                f"mat: {mat_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved mat: {mat_path}")
    print(f"boundary max: left={np.max(np.abs(c[:, 0])):.3e}, right={np.max(np.abs(c[:, -1])):.3e}")
    print(f"initial max error: {np.max(np.abs(c[0, :] - np.exp((args.v/(2*args.D))*x)*np.sin(x))):.3e}")
    print(f"L1 residual: max={np.nanmax(residual_abs):.3e}, mean={np.nanmean(residual_abs):.3e}")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
