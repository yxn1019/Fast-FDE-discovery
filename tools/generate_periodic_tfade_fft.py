"""Generate a periodic 1D time-fractional advection-diffusion benchmark.

PDE:
    D_t^alpha c = -v c_x + D c_xx, x in [0, 2*pi), periodic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.special import gamma

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transporteq_discovery.mittag_leffler import mittag_leffler


def periodic_gaussian(x: np.ndarray, center: float, sigma: float, length: float) -> np.ndarray:
    plume = np.zeros_like(x, dtype=float)
    for shift in range(-3, 4):
        plume += np.exp(-((x - center + shift * length) ** 2) / (2.0 * sigma**2))
    return plume


def spectral_derivatives(c: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_hat = np.fft.fft(c, axis=1)
    cx = np.real(np.fft.ifft((1j * k).reshape(1, -1) * c_hat, axis=1))
    cxx = np.real(np.fft.ifft((-(k**2)).reshape(1, -1) * c_hat, axis=1))
    return cx, cxx


def caputo_l1(c: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    dt = float(t[1] - t[0])
    nt = t.size
    out = np.full_like(c, np.nan, dtype=float)
    prefactor = 1.0 / (gamma(2.0 - alpha) * dt**alpha)
    weights = np.array([(j + 1) ** (1.0 - alpha) - j ** (1.0 - alpha) for j in range(nt)], dtype=float)
    increments = c[1:, :] - c[:-1, :]
    for n in range(1, nt):
        out[n, :] = prefactor * np.tensordot(weights[:n], increments[n - 1 :: -1, :], axes=(0, 0))
    return out


def generate(
    *,
    nx: int = 256,
    nt: int = 201,
    t_max: float = 5.0,
    alpha: float = 0.85,
    diffusivity: float = 0.5,
    velocity: float = -1.0,
    sigma: float = 0.4,
) -> dict[str, np.ndarray | float]:
    length = 2.0 * np.pi
    x = np.linspace(0.0, length, nx, endpoint=False)
    t = np.linspace(0.0, t_max, nt)
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    c0 = periodic_gaussian(x, np.pi, sigma, length)
    c0_hat = np.fft.fft(c0)
    lam = -diffusivity * k**2 - 1j * velocity * k
    z = lam.reshape(1, -1) * (t**alpha).reshape(-1, 1)
    c_hat = c0_hat.reshape(1, -1) * mittag_leffler(z, alpha, 1.0)
    c = np.real(np.fft.ifft(c_hat, axis=1))
    cx, cxx = spectral_derivatives(c, k)
    caputo = caputo_l1(c, t, alpha)
    residual = caputo + velocity * cx - diffusivity * cxx
    residual_abs = np.abs(residual[1:, :])
    return {
        "x": x.reshape(1, -1),
        "t": t.reshape(1, -1),
        "Exact": c,
        "C": c,
        "c0": c0.reshape(1, -1),
        "c_hat0": c0_hat.reshape(1, -1),
        "k": k.reshape(1, -1),
        "lambda_k": lam.reshape(1, -1),
        "cx": cx,
        "cxx": cxx,
        "caputo_l1": caputo,
        "residual": residual,
        "alpha": float(alpha),
        "D": float(diffusivity),
        "v": float(velocity),
        "sigma": float(sigma),
        "dx": dx,
        "t_max": float(t_max),
        "residual_max_abs": float(np.nanmax(residual_abs)),
        "residual_mean_abs": float(np.nanmean(residual_abs)),
    }


def write_plots(payload: dict[str, np.ndarray | float], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(payload["x"]).reshape(-1)
    t = np.asarray(payload["t"]).reshape(-1)
    c = np.asarray(payload["Exact"])
    residual = np.asarray(payload["residual"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx in np.linspace(0, t.size - 1, 6, dtype=int):
        ax.plot(x, c[idx], label=f"t={t[idx]:.2g}")
    ax.set_xlabel("x")
    ax.set_ylabel("c")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "periodic_tfade_plume_evolution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(
        np.abs(residual[1:, :]),
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], t[1], t[-1]],
        cmap="magma",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(image, ax=ax, label="|residual|")
    fig.tight_layout()
    fig.savefig(output_dir / "periodic_tfade_residual_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nt", type=int, default=201)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--D", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=-1.0)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "periodic_tfade_fft")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = generate(
        nx=args.nx,
        nt=args.nt,
        t_max=args.t_max,
        alpha=args.alpha,
        diffusivity=args.D,
        velocity=args.v,
        sigma=args.sigma,
    )
    mat_path = output_dir / "periodic_tfade_fft.mat"
    savemat(mat_path, payload)
    write_plots(payload, output_dir)
    summary_path = output_dir / "periodic_tfade_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"alpha_true: {payload['alpha']}",
                f"D_true: {payload['D']}",
                f"v_true: {payload['v']}",
                f"sigma: {payload['sigma']}",
                f"grid: Nx={args.nx}, Nt={args.nt}, t_max={args.t_max}",
                f"periodic endpoint jump c0: {abs(float(payload['c0'][0, 0]) - float(payload['c0'][0, -1])):.12e}",
                f"L1 residual max abs: {payload['residual_max_abs']:.12e}",
                f"L1 residual mean abs: {payload['residual_mean_abs']:.12e}",
                f"mat: {mat_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved mat: {mat_path}")
    print(f"periodic endpoint jump c0: {abs(float(payload['c0'][0, 0]) - float(payload['c0'][0, -1])):.3e}")
    print(f"L1 residual: max={payload['residual_max_abs']:.3e}, mean={payload['residual_mean_abs']:.3e}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
