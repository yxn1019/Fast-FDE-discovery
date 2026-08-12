"""Manuscript figure for the space-time-fractional Fisher-KPP benchmark.

CMAME revision, Reviewer #1 comment 5.

Claim defended by the figure: the equation discovered from the clean case on the
interior window reproduces the benchmark trajectory, including the interval
beyond the fitting window that never entered the regression.

Panels
  a  generated field
  b  field obtained by solving the discovered equation from the same initial
     state, on the same colour scale

A horizontal dashed line marks the start of the out-of-sample interval.

Panel b is produced with the L1/FFT scheme used to generate the benchmark, with
the discovered orders and coefficients substituted for the true ones, so the
comparison isolates the discovered model rather than the solver.

Outputs SVG, PDF, TIFF and PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.special import gamma as sp_gamma

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fisher_tfr_alpha07" / "raw_data" / "fisher_tfr_alpha07_noise0.mat"
OUT = ROOT / "results" / "fisher_tfr" / "fisher_benchmark"

FIT_T = (1.0, 4.0)
SUBSTEPS = 8

# Clean-case discovered model:
#   D_t^0.72362446 u = 0.3581 D_x^1.762884 u + 0.9233 u - 0.9111 u^2
DISC = dict(alpha=0.72362446, beta=1.7628842, D=0.35814, lin=0.92331, quad=-0.91107)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


def solve_discovered(x: np.ndarray, t_save: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """L1 Caputo in time, FFT in space, with the discovered orders/coefficients."""
    alpha, beta = DISC["alpha"], DISC["beta"]
    nt = (t_save.size - 1) * SUBSTEPS
    dt = (t_save[-1] - t_save[0]) / nt
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        symbol = np.where(k != 0.0, (1j * k) ** beta, 0.0 + 0.0j)

    c0 = dt ** (-alpha) / sp_gamma(2.0 - alpha)
    j = np.arange(nt + 1, dtype=float)
    w = (j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)
    denom = c0 - DISC["D"] * symbol

    u = np.empty((nt + 1, x.size))
    u_hat = np.empty((nt + 1, x.size), dtype=complex)
    u[0] = u0
    u_hat[0] = np.fft.fft(u0)
    for n in range(1, nt + 1):
        if n > 1:
            diffs = u_hat[1:n][::-1] - u_hat[0:n - 1][::-1]
            history = np.tensordot(w[1:n], diffs, axes=(0, 0))
        else:
            history = np.zeros(x.size, dtype=complex)
        reaction = DISC["lin"] * u[n - 1] + DISC["quad"] * u[n - 1] ** 2
        u_hat[n] = (c0 * (u_hat[n - 1] - history) + np.fft.fft(reaction)) / denom
        u[n] = np.fft.ifft(u_hat[n]).real
    return u[::SUBSTEPS]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m = sio.loadmat(DATA)
    u = m["Exact"].astype(float)
    t, x = m["t"].ravel(), m["x"].ravel()

    u_pred = solve_discovered(x, t, u[0])

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 62 / 25.4),
                             sharey=True, constrained_layout=True)
    extent = [x.min(), x.max(), t.min(), t.max()]
    vmin, vmax = float(u.min()), float(u.max())

    panels = (
        (axes[0], u, "generated field"),
        (axes[1], u_pred, "discovered equation"),
    )
    for ax, field, title in panels:
        im = ax.imshow(field, origin="lower", aspect="auto", cmap="viridis",
                       extent=extent, vmin=vmin, vmax=vmax)
        ax.axhline(FIT_T[1], color="white", lw=1.2, ls="--", alpha=0.95)
        ax.set_xlabel("$x$")
        ax.set_title(title, pad=3)
    axes[0].set_ylabel("$t$")

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.015)
    cb.set_label("$u$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)

    for ax, tag in zip(axes, "ab"):
        ax.text(-0.075, 1.10, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")

    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})

    def rel(mask):
        return np.linalg.norm(u_pred[mask] - u[mask]) / np.linalg.norm(u[mask])

    inside = (t >= FIT_T[0]) & (t < FIT_T[1])
    beyond = t >= FIT_T[1]
    print(f"forecast rel L2, whole record   = {rel(np.ones_like(t, bool)):.3e}")
    print(f"forecast rel L2, fitting window = {rel(inside):.3e}")
    print(f"forecast rel L2, t >= {FIT_T[1]}      = {rel(beyond):.3e}")
    print(f"wrote {OUT}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
