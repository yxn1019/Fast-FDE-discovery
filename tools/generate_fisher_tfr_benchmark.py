"""Generate a time-fractional Fisher-KPP benchmark for the CMAME revision.

Reviewer #1 comment 5 asks how the method performs on strongly nonlinear
fractional equations. This benchmark solves the time-fractional Fisher-KPP
equation with a Caputo time derivative on a periodic domain,

    D_t^alpha u = D u_xx + r u (1 - u),      x in [0, L), t in (0, T],

Design note: a decaying smooth perturbation homogenizes under diffusion, so
u_xx falls below the neural-surrogate noise floor and the diffusion
coefficient stops being identifiable. This benchmark instead uses the
characteristic Fisher-KPP behaviour: a localized initial population develops
two counter-propagating fronts whose gradients persist, which keeps u_xx and
the logistic reaction of comparable magnitude for the whole fitting window.

The solver is the L1 scheme in time (semi-implicit: spectral implicit
diffusion, explicit reaction) with FFT in space. The script reports a
time-step refinement error and a term-balance / order-identifiability check on
the exact field, so the benchmark can be validated before any surrogate is
trained.

Output follows the tsfade raw-data conventions (.mat with x, t, Exact/C/c) so
the existing surrogate-training and discovery pipeline can consume it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.special import gamma as sp_gamma

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "fisher_tfr_alpha07" / "raw_data"

ALPHA = 0.7
BETA = 1.8             # true spatial fractional order (2.0 recovers classical diffusion)
DIFFUSIVITY = 0.5
REACTION = 1.0
DOMAIN_LENGTH = 40.0
NX = 256
T_FINAL = 8.0
NT_SAVE = 321          # saved snapshots including t=0 -> dt_save = 0.025
SUBSTEPS = 8           # solver substeps per saved interval
IC_CENTER = 20.0
IC_WIDTH = 3.0
IC_AMPLITUDE = 0.95
NOISE_SEED = 525


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Localized population; periodic by construction (image sum)."""
    u0 = np.zeros_like(x)
    for m in (-1, 0, 1):
        u0 += IC_AMPLITUDE * np.exp(-(((x - IC_CENTER + DOMAIN_LENGTH * m) / IC_WIDTH) ** 2))
    return u0


def ik_power(k: np.ndarray, order: float) -> np.ndarray:
    """Fourier symbol of the one-sided fractional derivative; zero mode set to 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k != 0.0, (1j * k) ** order, 0.0 + 0.0j)


def solve_l1_fisher(nx: int, nt: int, t_final: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L1/FFT semi-implicit solver; returns (x, t, u) with u shape (nt+1, nx)."""
    x = np.arange(nx) * (DOMAIN_LENGTH / nx)
    dt = t_final / nt
    t = np.arange(nt + 1) * dt
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=DOMAIN_LENGTH / nx)
    symbol = ik_power(k, BETA)
    c0 = dt ** (-ALPHA) / sp_gamma(2.0 - ALPHA)
    j = np.arange(nt + 1, dtype=float)
    b = (j + 1.0) ** (1.0 - ALPHA) - j ** (1.0 - ALPHA)

    u = np.empty((nt + 1, nx))
    u[0] = initial_condition(x)
    u_hat = np.empty((nt + 1, nx), dtype=complex)
    u_hat[0] = np.fft.fft(u[0])
    denom = c0 - DIFFUSIVITY * symbol

    for n in range(1, nt + 1):
        if n > 1:
            diffs = u_hat[1:n][::-1] - u_hat[0:n - 1][::-1]
            history = np.tensordot(b[1:n], diffs, axes=(0, 0))
        else:
            history = np.zeros(nx, dtype=complex)
        reaction = REACTION * u[n - 1] * (1.0 - u[n - 1])
        rhs = c0 * (u_hat[n - 1] - history) + np.fft.fft(reaction)
        u_hat[n] = rhs / denom
        u[n] = np.fft.ifft(u_hat[n]).real
    return x, t, u


def l1_caputo(u: np.ndarray, dt: float, alpha: float) -> np.ndarray:
    nt = u.shape[0]
    du = np.diff(u, axis=0)
    c0 = dt ** (-alpha) / sp_gamma(2.0 - alpha)
    j = np.arange(nt, dtype=float)
    b = (j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)
    out = np.zeros_like(u)
    for n in range(1, nt):
        out[n] = c0 * np.sum(b[:n][::-1, None] * du[:n], axis=0)
    return out


def identifiability_report(x: np.ndarray, t: np.ndarray, u: np.ndarray,
                           fit_t: tuple[float, float]) -> dict:
    """Term balance and alpha-valley on the exact field (no surrogate)."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    dbeta = lambda order: np.fft.ifft(ik_power(k, order)[None, :] * np.fft.fft(u, axis=1), axis=1).real
    uxx = dbeta(BETA)
    mask = (t >= fit_t[0]) & (t < fit_t[1])

    diff_term = DIFFUSIVITY * uxx[mask]
    reac_term = REACTION * u[mask] * (1.0 - u[mask])
    balance = float(np.linalg.norm(diff_term) / np.linalg.norm(reac_term))

    beta_valley = {}
    for b in (1.6, 1.7, 1.8, 1.9, 2.0):
        target = l1_caputo(u, dt, ALPHA)[mask].ravel()
        col = dbeta(b)[mask].ravel()
        A = np.column_stack([np.ones_like(target), u[mask].ravel(), (u[mask] ** 2).ravel(), col])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        beta_valley[b] = (float(np.linalg.norm(A @ coef - target) / np.linalg.norm(target)), coef)

    valley = {}
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        target = l1_caputo(u, dt, a)[mask].ravel()
        A = np.column_stack([
            np.ones_like(target), u[mask].ravel(), (u[mask] ** 2).ravel(), uxx[mask].ravel(),
        ])
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        res = float(np.linalg.norm(A @ coef - target) / np.linalg.norm(target))
        valley[a] = (res, coef)
    return {"diffusion_to_reaction_norm_ratio": balance, "alpha_valley": valley,
            "beta_valley": beta_valley}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nt = (NT_SAVE - 1) * SUBSTEPS
    x, t_full, u_full = solve_l1_fisher(NX, nt, T_FINAL)
    _, _, u_coarse = solve_l1_fisher(NX, nt // 2, T_FINAL)
    refine_err = float(
        np.linalg.norm(u_full[::SUBSTEPS] - u_coarse[:: SUBSTEPS // 2])
        / np.linalg.norm(u_full[::SUBSTEPS])
    )

    field = u_full[::SUBSTEPS]
    t = t_full[::SUBSTEPS]
    print(f"field shape: {field.shape}, u range: [{field.min():.4f}, {field.max():.4f}]")
    print(f"time-refinement relative error (dt vs 2dt): {refine_err:.3e}")

    fit_t = (1.0, 4.0)
    rep = identifiability_report(x, t, field, fit_t)
    print(f"||D u_xx|| / ||r u(1-u)|| on t in {fit_t}: {rep['diffusion_to_reaction_norm_ratio']:.3f}")
    print(f"alpha valley (beta fixed at {BETA}); truth coef = "
          f"[0, {REACTION:+.4f}, {-REACTION:+.4f}, {DIFFUSIVITY:+.4f}]:")
    for a, (res, coef) in rep["alpha_valley"].items():
        print(f"   a={a:.2f}  rel_res={res:.3e}  coef=[{coef[0]:+.4f} {coef[1]:+.4f} "
              f"{coef[2]:+.4f} {coef[3]:+.4f}]")
    print(f"beta valley (alpha fixed at {ALPHA}):")
    for b, (res, coef) in rep["beta_valley"].items():
        print(f"   b={b:.2f}  rel_res={res:.3e}  coef=[{coef[0]:+.4f} {coef[1]:+.4f} "
              f"{coef[2]:+.4f} {coef[3]:+.4f}]")

    stems = {"fisher_tfr_alpha07_noise0": 0.0, "fisher_tfr_alpha07_noise5": 5.0, "fisher_tfr_alpha07_noise25": 25.0}
    for stem, level in stems.items():
        observed = field
        if level > 0.0:
            rng = np.random.default_rng(NOISE_SEED)
            observed = field * (1.0 + 0.01 * level * rng.uniform(-1.0, 1.0, field.shape))
        payload = {
            "x": x.reshape(1, -1),
            "t": t.reshape(1, -1),
            "Exact": observed.astype(np.float32),
            "C": observed.astype(np.float32),
            "c": observed.astype(np.float32),
            "alpha": np.array([[ALPHA]]),
            "beta": np.array([[BETA]]),
            "D": np.array([[DIFFUSIVITY]]),
            "r": np.array([[REACTION]]),
        }
        savemat(OUT_DIR / f"{stem}.mat", payload)

    metadata = {
        "equation": f"D_t^{ALPHA} u = {DIFFUSIVITY} D_x^{BETA} u + {REACTION} u (1 - u)",
        "alpha": ALPHA,
        "beta": BETA,
        "diffusivity": DIFFUSIVITY,
        "reaction_rate": REACTION,
        "initial_condition": (
            f"sum_m {IC_AMPLITUDE}*exp(-((x-{IC_CENTER}+{DOMAIN_LENGTH}m)/{IC_WIDTH})^2), "
            "two counter-propagating fronts"
        ),
        "boundary_condition": f"periodic on [0, {DOMAIN_LENGTH})",
        "solver": "L1 Caputo (semi-implicit) + FFT spectral diffusion",
        "solver_substeps_per_save": SUBSTEPS,
        "time_refinement_relative_error": refine_err,
        "diffusion_to_reaction_norm_ratio": rep["diffusion_to_reaction_norm_ratio"],
        "shape": [int(field.shape[0]), int(field.shape[1])],
        "x_step": DOMAIN_LENGTH / NX,
        "t_step": float(t[1] - t[0]),
        "t_final": T_FINAL,
        "noise_convention": "noiseN: Exact = clean * (1 + 0.01*N*U(-1,1)), seed 525",
    }
    (OUT_DIR / "raw_data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote benchmark files to {OUT_DIR}")


if __name__ == "__main__":
    main()
