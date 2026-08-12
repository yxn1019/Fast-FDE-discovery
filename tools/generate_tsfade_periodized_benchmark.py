"""Generate a periodic-initial-condition twin of the space-time-fractional benchmark.

CMAME revision, Reviewer #1 comment 7. The submitted benchmark samples the
Gaussian 10*exp(-((x-6)/10)^2) on [0,30) and evolves it as one period of a
30-periodic field, so the realized initial state carries a periodic endpoint
mismatch of 6.94. This script builds the twin in which that mismatch is removed
exactly, by periodizing the *same* Gaussian through image summation,

    c0(x) = sum_m 10 * exp(-((x - 6 + 30 m) / 10)^2),

and regenerating the field by the same per-mode Mittag-Leffler evolution,

    c_hat_k(t) = c_hat_k(0) * E_alpha(lambda_k t^alpha),
    lambda_k   = -(i k) + D * (i k)^beta .

Everything else (equation, orders, coefficients, grid, noise convention) is
identical to the submitted benchmark, so discovery on the two datasets isolates
the effect of the endpoint mismatch.

Output goes to a separate directory; the submitted raw data is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transporteq_discovery.mittag_leffler import mittag_leffler

DEFAULT_OUT = ROOT / "data" / "tsfade_periodic_ic" / "raw_data"

ALPHA = 0.78
BETA = 1.83
VELOCITY = 1.0
DIFFUSIVITY = 0.5
X_STEP = 0.25
T_STEP = 0.1
NX = 120
NT = 150
DOMAIN_LENGTH = 30.0
IC_CENTER = 6.0
IC_AMPLITUDE = 10.0
IMAGE_TERMS = 12
NOISE_SEED = 525


def periodized_initial_condition(x: np.ndarray, width: float) -> np.ndarray:
    c0 = np.zeros_like(x)
    for m in range(-IMAGE_TERMS, IMAGE_TERMS + 1):
        c0 += IC_AMPLITUDE * np.exp(-(((x - IC_CENTER + DOMAIN_LENGTH * m) / width) ** 2))
    return c0


def plain_initial_condition(x: np.ndarray, width: float) -> np.ndarray:
    return IC_AMPLITUDE * np.exp(-(((x - IC_CENTER) / width) ** 2))


def ik_power(k: np.ndarray, order: float) -> np.ndarray:
    ik = 1j * k
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k != 0.0, ik**order, 0.0 + 0.0j)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=10.0,
                        help="Gaussian width; 10.0 reproduces the submitted initial profile")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default="tsfade_periodic_ic")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(NX) * X_STEP
    t = np.arange(NT) * T_STEP
    c0 = periodized_initial_condition(x, args.width)

    k = 2.0 * np.pi * np.fft.fftfreq(NX, d=X_STEP)
    c0_hat = np.fft.fft(c0)
    lambda_k = -VELOCITY * ik_power(k, 1.0) + DIFFUSIVITY * ik_power(k, BETA)
    ml = mittag_leffler((lambda_k[None, :] * (t[:, None] ** ALPHA)).reshape(-1), ALPHA, 1.0)
    c_hat = c0_hat[None, :] * ml.reshape(NT, NX)

    field_complex = np.fft.ifft(c_hat, axis=1)
    field = field_complex.real
    max_imag = float(np.max(np.abs(field_complex.imag)))

    cx = np.fft.ifft(ik_power(k, 1.0)[None, :] * c_hat, axis=1).real
    d_beta_c = np.fft.ifft(ik_power(k, BETA)[None, :] * c_hat, axis=1).real
    dt_alpha_c = np.fft.ifft(lambda_k[None, :] * c_hat, axis=1).real
    residual = dt_alpha_c - (-VELOCITY * cx + DIFFUSIVITY * d_beta_c)
    residual_rel = float(np.linalg.norm(residual) / np.linalg.norm(dt_alpha_c))

    gap_periodized = float(abs(periodized_initial_condition(np.array([0.0]), args.width)[0]
                               - periodized_initial_condition(np.array([DOMAIN_LENGTH]), args.width)[0]))
    gap_plain = float(abs(plain_initial_condition(np.array([0.0]), args.width)[0]
                          - plain_initial_condition(np.array([DOMAIN_LENGTH - X_STEP]), args.width)[0]))

    print(f"Gaussian width                     : {args.width}")
    print(f"endpoint mismatch, submitted IC    : {gap_plain:.4f}")
    print(f"endpoint mismatch, periodized IC   : {gap_periodized:.3e}")
    print(f"max imaginary leakage              : {max_imag:.3e}")
    print(f"spectral residual (relative L2)    : {residual_rel:.3e}")
    print(f"field range                        : [{field.min():.4f}, {field.max():.4f}]")

    payload = {
        "x": x.reshape(1, -1), "t": t.reshape(1, -1),
        "c0": c0.reshape(1, -1), "cx": cx, "D_beta_c": d_beta_c,
        "Dt_alpha_c": dt_alpha_c, "residual": residual,
        "k": k.reshape(1, -1), "lambda_k": lambda_k.reshape(1, -1),
        "alpha": np.array([[ALPHA]]), "beta": np.array([[BETA]]),
        "v": np.array([[VELOCITY]]), "D": np.array([[DIFFUSIVITY]]),
    }
    # Uniform relative noise, matching the submitted benchmark convention.
    for level in (0.0, 5.0, 25.0):
        observed = field
        if level > 0.0:
            rng = np.random.default_rng(NOISE_SEED)
            observed = field * (1.0 + 0.01 * level * rng.uniform(-1.0, 1.0, field.shape))
        item = dict(payload)
        for key in ("Exact", "C", "c"):
            item[key] = observed.astype(np.float32)
        savemat(out_dir / f"{args.stem}_noise{int(level)}.mat", item)

    metadata = {
        "equation": "D_t^0.78 c = -c_x + 0.5 D_x^1.83 c",
        "alpha": ALPHA, "beta": BETA, "velocity": VELOCITY, "diffusivity": DIFFUSIVITY,
        "initial_condition": (
            f"sum_(m=-12..12) {IC_AMPLITUDE}*exp(-((x-{IC_CENTER}+{DOMAIN_LENGTH}m)/{args.width})^2)"
        ),
        "gaussian_width": args.width,
        "boundary_condition": "periodic c(x+30,t)=c(x,t), implemented by FFT on [0,30)",
        "x_step": X_STEP, "t_step": T_STEP, "shape": [NT, NX],
        "endpoint_mismatch_submitted_ic": gap_plain,
        "endpoint_mismatch_periodized_ic": gap_periodized,
        "max_imaginary_leakage": max_imag,
        "residual_relative_l2": residual_rel,
        "noise_convention": "noiseN: Exact = clean * (1 + 0.01*N*U(-1,1)), seed 525",
        "purpose": "periodic-IC twin of the submitted benchmark for the R1-7 robustness check",
    }
    (out_dir / "raw_data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote periodic-IC benchmark to {out_dir}")


if __name__ == "__main__":
    main()
