"""Evaluate Fisher surrogate quality: field error, u_xx error, and alpha valley.

Usage: python tools/probe_fisher_surrogate_quality.py <checkpoint_dir> [...]
Each checkpoint_dir must contain best.pkl and config.json.

The alpha valley regresses the L1-Caputo time derivative of the surrogate field
on [1, u, u^2, u_xx] over the fitting window, for a range of trial orders. A
sharp minimum at the true alpha with coefficients close to (0, r, -r, D) means
the surrogate supports order identification; a flat valley or a strongly
attenuated u_xx coefficient means it does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.special import gamma as sp_gamma

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fisher_tfr_alpha07" / "raw_data" / "fisher_tfr_alpha07_noise0.mat"
FIT_T = (0.2, 0.95)
TRIAL_ALPHAS = (0.60, 0.65, 0.70, 0.75, 0.80)


class Sin(nn.Module):
    def forward(self, z):
        return torch.sin(z)


def build_net(cfg: dict) -> nn.Module:
    act = {"tanh": nn.Tanh, "sin": Sin}[cfg["activation"]]
    mods = [nn.Linear(2, cfg["neurons"]), act()]
    for _ in range(cfg["hidden_layers"] - 1):
        mods += [nn.Linear(cfg["neurons"], cfg["neurons"]), act()]
    mods += [nn.Linear(cfg["neurons"], 1)]
    return nn.Sequential(*mods)


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


def main() -> None:
    m = sio.loadmat(DATA)
    u_true = m["Exact"].astype(float)
    t, x = m["t"].ravel(), m["x"].ravel()
    truth = (0.0, float(m["r"][0, 0]), -float(m["r"][0, 0]), float(m["D"][0, 0]))
    dt = t[1] - t[0]
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])

    beta_true = float(m["beta"][0, 0]) if "beta" in m else 2.0

    def uxx_of(u, order=None):
        order = beta_true if order is None else order
        with np.errstate(divide="ignore", invalid="ignore"):
            sym = np.where(k != 0.0, (1j * k) ** order, 0.0 + 0.0j)
        return np.fft.ifft(sym[None, :] * np.fft.fft(u, axis=1), axis=1).real

    tm = (t >= FIT_T[0]) & (t < FIT_T[1])
    uxx_true = uxx_of(u_true)
    print(f"truth coefficients: [{truth[0]:+.4f} {truth[1]:+.4f} {truth[2]:+.4f} {truth[3]:+.4f}]  (regressor = D_x^{beta_true} u)")

    for ckpt_dir in sys.argv[1:]:
        ckpt = Path(ckpt_dir)
        cfg = json.load(open(ckpt / "config.json"))
        net = build_net(cfg)
        net.load_state_dict(torch.load(ckpt / "best.pkl", map_location="cpu", weights_only=True))
        net.eval()
        xs, ts = cfg["input_x_range"], cfg["input_t_range"]
        shift, scale = cfg["target_c_shift"], cfg["target_c_scale"]
        tt, xx = np.meshgrid(t, x, indexing="ij")
        inp = torch.tensor(
            np.column_stack([
                ((xx - xs[0]) / (xs[1] - xs[0])).ravel(),
                ((tt - ts[0]) / (ts[1] - ts[0])).ravel(),
            ]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            u_hat = net(inp).numpy().reshape(u_true.shape) * scale + shift
        uxx_hat = uxx_of(u_hat)
        field_err = np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true)
        uxx_err = np.linalg.norm(uxx_hat[tm] - uxx_true[tm]) / np.linalg.norm(uxx_true[tm])
        print(f"\n== {ckpt.name} ==  field rel L2={field_err:.3e}   D_x^beta rel L2={uxx_err:.3e}")
        best = None
        for alpha in TRIAL_ALPHAS:
            y = l1_caputo(u_hat, dt, alpha)[tm].ravel()
            A = np.column_stack([
                np.ones_like(y), u_hat[tm].ravel(), (u_hat[tm] ** 2).ravel(), uxx_hat[tm].ravel(),
            ])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            res = float(np.linalg.norm(A @ coef - y) / np.linalg.norm(y))
            flag = ""
            if best is None or res < best[1]:
                best = (alpha, res)
            print(f"   a={alpha:.2f} rel_res={res:.3e} "
                  f"coef=[{coef[0]:+.4f} {coef[1]:+.4f} {coef[2]:+.4f} {coef[3]:+.4f}]{flag}")
        print(f"   -> valley minimum at alpha={best[0]:.2f}")


if __name__ == "__main__":
    main()
