"""Plot the time-fractional Fisher benchmark: field, surrogate, and derivatives.

Three rows:
  1. heatmaps of the generated field, the neural surrogate, and the absolute error
  2. u(x) profiles at selected times, data versus surrogate
  3. u_xx(x) profiles at the same times, data versus surrogate

Row 3 is the diagnostic that matters for the discovery result: the diffusion
coefficient is recovered from u_xx, so any loss of second-derivative accuracy
shows up directly as an attenuated coefficient.

Usage: python tools/plot_fisher_surrogate_profiles.py [checkpoint_dir] [--noise 0|5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "fisher_tfr"
PROFILE_TIMES = (1.5, 4.0, 7.0)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="?",
                    default=str(ROOT / "data" / "models" / "fisher_tfr_alpha07_tanh" / "fisher-2000-0"))
    ap.add_argument("--noise", default="0")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "data" / "fisher_tfr_alpha07" / "raw_data" / f"fisher_tfr_alpha07_noise{args.noise}.mat"
    m = sio.loadmat(data_path)
    u_obs = m["Exact"].astype(float)
    t, x = m["t"].ravel(), m["x"].ravel()
    clean = sio.loadmat(ROOT / "data" / "fisher_tfr_alpha07" / "raw_data" / "fisher_tfr_alpha07_noise0.mat")
    u_clean = clean["Exact"].astype(float)

    ckpt = Path(args.checkpoint)
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
        u_hat = net(inp).numpy().reshape(u_clean.shape) * scale + shift

    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    beta_true = float(clean["beta"][0, 0]) if "beta" in clean else 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        _sym = np.where(k != 0.0, (1j * k) ** beta_true, 0.0 + 0.0j)

    def uxx_of(u):
        return np.fft.ifft(_sym[None, :] * np.fft.fft(u, axis=1), axis=1).real

    uxx_clean, uxx_hat = uxx_of(u_clean), uxx_of(u_hat)
    err = np.abs(u_hat - u_clean)

    fig = plt.figure(figsize=(12.0, 9.5))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)

    extent = [x.min(), x.max(), t.min(), t.max()]
    for col, (field, title) in enumerate((
        (u_obs, f"data ({args.noise}% noise)" if args.noise != "0" else "data (clean)"),
        (u_hat, "neural surrogate"),
        (err, "absolute error"),
    )):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis" if col < 2 else "magma")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$t$")
        fig.colorbar(im, ax=ax, fraction=0.046)

    idx = [int(np.argmin(np.abs(t - tv))) for tv in PROFILE_TIMES]
    for col, i in enumerate(idx):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(x, u_clean[i], "k-", lw=1.6, label="data")
        ax.plot(x, u_hat[i], "r--", lw=1.4, label="surrogate")
        ax.set_title(f"$u(x)$ at $t={t[i]:.2f}$", fontsize=10)
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$u$")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for col, i in enumerate(idx):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(x, uxx_clean[i], "k-", lw=1.6, label="data")
        ax.plot(x, uxx_hat[i], "r--", lw=1.2, label="surrogate")
        rel = np.linalg.norm(uxx_hat[i] - uxx_clean[i]) / np.linalg.norm(uxx_clean[i])
        ax.set_title(f"$D_x^{{1.8}}u$ at $t={t[i]:.2f}$  (rel. err {rel:.2f})", fontsize=10)
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$D_x^{1.8}u$")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Time-fractional Fisher-KPP benchmark: "
        r"$D_t^{0.7}u = 0.5\,D_x^{1.8}u + u(1-u)$",
        fontsize=12,
    )
    stem = OUT_DIR / f"fisher_surrogate_profiles_noise{args.noise}"
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"wrote {stem}.png and .pdf")

    tm = (t >= 1.0) & (t < 7.5)
    print(f"field rel L2 = {np.linalg.norm(u_hat - u_clean) / np.linalg.norm(u_clean):.3e}")
    print(f"D_x^beta rel L2 (fit window) = "
          f"{np.linalg.norm(uxx_hat[tm] - uxx_clean[tm]) / np.linalg.norm(uxx_clean[tm]):.3e}")


if __name__ == "__main__":
    main()
