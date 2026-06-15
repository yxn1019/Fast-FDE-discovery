"""Plot raw synthetic field versus NN surrogate fidelity on the same grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_case_params import PAPER_FIT_X_MAX, PAPER_FIT_X_MIN
from transporteq_discovery.fractional_discoverer import FractionalPDEDiscoverer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT
        / "data"
        / "tsfade_retrained_alpha078_beta183"
        / "raw_data"
        / "tsfade_alpha078_beta183_noise5.mat",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "data"
        / "models"
        / "tsfade_retrained_alpha078_beta183_tanh"
        / "draft-2000-5"
        / "best.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "tsfade_retrained_alpha078_beta183" / "fidelity_plots",
    )
    parser.add_argument("--early-t-max", type=float, default=1.0)
    parser.add_argument("--case", choices=("tsfade_fft", "periodic_tfade_fft"), default="tsfade_fft")
    return parser.parse_args()


def load_raw_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = loadmat(path)
    x = np.asarray(payload["x"], dtype=float).reshape(-1)
    t = np.asarray(payload["t"], dtype=float).reshape(-1)
    key = "c" if "c" in payload else "Exact"
    c = np.asarray(payload[key], dtype=float)
    if c.shape == (x.size, t.size):
        c = c.T
    return x, t, c


def checkpoint_config(checkpoint: Path) -> dict:
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get("config", payload) if isinstance(payload, dict) else {}


def noisy_observation(raw: np.ndarray, config: dict) -> np.ndarray:
    noise_type = str(config.get("noise_type", "none")).lower()
    noise_level = float(config.get("noise_level", 0.0) or 0.0)
    if noise_level == 0.0 or noise_type == "none":
        return raw.copy()
    rng = np.random.default_rng(int(config.get("seed", 525)))
    scale = 0.01 * noise_level
    if noise_type == "uniform_relative":
        return raw * (1.0 + scale * rng.uniform(-1.0, 1.0, raw.shape))
    if noise_type == "gaussian_relative":
        return raw * (1.0 + scale * rng.normal(0.0, 1.0, raw.shape))
    if noise_type == "gaussian_additive":
        return raw + scale * float(np.std(raw)) * rng.normal(0.0, 1.0, raw.shape)
    return raw.copy()


def reconstruct_nn_on_raw_grid(checkpoint: Path, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    discoverer = FractionalPDEDiscoverer()
    torch = discoverer._torch()
    metadata = discoverer._load_training_metadata(checkpoint)
    network_spec = discoverer._network_spec(metadata)
    net = discoverer._build_network(
        torch,
        activation=str(network_spec["activation"]),
        hidden_layers=int(network_spec["hidden_layers"]),
        neurons=int(network_spec["neurons"]),
    )
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else metadata
    input_normalization = str(config.get("input_normalization", "none")) if isinstance(config, dict) else "none"
    output_normalization = str(config.get("output_normalization", "none")) if isinstance(config, dict) else "none"
    x_min = float(config.get("input_x_range", [x[0], x[-1]])[0]) if isinstance(config, dict) else float(x[0])
    x_max = float(config.get("input_x_range", [x[0], x[-1]])[1]) if isinstance(config, dict) else float(x[-1])
    t_min = float(config.get("input_t_range", [t[0], t[-1]])[0]) if isinstance(config, dict) else float(t[0])
    t_max = float(config.get("input_t_range", [t[0], t[-1]])[1]) if isinstance(config, dict) else float(t[-1])
    x_scale = float(config.get("input_x_scale", max(x_max - x_min, np.finfo(float).eps))) if isinstance(config, dict) else float(max(x_max - x_min, np.finfo(float).eps))
    t_scale = float(config.get("input_t_scale", max(t_max - t_min, np.finfo(float).eps))) if isinstance(config, dict) else float(max(t_max - t_min, np.finfo(float).eps))

    x_torch = torch.tensor(x, dtype=torch.float32)
    t_torch = torch.tensor(t, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_torch, x_torch, indexing="ij")
    if input_normalization == "unit_box":
        xx_net = (xx - x_min) / max(x_scale, np.finfo(float).eps)
        tt_net = (tt - t_min) / max(t_scale, np.finfo(float).eps)
    else:
        xx_net = xx
        tt_net = tt
    database = torch.stack((xx_net.reshape(-1), tt_net.reshape(-1)), dim=1)
    with torch.no_grad():
        H = discoverer._apply_output_activation(torch, net(database), metadata)
    H_np = H.detach().cpu().numpy().reshape(t.size, x.size).astype(float)
    if output_normalization == "unit_interval":
        target_c_shift = float(config.get("target_c_shift", 0.0)) if isinstance(config, dict) else 0.0
        target_c_scale = float(config.get("target_c_scale", 1.0)) if isinstance(config, dict) else 1.0
        H_np = H_np * max(target_c_scale, np.finfo(float).eps) + target_c_shift
    return H_np


def metrics(raw: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        raw_sel = raw
        pred_sel = pred
    else:
        raw_sel = raw[mask]
        pred_sel = pred[mask]
    err = pred_sel - raw_sel
    raw_norm = max(float(np.linalg.norm(raw_sel)), np.finfo(float).eps)
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "relative_l2": float(np.linalg.norm(err) / raw_norm),
        "bias": float(np.mean(err)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
    }


def save_heatmap(
    values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "value",
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(t[0]), float(t[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_histogram(error: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(error.reshape(-1), bins=80, color="#4472c4", alpha=0.9, edgecolor="white")
    ax.set_xlabel("prediction error (NN - raw)")
    ax.set_ylabel("count")
    ax.set_title("NN Surrogate Error Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_curve(x: np.ndarray, y: np.ndarray, path: Path, *, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(x, y, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_profile_comparison(x: np.ndarray, raw: np.ndarray, nn: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(x, raw[0, :], label="raw t=0", linewidth=2.0)
    ax.plot(x, nn[0, :], label="NN t=0", linewidth=1.8, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("c")
    ax.set_title("Initial Profile Comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_spectrum_comparison(raw: np.ndarray, nn: np.ndarray, path: Path) -> None:
    raw_hat = np.abs(np.fft.fft(raw[0, :])) / raw.shape[1]
    nn_hat = np.abs(np.fft.fft(nn[0, :])) / nn.shape[1]
    mode = np.arange(raw_hat.size // 2 + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.semilogy(mode, raw_hat[: mode.size], label="raw t=0", linewidth=2.0)
    ax.semilogy(mode, nn_hat[: mode.size], label="NN t=0", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Fourier mode")
    ax.set_ylabel("|c_hat_k|")
    ax.set_title("Initial Spectrum Comparison")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def derivative_x(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    return np.fft.ifft((1j * k).reshape(1, -1) * np.fft.fft(values, axis=1), axis=1).real


def recommended_windows(x: np.ndarray, t: np.ndarray, abs_err: np.ndarray, case: str) -> dict:
    x_rmse = np.sqrt(np.mean(abs_err**2, axis=0))
    t_rmse = np.sqrt(np.mean(abs_err**2, axis=1))
    x_threshold = float(np.percentile(x_rmse, 90))
    good_x = x_rmse <= x_threshold
    if case == "tsfade_fft":
        default_x_min, default_x_max = float(PAPER_FIT_X_MIN), float(PAPER_FIT_X_MAX)
    else:
        default_x_min, default_x_max = float(x[0]), float(x[-1] + (x[1] - x[0]))
    default_mask = (x >= default_x_min) & (x < default_x_max)
    combined = good_x & default_mask
    if np.count_nonzero(combined) >= 4:
        idx = np.flatnonzero(combined)
        x_min = float(x[idx[0]])
        x_max = float(x[idx[-1]] + (x[1] - x[0]))
    else:
        x_min, x_max = default_x_min, default_x_max
    t_candidates = [0.0, 0.5, 1.0, 2.0]
    t_scores = []
    for value in t_candidates:
        mask = t >= value
        t_scores.append(float(np.mean(t_rmse[mask])) if np.any(mask) else float("inf"))
    t_min = float(t_candidates[int(np.argmin(t_scores))])
    return {
        "default": {"fit_x_min": default_x_min, "fit_x_max": default_x_max, "fit_t_min": None, "fit_t_max": None},
        "error_trimmed": {"fit_x_min": x_min, "fit_x_max": x_max, "fit_t_min": t_min, "fit_t_max": None},
        "reason": {
            "x_rule": "keep x columns below the 90th percentile of x-wise RMSE within the default window",
            "t_rule": "choose t_min from 0,0.5,1.0,2.0 with lowest mean t-wise RMSE",
            "x_rmse_p90": x_threshold,
            "t_min_scores": dict(zip((str(v) for v in t_candidates), t_scores)),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, t, raw = load_raw_field(args.data)
    config = checkpoint_config(args.checkpoint)
    noisy = noisy_observation(raw, config)
    nn = reconstruct_nn_on_raw_grid(args.checkpoint, x, t)
    err = nn - raw
    abs_err = np.abs(err)
    noisy_err = noisy - raw
    raw_x = derivative_x(raw, x)
    nn_x = derivative_x(nn, x)
    dx_abs_err = np.abs(nn_x - raw_x)

    fit_mask_x = (x >= float(PAPER_FIT_X_MIN)) & (x < float(PAPER_FIT_X_MAX))
    fit_mask = np.zeros_like(raw, dtype=bool)
    fit_mask[:, fit_mask_x] = True

    full_metrics = metrics(raw, nn)
    fit_metrics = metrics(raw, nn, fit_mask)
    noisy_metrics = metrics(raw, noisy)
    derivative_metrics = metrics(raw_x, nn_x)

    field_vmin = float(min(np.min(raw), np.min(nn)))
    field_vmax = float(max(np.max(raw), np.max(nn)))
    err_lim = float(np.max(np.abs(err)))

    save_heatmap(
        raw,
        x,
        t,
        args.output_dir / "raw_field_heatmap.png",
        title="Raw Field c(x,t)",
        cmap="viridis",
        vmin=field_vmin,
        vmax=field_vmax,
        colorbar_label="c",
    )
    save_heatmap(
        noisy,
        x,
        t,
        args.output_dir / "noisy_observation_heatmap.png",
        title="Noisy Observation",
        cmap="viridis",
        vmin=field_vmin,
        vmax=field_vmax,
        colorbar_label="c",
    )
    save_heatmap(
        nn,
        x,
        t,
        args.output_dir / "nn_surrogate_heatmap.png",
        title="NN Surrogate c_hat(x,t)",
        cmap="viridis",
        vmin=field_vmin,
        vmax=field_vmax,
        colorbar_label="c",
    )
    save_heatmap(
        err,
        x,
        t,
        args.output_dir / "signed_error_heatmap.png",
        title="Signed Error (NN - Raw)",
        cmap="coolwarm",
        vmin=-err_lim,
        vmax=err_lim,
        colorbar_label="error",
    )
    save_heatmap(
        abs_err,
        x,
        t,
        args.output_dir / "absolute_error_heatmap.png",
        title="Absolute Error |NN - Raw|",
        cmap="magma",
        colorbar_label="absolute error",
    )
    save_heatmap(
        noisy_err,
        x,
        t,
        args.output_dir / "noisy_signed_error_heatmap.png",
        title="Noisy Observation Error (Noisy - Raw)",
        cmap="coolwarm",
        vmin=-float(np.max(np.abs(noisy_err))),
        vmax=float(np.max(np.abs(noisy_err))),
        colorbar_label="error",
    )
    save_heatmap(
        dx_abs_err,
        x,
        t,
        args.output_dir / "cx_absolute_error_heatmap.png",
        title="Spectral c_x Absolute Error",
        cmap="magma",
        colorbar_label="absolute c_x error",
    )
    save_histogram(err, args.output_dir / "error_histogram.png")
    save_profile_comparison(x, raw, nn, args.output_dir / "initial_profile_comparison.png")
    save_spectrum_comparison(raw, nn, args.output_dir / "initial_spectrum_comparison.png")
    save_curve(t, np.sqrt(np.mean(abs_err**2, axis=1)), args.output_dir / "timewise_rmse.png", xlabel="t", ylabel="RMSE", title="Time-wise NN RMSE")
    save_curve(x, np.sqrt(np.mean(abs_err**2, axis=0)), args.output_dir / "xwise_rmse.png", xlabel="x", ylabel="RMSE", title="X-wise NN RMSE")

    early_mask = t <= float(args.early_t_max)
    if np.count_nonzero(early_mask) > 1:
        save_heatmap(
            abs_err[early_mask, :],
            x,
            t[early_mask],
            args.output_dir / "early_absolute_error_heatmap.png",
            title=f"Early Absolute Error |NN - Raw|, t <= {args.early_t_max:g}",
            cmap="magma",
            colorbar_label="absolute error",
        )

    summary = {
        "data_file": str(args.data),
        "checkpoint": str(args.checkpoint),
        "grid_shape": list(raw.shape),
        "fit_window_x": [float(PAPER_FIT_X_MIN), float(PAPER_FIT_X_MAX)],
        "full_domain_metrics": full_metrics,
        "fit_window_metrics": fit_metrics,
        "noisy_observation_metrics": noisy_metrics,
        "spectral_cx_metrics": derivative_metrics,
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "nn_min": float(np.min(nn)),
        "nn_max": float(np.max(nn)),
        "initial_profile_rmse": float(np.sqrt(np.mean((nn[0, :] - raw[0, :]) ** 2))),
        "initial_profile_max_abs_error": float(np.max(np.abs(nn[0, :] - raw[0, :]))),
    }
    windows = recommended_windows(x, t, abs_err, args.case)
    (args.output_dir / "fidelity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "recommended_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")

    print(json.dumps({"summary": summary, "recommended_windows": windows}, indent=2))


if __name__ == "__main__":
    main()
