"""Train a reusable NN reconstruction model for the local time-FADE data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


MAX_TRAIN_POINTS = 2000


@dataclass(frozen=True)
class TrainingConfig:
    case: str = "tsfade_fft"
    data_file: str = str(
        ROOT
        / "data"
        / "tsfade_retrained_alpha078_beta183"
        / "raw_data"
        / "tsfade_alpha078_beta183_noise5.mat"
    )
    output_dir: str = str(
        ROOT
        / "data"
        / "models"
        / "tsfade_retrained_alpha078_beta183_tanh"
        / "draft-2000-5"
    )
    train_points: int = 2000
    val_points: int = 2000
    max_steps: int = 15000
    noise_level: float = 0.0
    noise_type: str = "gaussian_relative"
    activation: str = "tanh"
    hidden_layers: int = 8
    neurons: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    seed: int = 525
    eval_every: int = 100
    print_every: int = 1000
    patience_steps: int = 0
    min_delta_rel: float = 0.0
    periodic_boundary_weight: float = 0.0
    periodic_derivative_weight: float = 0.0
    periodic_boundary_points: int = 0
    early_time_weight: float = 0.0
    early_time_cutoff: float = 1.0
    spectral_weight: float = 0.0
    spectral_high_k_gamma: float = 0.0
    spectral_high_k_power: float = 2.0
    spectral_time_batch: int = 8
    integer_derivative_weight: float = 0.0
    peak_weight: float = 0.0
    peak_quantile: float = 0.90
    peak_time_cutoff: float = 1.0
    peak_use_absolute: bool = True
    gaussian_scale: float = 0.5
    denoise_mode: str = "none"
    denoise_fft_cutoff: float = 0.0
    allow_large_train_set: bool = False
    input_normalization: str = "none"
    output_normalization: str = "none"
    output_activation: str = "auto"
    data_loss: str = "mse"
    log_loss_epsilon: float = 0.0
    physical_loss_weight: float = 1.0
    log_loss_weight: float = 1.0
    selection_metric: str = "val"
    selection_derivative_weight: float = 1.0
    selection_spectral_weight: float = 1.0


def build_network(torch, activation: str, hidden_layers: int, neurons: int):
    from torch import nn

    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    class Gaussian(nn.Module):
        def __init__(self, scale: float = 0.5):
            super().__init__()
            self.scale = float(scale)

        def forward(self, x):
            return torch.exp(-0.5 * (self.scale * x) ** 2)

    activation_key = activation.lower()
    if activation_key == "tanh":
        activation_factory = nn.Tanh
    elif activation_key == "sin":
        activation_factory = Sin
    elif activation_key.startswith("gaussian"):
        if ":" in activation_key:
            scale = float(activation_key.split(":", 1)[1])
        else:
            scale = 0.5
        activation_factory = lambda: Gaussian(scale)
    else:
        raise NotImplementedError("Supported activations are 'sin', 'tanh', and 'gaussian'.")

    layers: list[nn.Module] = [nn.Linear(2, neurons), activation_factory()]
    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(neurons, neurons), activation_factory()])
    layers.append(nn.Linear(neurons, 1))
    return nn.Sequential(*layers)


def load_training_arrays(config: TrainingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(config.data_file)
    field_key = "Exact" if "Exact" in data else "c"
    values = np.asarray(data[field_key], dtype=np.float32)
    time = np.asarray(data["t"], dtype=np.float32).reshape(-1)
    position = np.asarray(data["x"], dtype=np.float32).reshape(-1)
    if values.shape == (position.size, time.size):
        values = values.T
    if values.shape != (time.size, position.size):
        raise ValueError(f"Expected Exact shape {(time.size, position.size)}, got {values.shape}")
    return time, position, values


def make_dataset(config: TrainingConfig, torch):
    time, position, values = load_training_arrays(config)
    observed_values = values.copy()
    if config.input_normalization not in {"none", "unit_box"}:
        raise ValueError("--input-normalization must be one of none, unit_box")
    if config.output_normalization not in {"none", "unit_interval"}:
        raise ValueError("--output-normalization must be one of none, unit_interval")
    if config.output_activation not in {"auto", "identity", "softplus"}:
        raise ValueError("--output-activation must be one of auto, identity, softplus")
    if config.data_loss not in {"mse", "log_mse", "hybrid_log_mse"}:
        raise ValueError("--data-loss must be one of mse, log_mse, hybrid_log_mse")
    x_min = float(position[0])
    x_max = float(position[-1])
    t_min = float(time[0])
    t_max = float(time[-1])
    x_scale = max(x_max - x_min, np.finfo(np.float32).eps)
    t_scale = max(t_max - t_min, np.finfo(np.float32).eps)

    def normalize_coords(x_vals: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
        if config.input_normalization == "unit_box":
            x_out = (x_vals - x_min) / x_scale
            t_out = (t_vals - t_min) / t_scale
        else:
            x_out = x_vals
            t_out = t_vals
        return np.column_stack((x_out.reshape(-1), t_out.reshape(-1))).astype(np.float32)

    tt, xx = np.meshgrid(time, position, indexing="ij")
    coordinates = normalize_coords(xx, tt)
    noise_type = config.noise_type.lower()
    if noise_type not in {"none", "uniform_relative", "gaussian_relative", "gaussian_additive"}:
        raise ValueError(
            "--noise-type must be one of none, uniform_relative, gaussian_relative, gaussian_additive"
        )
    if config.noise_level != 0.0 and noise_type != "none":
        rng = np.random.default_rng(config.seed)
        scale = 0.01 * float(config.noise_level)
        if noise_type == "uniform_relative":
            observed_values = observed_values * (1.0 + scale * rng.uniform(-1.0, 1.0, observed_values.shape))
        elif noise_type == "gaussian_relative":
            observed_values = observed_values * (1.0 + scale * rng.normal(0.0, 1.0, observed_values.shape))
        elif noise_type == "gaussian_additive":
            amplitude = float(np.std(observed_values))
            observed_values = observed_values + scale * amplitude * rng.normal(0.0, 1.0, observed_values.shape)

    target_values = observed_values
    denoise_mode = config.denoise_mode.lower()
    if denoise_mode not in {"none", "hard_lowpass", "soft_spectral"}:
        raise ValueError("--denoise-mode must be one of none, hard_lowpass, soft_spectral")
    if config.denoise_fft_cutoff > 0.0 and denoise_mode == "none":
        denoise_mode = "hard_lowpass"
    if denoise_mode == "hard_lowpass":
        if not 0.0 < config.denoise_fft_cutoff <= 1.0:
            raise ValueError("--denoise-fft-cutoff must be in (0, 1] when enabled")
        spectrum = np.fft.fft(observed_values, axis=1)
        k_abs = np.abs(np.fft.fftfreq(position.size, d=float(position[1] - position[0])))
        keep = k_abs <= config.denoise_fft_cutoff * float(k_abs.max())
        target_values = np.fft.ifft(spectrum * keep.reshape(1, -1), axis=1).real.astype(np.float32)
    elif denoise_mode == "soft_spectral":
        spectrum = np.fft.fft(observed_values, axis=1)
        power = np.abs(spectrum) ** 2
        tail = power[:, max(1, int(0.7 * power.shape[1])) :]
        noise_floor = np.median(tail, axis=1, keepdims=True)
        gain = power / (power + noise_floor + np.finfo(np.float32).eps)
        target_values = np.fft.ifft(spectrum * gain, axis=1).real.astype(np.float32)

    target_c_shift = float(np.min(target_values))
    target_c_max = float(np.max(target_values))
    target_c_scale = float(max(target_c_max - target_c_shift, np.finfo(np.float32).eps))
    if config.output_normalization == "unit_interval":
        target_values_train = ((target_values - target_c_shift) / target_c_scale).astype(np.float32)
    else:
        target_values_train = target_values.astype(np.float32)
    output_activation = config.output_activation
    if output_activation == "auto":
        output_activation = "softplus" if config.data_loss in {"log_mse", "hybrid_log_mse"} else "identity"
    positive_physical = target_values[target_values > 0.0]
    positive_train = target_values_train[target_values_train > 0.0]
    physical_log_loss_epsilon = (
        float(config.log_loss_epsilon)
        if config.log_loss_epsilon > 0.0
        else float(max(1.0e-8, 0.01 * np.percentile(positive_physical, 5))) if positive_physical.size else 1.0e-8
    )
    train_log_loss_epsilon = (
        float(config.log_loss_epsilon)
        if config.log_loss_epsilon > 0.0
        else float(max(1.0e-8, 0.01 * np.percentile(positive_train, 5))) if positive_train.size else 1.0e-8
    )
    targets = target_values_train.reshape(-1, 1).astype(np.float32)

    # Build a peak mask from training targets so peak weighting does not leak PDE information.
    peak_basis = np.abs(target_values) if config.peak_use_absolute else target_values
    peak_time_mask = time <= float(config.peak_time_cutoff)
    if not np.any(peak_time_mask):
        peak_time_mask = np.ones_like(time, dtype=bool)
    early_values = peak_basis[peak_time_mask, :]
    peak_threshold = float(np.quantile(early_values.reshape(-1), float(config.peak_quantile)))
    peak_mask = peak_basis >= peak_threshold

    total = coordinates.shape[0]
    required = config.train_points + config.val_points
    if required > total:
        raise ValueError(f"train_points + val_points = {required} exceeds available samples {total}")
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(total)
    train_idx = indices[: config.train_points]
    val_idx = indices[config.train_points : config.train_points + config.val_points]
    if config.periodic_boundary_points > 0:
        boundary_count = min(config.periodic_boundary_points, time.size)
        boundary_time = np.sort(rng.choice(time, size=boundary_count, replace=False))
    else:
        boundary_time = time
    boundary_left = normalize_coords(
        np.full(boundary_time.shape, position[0], dtype=np.float32), boundary_time
    )
    boundary_right = normalize_coords(
        np.full(boundary_time.shape, position[-1], dtype=np.float32), boundary_time
    )
    return {
        "time": time,
        "position": position,
        "input_normalization": config.input_normalization,
        "input_x_range": [x_min, x_max],
        "input_t_range": [t_min, t_max],
        "input_x_scale": float(x_scale),
        "input_t_scale": float(t_scale),
        "output_normalization": config.output_normalization,
        "output_activation": output_activation,
        "data_loss": config.data_loss,
        "log_loss_epsilon": train_log_loss_epsilon,
        "physical_log_loss_epsilon": physical_log_loss_epsilon,
        "physical_loss_weight": float(config.physical_loss_weight),
        "log_loss_weight": float(config.log_loss_weight),
        "target_c_range": [target_c_shift, target_c_max],
        "target_c_shift": target_c_shift,
        "target_c_scale": target_c_scale,
        "values_shape": values.shape,
        "values_grid": torch.from_numpy(target_values_train.astype(np.float32)),
        "observed_values_grid": torch.from_numpy(observed_values.astype(np.float32)),
        "train_x": torch.from_numpy(coordinates[train_idx]),
        "train_y": torch.from_numpy(targets[train_idx]),
        "val_x": torch.from_numpy(coordinates[val_idx]),
        "val_y": torch.from_numpy(targets[val_idx]),
        "early_train_x": torch.from_numpy(coordinates[train_idx][coordinates[train_idx][:, 1] <= config.early_time_cutoff]),
        "early_train_y": torch.from_numpy(targets[train_idx][coordinates[train_idx][:, 1] <= config.early_time_cutoff]),
        "peak_train_x": torch.from_numpy(coordinates[train_idx][peak_mask.reshape(-1)[train_idx]]),
        "peak_train_y": torch.from_numpy(targets[train_idx][peak_mask.reshape(-1)[train_idx]]),
        "peak_threshold": peak_threshold,
        "peak_mask_fraction": float(np.mean(peak_mask)),
        "boundary_left_x": torch.from_numpy(boundary_left),
        "boundary_right_x": torch.from_numpy(boundary_right),
    }


def apply_output_activation(torch, values, activation: str):
    if activation == "identity":
        return values
    if activation == "softplus":
        return torch.nn.functional.softplus(values)
    raise ValueError("output activation must be identity or softplus")


def predict_values(torch, net, x, dataset):
    raw = net(x)
    return apply_output_activation(torch, raw, str(dataset.get("output_activation", "identity")))


def data_fit_loss(torch, prediction, target, dataset, config: TrainingConfig):
    mode = str(config.data_loss)
    if mode == "mse":
        return torch.mean((prediction - target) ** 2)
    eps = float(dataset.get("log_loss_epsilon", 1.0e-8))
    eps_tensor = prediction.new_tensor(eps)
    pred_pos = torch.clamp(prediction, min=0.0)
    target_pos = torch.clamp(target, min=0.0)
    log_loss = torch.mean((torch.log(pred_pos + eps_tensor) - torch.log(target_pos + eps_tensor)) ** 2)
    if mode == "log_mse":
        return log_loss
    if mode == "hybrid_log_mse":
        denom = torch.mean(target_pos**2).clamp_min(eps_tensor**2)
        physical_loss = torch.mean((prediction - target) ** 2) / denom
        return float(config.physical_loss_weight) * physical_loss + float(config.log_loss_weight) * log_loss
    raise ValueError("--data-loss must be one of mse, log_mse, hybrid_log_mse")


def evaluate(torch, net, x, y, dataset, config: TrainingConfig) -> float:
    with torch.no_grad():
        prediction = predict_values(torch, net, x, dataset)
        return float(data_fit_loss(torch, prediction, y, dataset, config).detach().cpu().item())


def evaluate_periodic_boundary(torch, net, dataset) -> float:
    with torch.no_grad():
        left = predict_values(torch, net, dataset["boundary_left_x"], dataset)
        right = predict_values(torch, net, dataset["boundary_right_x"], dataset)
        return float(torch.mean((left - right) ** 2).detach().cpu().item())


def periodic_derivative_loss(torch, net, dataset):
    left_x = dataset["boundary_left_x"].detach().clone().requires_grad_(True)
    right_x = dataset["boundary_right_x"].detach().clone().requires_grad_(True)
    left = apply_output_activation(torch, net(left_x), str(dataset.get("output_activation", "identity")))
    right = apply_output_activation(torch, net(right_x), str(dataset.get("output_activation", "identity")))
    left_grad = torch.autograd.grad(left.sum(), left_x, create_graph=True)[0][:, 0:1]
    right_grad = torch.autograd.grad(right.sum(), right_x, create_graph=True)[0][:, 0:1]
    return torch.mean((left_grad - right_grad) ** 2)


def evaluate_periodic_derivative(torch, net, dataset) -> float:
    return float(periodic_derivative_loss(torch, net, dataset).detach().cpu().item())


def early_time_loss(torch, net, dataset, config: TrainingConfig):
    if dataset["early_train_x"].numel() == 0:
        return dataset["train_y"].new_tensor(0.0)
    prediction = predict_values(torch, net, dataset["early_train_x"], dataset)
    return data_fit_loss(torch, prediction, dataset["early_train_y"], dataset, config)


def peak_loss(torch, net, dataset, config: TrainingConfig):
    if dataset["peak_train_x"].numel() == 0:
        return dataset["train_y"].new_tensor(0.0)
    prediction = predict_values(torch, net, dataset["peak_train_x"], dataset)
    return data_fit_loss(torch, prediction, dataset["peak_train_y"], dataset, config)


def _spectral_time_indices(torch, dataset, time_batch: int, *, random_sample: bool = True):
    nt = int(dataset["time"].shape[0])
    if time_batch <= 0 or time_batch >= nt:
        return torch.arange(nt, dtype=torch.long)
    if not random_sample:
        return torch.linspace(0, nt - 1, steps=time_batch).round().long()
    return torch.randint(0, nt, (time_batch,), dtype=torch.long)


def _grid_prediction_at_times(torch, net, dataset, time_indices):
    time = torch.as_tensor(dataset["time"], dtype=torch.float32)
    position = torch.as_tensor(dataset["position"], dtype=torch.float32)
    selected_time = time[time_indices]
    tt, xx = torch.meshgrid(selected_time, position, indexing="ij")
    if dataset.get("input_normalization", "none") == "unit_box":
        x_min, x_max = dataset["input_x_range"]
        t_min, t_max = dataset["input_t_range"]
        x_scale = max(float(x_max - x_min), np.finfo(np.float32).eps)
        t_scale = max(float(t_max - t_min), np.finfo(np.float32).eps)
        xx_net = (xx - float(x_min)) / x_scale
        tt_net = (tt - float(t_min)) / t_scale
    else:
        xx_net = xx
        tt_net = tt
    coordinates = torch.stack((xx_net.reshape(-1), tt_net.reshape(-1)), dim=1)
    prediction = predict_values(torch, net, coordinates, dataset)
    return prediction.reshape(selected_time.numel(), position.numel())


def spectral_loss(torch, net, dataset, config: TrainingConfig, *, random_sample: bool = True):
    time_indices = _spectral_time_indices(torch, dataset, config.spectral_time_batch, random_sample=random_sample)
    prediction = _grid_prediction_at_times(torch, net, dataset, time_indices)
    target = dataset["values_grid"][time_indices, :]
    pred_hat = torch.fft.fft(prediction, dim=1) / prediction.shape[1]
    target_hat = torch.fft.fft(target, dim=1) / target.shape[1]
    k_norm = torch.fft.fftfreq(prediction.shape[1], d=float(dataset["position"][1] - dataset["position"][0]))
    k_abs = torch.abs(k_norm)
    max_k = torch.max(k_abs).clamp_min(torch.finfo(prediction.dtype).eps)
    weights = 1.0 + float(config.spectral_high_k_gamma) * (k_abs / max_k) ** float(config.spectral_high_k_power)
    diff = pred_hat - target_hat
    return torch.mean(weights.reshape(1, -1) * (diff.real**2 + diff.imag**2))


def integer_derivative_loss(torch, net, dataset, config: TrainingConfig, *, random_sample: bool = True):
    time_indices = _spectral_time_indices(torch, dataset, config.spectral_time_batch, random_sample=random_sample)
    prediction = _grid_prediction_at_times(torch, net, dataset, time_indices)
    target = dataset["values_grid"][time_indices, :]
    dx = float(dataset["position"][1] - dataset["position"][0])
    k = 2.0 * np.pi * torch.fft.fftfreq(prediction.shape[1], d=dx)
    pred_x = torch.fft.ifft((1j * k).reshape(1, -1) * torch.fft.fft(prediction, dim=1), dim=1).real
    target_x = torch.fft.ifft((1j * k).reshape(1, -1) * torch.fft.fft(target, dim=1), dim=1).real
    return torch.mean((pred_x - target_x) ** 2)


def evaluate_auxiliary_losses(torch, net, dataset, config: TrainingConfig) -> dict[str, float]:
    with torch.no_grad():
        early = float(early_time_loss(torch, net, dataset, config).detach().cpu().item())
    with torch.no_grad():
        peak = float(peak_loss(torch, net, dataset, config).detach().cpu().item())
    spectral = 0.0
    if config.spectral_weight > 0.0:
        with torch.no_grad():
            spectral = float(spectral_loss(torch, net, dataset, config).detach().cpu().item())
    integer_derivative = 0.0
    if config.integer_derivative_weight > 0.0:
        with torch.no_grad():
            integer_derivative = float(integer_derivative_loss(torch, net, dataset, config).detach().cpu().item())
    return {
        "early_time_loss": early,
        "peak_loss": peak,
        "spectral_loss": spectral,
        "integer_derivative_loss": integer_derivative,
    }


def evaluate_selection_losses(
    torch,
    net,
    dataset,
    config: TrainingConfig,
    val_loss: float,
) -> dict[str, float]:
    with torch.no_grad():
        spectral_val = float(spectral_loss(torch, net, dataset, config, random_sample=False).detach().cpu().item())
        integer_derivative_val = float(
            integer_derivative_loss(torch, net, dataset, config, random_sample=False).detach().cpu().item()
        )
    composite = (
        float(val_loss)
        + float(config.selection_derivative_weight) * integer_derivative_val
        + float(config.selection_spectral_weight) * spectral_val
    )
    if config.selection_metric == "val":
        selection_score = float(val_loss)
    elif config.selection_metric == "derivative_composite":
        selection_score = composite
    else:
        raise ValueError("--selection-metric must be one of val, derivative_composite")
    return {
        "spectral_val_loss": spectral_val,
        "integer_derivative_val_loss": integer_derivative_val,
        "composite_derivative_val_loss": composite,
        "selection_score": selection_score,
    }


def train(config: TrainingConfig) -> dict[str, float | int | str]:
    if config.train_points > MAX_TRAIN_POINTS and not config.allow_large_train_set:
        raise ValueError(f"--train-points must be <= {MAX_TRAIN_POINTS} unless --allow-large-train-set is set")

    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(config, torch)
    net = build_network(torch, config.activation, config.hidden_layers, config.neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    log_rows: list[dict[str, float | int]] = []
    best_step = 0
    best_train_loss = evaluate(torch, net, dataset["train_x"], dataset["train_y"], dataset, config)
    best_val_loss = evaluate(torch, net, dataset["val_x"], dataset["val_y"], dataset, config)
    best_periodic_loss = evaluate_periodic_boundary(torch, net, dataset)
    best_periodic_derivative_loss = evaluate_periodic_derivative(torch, net, dataset)
    best_auxiliary = evaluate_auxiliary_losses(torch, net, dataset, config)
    best_selection = evaluate_selection_losses(torch, net, dataset, config, best_val_loss)
    best_selection_score = best_selection["selection_score"]
    plateau_reference_score = float(best_selection_score)
    last_meaningful_improvement_step = 0
    completed_steps = 0
    stop_reason = "max_steps"
    torch.save(net.state_dict(), output_dir / "best.pkl")
    log_rows.append(
        {
            "step": 0,
            "train_loss": best_train_loss,
            "val_loss": best_val_loss,
            "periodic_boundary_loss": best_periodic_loss,
            "periodic_derivative_loss": best_periodic_derivative_loss,
            **best_auxiliary,
            **best_selection,
            "train_objective": (
                best_train_loss
                + config.periodic_boundary_weight * best_periodic_loss
                + config.periodic_derivative_weight * best_periodic_derivative_loss
                + config.early_time_weight * best_auxiliary["early_time_loss"]
                + config.peak_weight * best_auxiliary["peak_loss"]
                + config.spectral_weight * best_auxiliary["spectral_loss"]
                + config.integer_derivative_weight * best_auxiliary["integer_derivative_loss"]
            ),
            "is_best": 1,
        }
    )
    print(
        f"step=0 train_loss={best_train_loss:.8g} val_loss={best_val_loss:.8g} "
        f"periodic_loss={best_periodic_loss:.8g} periodic_deriv_loss={best_periodic_derivative_loss:.8g} best=*"
    )

    for step in range(1, config.max_steps + 1):
        completed_steps = step
        optimizer.zero_grad()
        prediction = predict_values(torch, net, dataset["train_x"], dataset)
        data_loss = data_fit_loss(torch, prediction, dataset["train_y"], dataset, config)
        if config.periodic_boundary_weight > 0.0:
            left = predict_values(torch, net, dataset["boundary_left_x"], dataset)
            right = predict_values(torch, net, dataset["boundary_right_x"], dataset)
            periodic_value_loss = torch.mean((left - right) ** 2)
        else:
            periodic_value_loss = data_loss.new_tensor(0.0)
        if config.periodic_derivative_weight > 0.0:
            periodic_gradient_loss = periodic_derivative_loss(torch, net, dataset)
        else:
            periodic_gradient_loss = data_loss.new_tensor(0.0)
        if config.early_time_weight > 0.0:
            early_loss = early_time_loss(torch, net, dataset, config)
        else:
            early_loss = data_loss.new_tensor(0.0)
        if config.peak_weight > 0.0:
            peak_focus_loss = peak_loss(torch, net, dataset, config)
        else:
            peak_focus_loss = data_loss.new_tensor(0.0)
        if config.spectral_weight > 0.0:
            frequency_loss = spectral_loss(torch, net, dataset, config)
        else:
            frequency_loss = data_loss.new_tensor(0.0)
        if config.integer_derivative_weight > 0.0:
            derivative_loss = integer_derivative_loss(torch, net, dataset, config)
        else:
            derivative_loss = data_loss.new_tensor(0.0)
        loss = (
            data_loss
            + config.periodic_boundary_weight * periodic_value_loss
            + config.periodic_derivative_weight * periodic_gradient_loss
            + config.early_time_weight * early_loss
            + config.peak_weight * peak_focus_loss
            + config.spectral_weight * frequency_loss
            + config.integer_derivative_weight * derivative_loss
        )
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            train_loss = evaluate(torch, net, dataset["train_x"], dataset["train_y"], dataset, config)
            val_loss = evaluate(torch, net, dataset["val_x"], dataset["val_y"], dataset, config)
            periodic_boundary_loss = evaluate_periodic_boundary(torch, net, dataset)
            periodic_boundary_derivative_loss = evaluate_periodic_derivative(torch, net, dataset)
            auxiliary = evaluate_auxiliary_losses(torch, net, dataset, config)
            selection = evaluate_selection_losses(torch, net, dataset, config, val_loss)
            is_best = selection["selection_score"] < best_selection_score
            if is_best:
                best_step = step
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_periodic_loss = periodic_boundary_loss
                best_periodic_derivative_loss = periodic_boundary_derivative_loss
                best_auxiliary = auxiliary
                best_selection = selection
                best_selection_score = selection["selection_score"]
                torch.save(net.state_dict(), output_dir / "best.pkl")
            min_delta = max(
                abs(float(plateau_reference_score)) * max(float(config.min_delta_rel), 0.0),
                np.finfo(float).eps,
            )
            if selection["selection_score"] < plateau_reference_score - min_delta:
                plateau_reference_score = float(selection["selection_score"])
                last_meaningful_improvement_step = step
            log_rows.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "periodic_boundary_loss": periodic_boundary_loss,
                    "periodic_derivative_loss": periodic_boundary_derivative_loss,
                    **auxiliary,
                    **selection,
                    "train_objective": (
                        train_loss
                        + config.periodic_boundary_weight * periodic_boundary_loss
                        + config.periodic_derivative_weight * periodic_boundary_derivative_loss
                        + config.early_time_weight * auxiliary["early_time_loss"]
                        + config.peak_weight * auxiliary["peak_loss"]
                        + config.spectral_weight * auxiliary["spectral_loss"]
                        + config.integer_derivative_weight * auxiliary["integer_derivative_loss"]
                    ),
                    "is_best": int(is_best),
                }
            )
            if step % config.print_every == 0 or step == config.max_steps:
                marker = " best=*" if is_best else ""
                print(
                    f"step={step} train_loss={train_loss:.8g} val_loss={val_loss:.8g} "
                    f"periodic_loss={periodic_boundary_loss:.8g} "
                    f"periodic_deriv_loss={periodic_boundary_derivative_loss:.8g} "
                    f"early_loss={auxiliary['early_time_loss']:.8g} "
                    f"peak_loss={auxiliary['peak_loss']:.8g} "
                    f"spectral_loss={auxiliary['spectral_loss']:.8g} "
                    f"integer_deriv_loss={auxiliary['integer_derivative_loss']:.8g} "
                    f"selection_score={selection['selection_score']:.8g}{marker}"
                )
            if (
                int(config.patience_steps) > 0
                and step - last_meaningful_improvement_step >= int(config.patience_steps)
            ):
                stop_reason = "selection_score_plateau"
                print(
                    f"early_stop step={step} reason={stop_reason} "
                    f"best_step={best_step} best_selection_score={best_selection_score:.8g}",
                    flush=True,
                )
                break

    torch.save(net.state_dict(), output_dir / "last.pkl")

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "train_loss",
                "val_loss",
                "periodic_boundary_loss",
                "periodic_derivative_loss",
                "early_time_loss",
                "peak_loss",
                "spectral_loss",
                "integer_derivative_loss",
                "spectral_val_loss",
                "integer_derivative_val_loss",
                "composite_derivative_val_loss",
                "selection_score",
                "train_objective",
                "is_best",
            ),
        )
        writer.writeheader()
        writer.writerows(log_rows)

    config_payload = asdict(config)
    config_payload.update(
        {
            "time_range": [float(dataset["time"][0]), float(dataset["time"][-1])],
            "x_range": [float(dataset["position"][0]), float(dataset["position"][-1])],
            "field_shape": list(dataset["values_shape"]),
            "best_step": best_step,
            "completed_steps": completed_steps,
            "stop_reason": stop_reason,
            "patience_steps": config.patience_steps,
            "min_delta_rel": config.min_delta_rel,
            "last_meaningful_improvement_step": last_meaningful_improvement_step,
            "plateau_reference_score": plateau_reference_score,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_periodic_boundary_loss": best_periodic_loss,
            "best_periodic_derivative_loss": best_periodic_derivative_loss,
            "best_early_time_loss": best_auxiliary["early_time_loss"],
            "best_peak_loss": best_auxiliary["peak_loss"],
            "best_spectral_loss": best_auxiliary["spectral_loss"],
            "best_integer_derivative_loss": best_auxiliary["integer_derivative_loss"],
            "best_spectral_val_loss": best_selection["spectral_val_loss"],
            "best_integer_derivative_val_loss": best_selection["integer_derivative_val_loss"],
            "best_composite_derivative_val_loss": best_selection["composite_derivative_val_loss"],
            "best_selection_score": best_selection["selection_score"],
            "peak_threshold": float(dataset["peak_threshold"]),
            "peak_mask_fraction": float(dataset["peak_mask_fraction"]),
            "checkpoint": str(output_dir / "best.pkl"),
            "input_normalization": dataset["input_normalization"],
            "input_x_range": dataset["input_x_range"],
            "input_t_range": dataset["input_t_range"],
            "input_x_scale": dataset["input_x_scale"],
            "input_t_scale": dataset["input_t_scale"],
            "output_normalization": dataset["output_normalization"],
            "output_activation": dataset["output_activation"],
            "data_loss": config.data_loss,
            "log_loss_epsilon": dataset["log_loss_epsilon"],
            "physical_log_loss_epsilon": dataset["physical_log_loss_epsilon"],
            "physical_loss_weight": config.physical_loss_weight,
            "log_loss_weight": config.log_loss_weight,
            "target_c_range": dataset["target_c_range"],
            "target_c_shift": dataset["target_c_shift"],
            "target_c_scale": dataset["target_c_scale"],
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    summary = "\n".join(
        [
            "Time-FADE NN reconstruction training",
            f"case: {config.case}",
            f"data_file: {config.data_file}",
            f"output_dir: {config.output_dir}",
            f"field_shape: {dataset['values_shape']}",
            f"time_range: [{float(dataset['time'][0]):.6g}, {float(dataset['time'][-1]):.6g}]",
            f"x_range: [{float(dataset['position'][0]):.6g}, {float(dataset['position'][-1]):.6g}]",
            f"noise_type: {config.noise_type}",
            f"noise_level: {config.noise_level}",
            f"activation: {config.activation}",
            f"hidden_layers: {config.hidden_layers}",
            f"neurons: {config.neurons}",
            f"learning_rate: {config.learning_rate}",
            f"weight_decay: {config.weight_decay}",
            f"train_points: {config.train_points}",
            f"val_points: {config.val_points}",
            f"max_steps: {config.max_steps}",
            f"completed_steps: {completed_steps}",
            f"stop_reason: {stop_reason}",
            f"patience_steps: {config.patience_steps}",
            f"min_delta_rel: {config.min_delta_rel}",
            f"last_meaningful_improvement_step: {last_meaningful_improvement_step}",
            f"plateau_reference_score: {plateau_reference_score:.10g}",
            f"periodic_boundary_weight: {config.periodic_boundary_weight}",
            f"periodic_derivative_weight: {config.periodic_derivative_weight}",
            f"periodic_boundary_points: {config.periodic_boundary_points}",
            f"early_time_weight: {config.early_time_weight}",
            f"early_time_cutoff: {config.early_time_cutoff}",
            f"peak_weight: {config.peak_weight}",
            f"peak_quantile: {config.peak_quantile}",
            f"peak_time_cutoff: {config.peak_time_cutoff}",
            f"peak_use_absolute: {config.peak_use_absolute}",
            f"peak_threshold: {float(dataset['peak_threshold']):.10g}",
            f"peak_mask_fraction: {float(dataset['peak_mask_fraction']):.10g}",
            f"spectral_weight: {config.spectral_weight}",
            f"spectral_high_k_gamma: {config.spectral_high_k_gamma}",
            f"spectral_high_k_power: {config.spectral_high_k_power}",
            f"spectral_time_batch: {config.spectral_time_batch}",
            f"integer_derivative_weight: {config.integer_derivative_weight}",
            f"selection_metric: {config.selection_metric}",
            f"selection_derivative_weight: {config.selection_derivative_weight}",
            f"selection_spectral_weight: {config.selection_spectral_weight}",
            f"gaussian_scale: {config.gaussian_scale}",
            f"denoise_mode: {config.denoise_mode}",
            f"denoise_fft_cutoff: {config.denoise_fft_cutoff}",
            f"allow_large_train_set: {config.allow_large_train_set}",
            f"input_normalization: {config.input_normalization}",
            f"input_x_range: [{dataset['input_x_range'][0]:.6g}, {dataset['input_x_range'][1]:.6g}]",
            f"input_t_range: [{dataset['input_t_range'][0]:.6g}, {dataset['input_t_range'][1]:.6g}]",
            f"input_x_scale: {dataset['input_x_scale']:.10g}",
            f"input_t_scale: {dataset['input_t_scale']:.10g}",
            f"output_normalization: {dataset['output_normalization']}",
            f"output_activation: {dataset['output_activation']}",
            f"data_loss: {config.data_loss}",
            f"log_loss_epsilon: {dataset['log_loss_epsilon']:.10g}",
            f"physical_log_loss_epsilon: {dataset['physical_log_loss_epsilon']:.10g}",
            f"physical_loss_weight: {config.physical_loss_weight}",
            f"log_loss_weight: {config.log_loss_weight}",
            f"target_c_range: [{dataset['target_c_range'][0]:.10g}, {dataset['target_c_range'][1]:.10g}]",
            f"target_c_shift: {dataset['target_c_shift']:.10g}",
            f"target_c_scale: {dataset['target_c_scale']:.10g}",
            f"best_step: {best_step}",
            f"best_train_loss: {best_train_loss:.10g}",
            f"best_val_loss: {best_val_loss:.10g}",
            f"best_periodic_boundary_loss: {best_periodic_loss:.10g}",
            f"best_periodic_derivative_loss: {best_periodic_derivative_loss:.10g}",
            f"best_early_time_loss: {best_auxiliary['early_time_loss']:.10g}",
            f"best_peak_loss: {best_auxiliary['peak_loss']:.10g}",
            f"best_spectral_loss: {best_auxiliary['spectral_loss']:.10g}",
            f"best_integer_derivative_loss: {best_auxiliary['integer_derivative_loss']:.10g}",
            f"best_spectral_val_loss: {best_selection['spectral_val_loss']:.10g}",
            f"best_integer_derivative_val_loss: {best_selection['integer_derivative_val_loss']:.10g}",
            f"best_composite_derivative_val_loss: {best_selection['composite_derivative_val_loss']:.10g}",
            f"best_selection_score: {best_selection['selection_score']:.10g}",
            "checkpoint: best.pkl",
        ]
    )
    (output_dir / "training_summary.txt").write_text(summary + "\n", encoding="utf-8")
    return config_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=TrainingConfig.case, choices=("tsfade_fft", "periodic_tfade_fft"))
    parser.add_argument("--data-file", default=TrainingConfig.data_file)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--train-points", type=int, default=TrainingConfig.train_points)
    parser.add_argument("--val-points", type=int, default=TrainingConfig.val_points)
    parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--noise-level", type=float, default=TrainingConfig.noise_level)
    parser.add_argument(
        "--noise-type",
        choices=("none", "uniform_relative", "gaussian_relative", "gaussian_additive"),
        default=TrainingConfig.noise_type,
    )
    parser.add_argument("--activation", choices=("sin", "tanh", "gaussian"), default=TrainingConfig.activation)
    parser.add_argument("--hidden-layers", type=int, default=TrainingConfig.hidden_layers)
    parser.add_argument("--neurons", type=int, default=TrainingConfig.neurons)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--eval-every", type=int, default=TrainingConfig.eval_every)
    parser.add_argument("--print-every", type=int, default=TrainingConfig.print_every)
    parser.add_argument("--patience-steps", type=int, default=TrainingConfig.patience_steps)
    parser.add_argument("--min-delta-rel", type=float, default=TrainingConfig.min_delta_rel)
    parser.add_argument("--periodic-boundary-weight", type=float, default=TrainingConfig.periodic_boundary_weight)
    parser.add_argument("--periodic-derivative-weight", type=float, default=TrainingConfig.periodic_derivative_weight)
    parser.add_argument("--periodic-boundary-points", type=int, default=TrainingConfig.periodic_boundary_points)
    parser.add_argument("--early-time-weight", type=float, default=TrainingConfig.early_time_weight)
    parser.add_argument("--early-time-cutoff", type=float, default=TrainingConfig.early_time_cutoff)
    parser.add_argument("--peak-weight", type=float, default=TrainingConfig.peak_weight)
    parser.add_argument("--peak-quantile", type=float, default=TrainingConfig.peak_quantile)
    parser.add_argument("--peak-time-cutoff", type=float, default=TrainingConfig.peak_time_cutoff)
    parser.add_argument("--peak-use-absolute", action="store_true", default=TrainingConfig.peak_use_absolute)
    parser.add_argument("--peak-use-signed", action="store_true", help="use signed values instead of absolute values for peak selection")
    parser.add_argument("--spectral-weight", type=float, default=TrainingConfig.spectral_weight)
    parser.add_argument("--spectral-high-k-gamma", type=float, default=TrainingConfig.spectral_high_k_gamma)
    parser.add_argument("--spectral-high-k-power", type=float, default=TrainingConfig.spectral_high_k_power)
    parser.add_argument("--spectral-time-batch", type=int, default=TrainingConfig.spectral_time_batch)
    parser.add_argument("--integer-derivative-weight", type=float, default=TrainingConfig.integer_derivative_weight)
    parser.add_argument("--gaussian-scale", type=float, default=TrainingConfig.gaussian_scale)
    parser.add_argument("--denoise-mode", choices=("none", "hard_lowpass", "soft_spectral"), default=TrainingConfig.denoise_mode)
    parser.add_argument("--denoise-fft-cutoff", type=float, default=TrainingConfig.denoise_fft_cutoff)
    parser.add_argument("--allow-large-train-set", action="store_true", default=TrainingConfig.allow_large_train_set)
    parser.add_argument("--input-normalization", choices=("none", "unit_box"), default=TrainingConfig.input_normalization)
    parser.add_argument("--output-normalization", choices=("none", "unit_interval"), default=TrainingConfig.output_normalization)
    parser.add_argument("--output-activation", choices=("auto", "identity", "softplus"), default=TrainingConfig.output_activation)
    parser.add_argument("--data-loss", choices=("mse", "log_mse", "hybrid_log_mse"), default=TrainingConfig.data_loss)
    parser.add_argument("--log-loss-epsilon", type=float, default=TrainingConfig.log_loss_epsilon)
    parser.add_argument("--physical-loss-weight", type=float, default=TrainingConfig.physical_loss_weight)
    parser.add_argument("--log-loss-weight", type=float, default=TrainingConfig.log_loss_weight)
    parser.add_argument(
        "--selection-metric",
        choices=("val", "derivative_composite"),
        default=TrainingConfig.selection_metric,
        help="checkpoint selection metric; derivative_composite uses noisy-only spectral/derivative validation losses",
    )
    parser.add_argument("--selection-derivative-weight", type=float, default=TrainingConfig.selection_derivative_weight)
    parser.add_argument("--selection-spectral-weight", type=float, default=TrainingConfig.selection_spectral_weight)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        case=args.case,
        data_file=args.data_file,
        output_dir=args.output_dir,
        train_points=args.train_points,
        val_points=args.val_points,
        max_steps=args.max_steps,
        noise_level=args.noise_level,
        noise_type=args.noise_type,
        activation=f"gaussian:{args.gaussian_scale}" if args.activation == "gaussian" else args.activation,
        hidden_layers=args.hidden_layers,
        neurons=args.neurons,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_every=args.eval_every,
        print_every=args.print_every,
        patience_steps=args.patience_steps,
        min_delta_rel=args.min_delta_rel,
        periodic_boundary_weight=args.periodic_boundary_weight,
        periodic_derivative_weight=args.periodic_derivative_weight,
        periodic_boundary_points=args.periodic_boundary_points,
        early_time_weight=args.early_time_weight,
        early_time_cutoff=args.early_time_cutoff,
        peak_weight=args.peak_weight,
        peak_quantile=args.peak_quantile,
        peak_time_cutoff=args.peak_time_cutoff,
        peak_use_absolute=not bool(args.peak_use_signed),
        spectral_weight=args.spectral_weight,
        spectral_high_k_gamma=args.spectral_high_k_gamma,
        spectral_high_k_power=args.spectral_high_k_power,
        spectral_time_batch=args.spectral_time_batch,
        integer_derivative_weight=args.integer_derivative_weight,
        gaussian_scale=args.gaussian_scale,
        denoise_mode=args.denoise_mode,
        denoise_fft_cutoff=args.denoise_fft_cutoff,
        allow_large_train_set=args.allow_large_train_set,
        input_normalization=args.input_normalization,
        output_normalization=args.output_normalization,
        output_activation=args.output_activation,
        data_loss=args.data_loss,
        log_loss_epsilon=args.log_loss_epsilon,
        physical_loss_weight=args.physical_loss_weight,
        log_loss_weight=args.log_loss_weight,
        selection_metric=args.selection_metric,
        selection_derivative_weight=args.selection_derivative_weight,
        selection_spectral_weight=args.selection_spectral_weight,
    )
    result = train(config)
    print(f"best_step={result['best_step']} best_val_loss={result['best_val_loss']:.8g}")
    print(f"checkpoint={result['checkpoint']}")


if __name__ == "__main__":
    main()
