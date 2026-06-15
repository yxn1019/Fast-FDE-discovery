"""Self-consistent derivative robustness diagnostics for retrained tsfade.

Each operator is compared against its own clean-surrogate derivative column:

    ||D_operator(noisy NN) - D_operator(clean NN)||_2
    -------------------------------------------------
              ||D_operator(clean NN)||_2

Space and time derivatives are reported independently; no discovery or support
selection is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from paper_case_params import make_config  # noqa: E402
from plot_raw_vs_nn_surrogate import reconstruct_nn_on_raw_grid  # noqa: E402
from transporteq_discovery.fractional_discoverer import FractionalPDEDiscoverer  # noqa: E402
from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer  # noqa: E402
from transporteq_discovery.spatial_fractional import fourier_fractional_derivative  # noqa: E402


TRUE_ALPHA = 0.78
TRUE_BETA = 1.83
FIT_X_MIN = 4.0
FIT_X_MAX = 26.0
FIT_T_MIN = 3.0
FIT_T_MAX = 14.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help="optional legacy training summary; omitted by default so diagnostics use repo-local paper artifacts",
    )
    parser.add_argument("--noise-levels", default="5,25")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT
        / "results"
        / "derivative_robustness"
        / "tsfade_derivative_robustness_self_consistent_normalized_tanh.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT
        / "results"
        / "derivative_robustness"
        / "tsfade_derivative_robustness_self_consistent_normalized_tanh.json",
    )
    return parser.parse_args()


def parse_noise_levels(text: str) -> set[float]:
    return {float(item.strip()) for item in text.split(",") if item.strip()}


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path)
    return np.asarray(payload["x"], dtype=float).reshape(-1), np.asarray(payload["t"], dtype=float).reshape(-1)


def crop(values: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x_mask = (x >= FIT_X_MIN) & (x < FIT_X_MAX)
    t_mask = (t >= FIT_T_MIN) & (t < FIT_T_MAX)
    return values[np.ix_(t_mask, x_mask)]


def rel_l2(noisy: np.ndarray, clean: np.ndarray) -> float:
    noisy = np.asarray(noisy, dtype=float)
    clean = np.asarray(clean, dtype=float)
    mask = np.isfinite(noisy) & np.isfinite(clean)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.linalg.norm(noisy[mask] - clean[mask]) / max(np.linalg.norm(clean[mask]), np.finfo(float).eps))


def rmse(noisy: np.ndarray, clean: np.ndarray) -> float:
    mask = np.isfinite(noisy) & np.isfinite(clean)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    err = noisy[mask] - clean[mask]
    return float(np.sqrt(np.mean(err**2)))


def max_abs(noisy: np.ndarray, clean: np.ndarray) -> float:
    mask = np.isfinite(noisy) & np.isfinite(clean)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.max(np.abs(noisy[mask] - clean[mask])))


def base_config(checkpoint: Path, *, gj_grid: bool = False) -> Any:
    if gj_grid:
        return make_config(
            case_name="tsfade_fft",
            checkpoint_file=checkpoint,
            model_root=checkpoint.parent.parent,
            model_alpha_tag="retrained_alpha078_beta183_normalized_tanh",
            trained_point=2000,
            noise_level=0.0,
            activation="tanh",
            hidden_layers=8,
            neurons=20,
            x_min=0.0,
            x_max=30.0,
            x_step=0.25,
            t_min=0.0,
            t_max=15.0,
            t_step=0.1,
            fit_x_min=FIT_X_MIN,
            fit_x_max=FIT_X_MAX,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
            laguerre_nodes=15,
            enable_spatial_fractional=True,
            beta_reference_orders=(2.0, 1.9, 1.8, 1.7, 1.6, 1.5),
            space_derivative_mode="autodiff",
            spatial_fractional_mode="gj_richardson",
            spatial_correction_method="gj_richardson",
            time_operator_mode="laplace_taylor",
            refit_mode="none",
        )
    return make_config(
        case_name="tsfade_fft",
        checkpoint_file=checkpoint,
        model_root=checkpoint.parent.parent,
        model_alpha_tag="retrained_alpha078_beta183_normalized_tanh",
        trained_point=2000,
        noise_level=0.0,
        activation="tanh",
        hidden_layers=8,
        neurons=20,
        x_min=0.0,
        x_max=30.0,
        x_step=0.25,
        t_min=0.0,
        t_max=15.0,
        t_step=0.1,
        fit_x_min=FIT_X_MIN,
        fit_x_max=FIT_X_MAX,
        fit_t_min=FIT_T_MIN,
        fit_t_max=FIT_T_MAX,
        laguerre_nodes=5,
        enable_spatial_fractional=True,
        beta_reference_orders=(2.0, 1.9, 1.8, 1.7, 1.6, 1.5),
        space_derivative_mode="fft",
        spatial_fractional_mode="gl_pycaputo",
        spatial_correction_method="gl_pycaputo",
        time_operator_mode="l1_pycaputo",
        spatial_log_quad_points=40,
        spatial_boundary_mode="raw",
        refit_mode="none",
    )


def fft_space(field: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    return crop(fourier_fractional_derivative(field, dx=dx, beta=TRUE_BETA, spatial_axis=-1), x, t_grid_global)


def gl_space(field: np.ndarray, x: np.ndarray, t: np.ndarray, checkpoint: Path) -> np.ndarray:
    discoverer = FractionalPDEDiscoverer(base_config(checkpoint, gj_grid=False))
    values = discoverer._spatial_fractional_values(field, x, TRUE_BETA)
    return crop(values, x, t)


def l1_time(field: np.ndarray, x: np.ndarray, t: np.ndarray, checkpoint: Path) -> np.ndarray:
    discoverer = FractionalPDEDiscoverer(base_config(checkpoint, gj_grid=False))
    values = discoverer._caputo_l1_columnwise(field, t, TRUE_ALPHA)
    return crop(values, x, t)


def gj_space_and_time(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runner = GJHybridDiscoverer(base_config(checkpoint, gj_grid=True))
    torch, net, metadata = runner._load_network()
    field = runner._build_base_field(torch, net, metadata)
    time_col = runner._compute_halpha(torch, net, field, TRUE_ALPHA)
    space_col = runner._compute_hbeta(torch, net, field, TRUE_BETA)
    return field.position, field.time, space_col, time_col


def repo_training_summary() -> dict[str, Any]:
    raw_root = ROOT / "data" / "tsfade_retrained_alpha078_beta183" / "raw_data"
    model_root = ROOT / "data" / "models" / "tsfade_retrained_alpha078_beta183_normalized_tanh"
    return {
        "source": "repo-local normalized tanh paper artifacts",
        "raw_data": {"mat_path": str(raw_root / "tsfade_alpha078_beta183_paper_grid.mat")},
        "training": [
            {
                "noise_level": 0.0,
                "checkpoint": str(model_root / "draft-2000-0" / "best.pkl"),
                "noisy_data_mat": str(raw_root / "tsfade_alpha078_beta183_noise0.mat"),
            },
            {
                "noise_level": 5.0,
                "checkpoint": str(model_root / "draft-2000-5" / "best.pkl"),
                "noisy_data_mat": str(raw_root / "tsfade_alpha078_beta183_noise5.mat"),
            },
            {
                "noise_level": 25.0,
                "checkpoint": str(model_root / "draft-2000-25" / "best.pkl"),
                "noisy_data_mat": str(raw_root / "tsfade_alpha078_beta183_noise25.mat"),
            },
        ],
    }


def row(
    *,
    noise: float,
    derivative_type: str,
    operator: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    clean_checkpoint: Path,
    noisy_checkpoint: Path,
    runtime_seconds: float,
    grid_note: str,
) -> dict[str, Any]:
    return {
        "noise_level": noise,
        "derivative_type": derivative_type,
        "operator": operator,
        "relative_l2_vs_clean": rel_l2(noisy, clean),
        "rmse_vs_clean": rmse(noisy, clean),
        "max_abs_vs_clean": max_abs(noisy, clean),
        "runtime_seconds": float(runtime_seconds),
        "reference": "same-operator clean surrogate derivative column",
        "checkpoint_clean": str(clean_checkpoint),
        "checkpoint_noisy": str(noisy_checkpoint),
        "truth_alpha_for_order": TRUE_ALPHA,
        "truth_beta_for_order": TRUE_BETA,
        "fit_window": "x in [4,26), t in [3,14)",
        "grid_note": grid_note,
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_rows(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    requested_csv = csv_path
    requested_json = json_path
    keys: list[str] = []
    for item in rows:
        for key in item:
            if key not in keys:
                keys.append(key)
    last_error: Exception | None = None
    for directory in (
        requested_csv.parent,
        Path(tempfile.gettempdir()) / "linear_fractional_results" / "derivative_robustness",
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / requested_csv.name
            json_path = directory / requested_json.name
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            json_path.write_text(json.dumps(sanitize(rows), indent=2), encoding="utf-8")
            print(f"wrote {csv_path}")
            print(f"wrote {json_path}")
            return
        except PermissionError as exc:
            last_error = exc
            continue
    raise PermissionError(f"No writable output directory found for derivative robustness diagnostics: {last_error}")


def main() -> int:
    args = parse_args()
    summary = (
        json.loads(args.training_summary.read_text(encoding="utf-8"))
        if args.training_summary is not None
        else repo_training_summary()
    )
    requested = parse_noise_levels(args.noise_levels)
    by_noise = {float(item["noise_level"]): item for item in summary["training"]}
    clean_item = by_noise[0.0]
    clean_checkpoint = Path(clean_item["checkpoint"])
    x, t = read_grid(Path(summary["raw_data"]["mat_path"]))
    global t_grid_global
    t_grid_global = t
    clean_field = reconstruct_nn_on_raw_grid(clean_checkpoint, x, t)

    rows: list[dict[str, Any]] = []
    for noise in sorted(requested):
        noisy_item = by_noise[float(noise)]
        noisy_checkpoint = Path(noisy_item["checkpoint"])
        noisy_field = reconstruct_nn_on_raw_grid(noisy_checkpoint, x, t)

        started = time.perf_counter()
        clean_fft = crop(fourier_fractional_derivative(clean_field, dx=float(x[1] - x[0]), beta=TRUE_BETA, spatial_axis=-1), x, t)
        noisy_fft = crop(fourier_fractional_derivative(noisy_field, dx=float(x[1] - x[0]), beta=TRUE_BETA, spatial_axis=-1), x, t)
        rows.append(row(noise=noise, derivative_type="space", operator="FFT/Fourier", clean=clean_fft, noisy=noisy_fft, clean_checkpoint=clean_checkpoint, noisy_checkpoint=noisy_checkpoint, runtime_seconds=time.perf_counter() - started, grid_note="full raw spatial grid FFT, then interior crop"))

        started = time.perf_counter()
        clean_gl = gl_space(clean_field, x, t, clean_checkpoint)
        noisy_gl = gl_space(noisy_field, x, t, noisy_checkpoint)
        rows.append(row(noise=noise, derivative_type="space", operator="G-L/pycaputo spatial", clean=clean_gl, noisy=noisy_gl, clean_checkpoint=clean_checkpoint, noisy_checkpoint=noisy_checkpoint, runtime_seconds=time.perf_counter() - started, grid_note="raw grid, interior crop"))

        started = time.perf_counter()
        clean_gj_x, clean_gj_t, clean_gj_space, clean_gj_time = gj_space_and_time(clean_checkpoint)
        noisy_gj_x, noisy_gj_t, noisy_gj_space, noisy_gj_time = gj_space_and_time(noisy_checkpoint)
        if not (np.allclose(clean_gj_x, noisy_gj_x) and np.allclose(clean_gj_t, noisy_gj_t)):
            raise ValueError("G-J clean/noisy grids are not aligned")
        gj_runtime = time.perf_counter() - started
        clean_gj_space_fit = crop(clean_gj_space, clean_gj_x, clean_gj_t)
        noisy_gj_space_fit = crop(noisy_gj_space, noisy_gj_x, noisy_gj_t)
        clean_gj_time_fit = crop(clean_gj_time, clean_gj_x, clean_gj_t)
        noisy_gj_time_fit = crop(noisy_gj_time, noisy_gj_x, noisy_gj_t)
        rows.append(row(noise=noise, derivative_type="space", operator="G-J spatial", clean=clean_gj_space_fit, noisy=noisy_gj_space_fit, clean_checkpoint=clean_checkpoint, noisy_checkpoint=noisy_checkpoint, runtime_seconds=gj_runtime, grid_note="full-domain G-J operator grid, then interior crop"))

        started = time.perf_counter()
        clean_l1 = l1_time(clean_field, x, t, clean_checkpoint)
        noisy_l1 = l1_time(noisy_field, x, t, noisy_checkpoint)
        rows.append(row(noise=noise, derivative_type="time", operator="L1/pycaputo Caputo", clean=clean_l1, noisy=noisy_l1, clean_checkpoint=clean_checkpoint, noisy_checkpoint=noisy_checkpoint, runtime_seconds=time.perf_counter() - started, grid_note="raw grid, interior crop"))

        rows.append(row(noise=noise, derivative_type="time", operator="G-J Caputo", clean=clean_gj_time_fit, noisy=noisy_gj_time_fit, clean_checkpoint=clean_checkpoint, noisy_checkpoint=noisy_checkpoint, runtime_seconds=0.0, grid_note="full-domain G-J time column reused from G-J spatial/time evaluation, then interior crop"))

    write_rows(rows, args.output_csv, args.output_json)
    print(f"rows={len(rows)}")
    print(f"output_csv={args.output_csv}")
    print(f"output_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
