"""Run the fractional discovery workflows used by the paper cases."""

from __future__ import annotations

import argparse
import subprocess
import re
import sys
import tempfile
from pathlib import Path

# Reproducibility: multithreaded BLAS/OpenMP reduction order can flip near-tied
# STRidge model selections run-to-run. Pin thread counts to 1 *before* NumPy/torch
# load. Their native runtimes read these vars at load time, which can precede this
# module finishing import, so setting them post-import is unreliable; instead we
# set them and relaunch the interpreter once in a child that inherits them. Export
# any of these vars (or _FDE_THREADS_PINNED=1) before launching to override.
import os

if os.environ.get("_FDE_THREADS_PINNED") != "1":
    os.environ["_FDE_THREADS_PINNED"] = "1"
    for _thread_env in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_thread_env, "1")
    raise SystemExit(subprocess.run([sys.executable, *sys.argv]).returncode)

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_case_params import (
    BETA_BOUNDS,
    BETA_REFERENCE_ORDERS,
    ACTIVE_MODEL_PRESET,
    ACTIVE_PAPER_EXAMPLE,
    CASES,
    DEFAULT_ALPHA_TAG,
    DEFAULT_MODEL_CHECKPOINT,
    DEFAULT_MODEL_ROOT,
    ENABLE_SPATIAL_FRACTIONAL,
    MODEL_PRESETS,
    PAPER_ALPHA_CORRECTION_TOL,
    PAPER_BETA_CORRECTION_TOL,
    PAPER_FRACTIONAL_CORRECTION_LAMB,
    PAPER_FRACTIONAL_CORRECTION_TOL_SCALE,
    PAPER_FSTRIDGE_LAMB,
    PAPER_EXAMPLES,
    PAPER_REPRODUCTION_GROUPS,
    PAPER_TASK_SCRIPTS,
    PAPER_LAGUERRE_NODES,
    PAPER_LAPLACE_S_MAX,
    PAPER_LAPLACE_S_MIN,
    PAPER_LAPLACE_S_POINTS,
    PAPER_T_MAX,
    PAPER_T_MIN,
    PAPER_T_STEP,
    PAPER_FIT_X_MIN,
    PAPER_FIT_X_MAX,
    PAPER_FIT_T_MIN,
    PAPER_FIT_T_MAX,
    PAPER_DELTA_ALPHA_PRUNE_THRESHOLD,
    PAPER_DELTA_ALPHA_BOUNDS,
    PAPER_DELTA_BETA_PRUNE_THRESHOLD,
    PAPER_D_TOL,
    PAPER_ORDER_RADIUS_MODE,
    PAPER_ORDER_RADIUS_TOLERANCE,
    PAPER_REFIT_MODE,
    PAPER_SELECTION_OBJECTIVE,
    PAPER_X_MAX,
    PAPER_X_MIN,
    PAPER_X_STEP,
    PAPER_ORDER_UPDATE_MODE,
    PAPER_ITER_START_ALPHA,
    PAPER_ITER_START_BETA,
    PAPER_ITER_MAX_ITERS,
    PAPER_ITER_ORDER_TOL,
    PAPER_ITER_DAMPING,
    PAPER_ITER_MAX_STEP_ALPHA,
    PAPER_ITER_MAX_STEP_BETA,
    PAPER_ITER_DELTA_ALPHA_BOUNDS,
    PAPER_ITER_DELTA_BETA_BOUNDS,
    PAPER_ITER_STOP_ON_NON_DECREASING_OBJECTIVE,
    PAPER_ITER_OBJECTIVE_MIN_DELTA,
    PAPER_ITER_SELECTION_MODE,
    SPATIAL_FRACTIONAL_LOWER_BOUND,
    SPATIAL_BOUNDARY_MODE,
    SPATIAL_BOUNDARY_VALUE,
    SPATIAL_CORRECTION_METHOD,
    SPATIAL_FRACTIONAL_MODE,
    SPATIAL_LOG_QUAD_POINTS,
    SPACE_DERIVATIVE_MODE,
    TSFADE_DEFAULT_ROUTE,
    TSFADE_GJ_QUADRATURE_POINTS,
    TIME_OPERATOR_MODE,
    apply_paper_example_overrides,
    build_model_file,
    format_paper_examples,
    make_config,
    paper_reproduction_commands,
    preset_for_case,
)
from transporteq_discovery.fractional_discoverer import FractionalPDEDiscoverer
from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer
from transporteq_discovery.reporting import format_console_result, format_detailed_result


def _sanitize_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "checkpoint"


def _checkpoint_derived_example_name(checkpoint: Path) -> str:
    """Return a display name that follows an explicit checkpoint path."""
    path = checkpoint.expanduser()
    leaf = path.stem if path.suffix else path.name
    parent = path.parent.name
    grandparent = path.parent.parent.name
    if path.suffix.lower() == ".pkl" and parent:
        if grandparent and grandparent.lower() not in {"model_save", "models", "data"}:
            leaf = f"{grandparent}_{parent}"
        else:
            leaf = parent
    return _sanitize_label(leaf)


def _truth_from_checkpoint_label(label: str) -> tuple[float | None, float | None]:
    """Infer orders only when the checkpoint label explicitly encodes them."""
    compact = label.lower().replace(".", "")
    match = re.search(r"alpha(?P<alpha>\d{3})[_-]?beta(?P<beta>\d{3})", compact)
    if not match:
        return None, None
    return float(match.group("alpha")) / 100.0, float(match.group("beta")) / 100.0



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-example",
        choices=sorted(PAPER_EXAMPLES),
        default=None,
        help=(
            "one-step paper example selector; overrides --case/--example. "
            f"Default behavior is equivalent to --paper-example {ACTIVE_PAPER_EXAMPLE}"
        ),
    )
    parser.add_argument(
        "--list-paper-examples",
        action="store_true",
        help="print available paper examples with their case/preset mapping and exit",
    )
    parser.add_argument(
        "--paper-task",
        choices=sorted(PAPER_TASK_SCRIPTS),
        default=None,
        help=(
            "run a manuscript-level batch task through main.py. "
            "Single equations should usually use --paper-example."
        ),
    )
    parser.add_argument(
        "--paper-reproduce",
        choices=sorted(PAPER_REPRODUCTION_GROUPS),
        default=None,
        help=(
            "run a curated group of manuscript reproduction commands through main.py; "
            "use --paper-reproduce-dry-run to print commands only"
        ),
    )
    parser.add_argument(
        "--paper-reproduce-dry-run",
        action="store_true",
        help="print the commands selected by --paper-reproduce without executing them",
    )
    parser.add_argument(
        "--legacy-maxiter",
        type=int,
        default=100,
        help="DE maxiter used by --paper-task legacy-de; the manuscript benchmark uses 100",
    )
    parser.add_argument(
        "--legacy-noise-levels",
        default="0,5,25",
        help="comma-separated noise levels for --paper-task legacy-de",
    )
    parser.add_argument(
        "--legacy-stridge-mode",
        choices=("legacy", "same_hyperparams", "same_stridge_core"),
        default="same_stridge_core",
        help="STRidge fairness mode for --paper-task legacy-de",
    )
    parser.add_argument(
        "--legacy-out-dir",
        type=Path,
        default=ROOT / "results" / "legacy_de_normalized_tsfade_alpha078_beta183",
        help="output directory for --paper-task legacy-de",
    )
    parser.add_argument(
        "--legacy-quiet",
        action="store_true",
        help="suppress per-trial DE logging for --paper-task legacy-de",
    )
    parser.add_argument(
        "--force-data-regeneration",
        action="store_true",
        help="overwrite existing analytical/periodic data when using generate-* paper tasks",
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="tsfade_fft",
        help="low-level discovery case; --paper-example is the clearer paper-facing selector",
    )
    parser.add_argument(
        "--example",
        choices=sorted(MODEL_PRESETS),
        default=ACTIVE_MODEL_PRESET,
        help="named dataset/model preset; default runs the current paper tsfade retrained alpha=0.78 beta=1.83 benchmark",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--legacy-checkpoint-layout", action="store_true", help="use model-root/model-file/checkpoint-iteration instead of --checkpoint")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-alpha-tag", default=None, help="alpha portion of model filename, e.g. '082' for alpha=0.82")
    parser.add_argument("--trained-point", type=int, default=None)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--neurons", type=int, default=None)
    parser.add_argument("--checkpoint-iteration", type=int, default=5000)
    parser.add_argument("--lamb", type=float, default=None, help="fSTRidge sparsity lamb for non-correction terms")
    parser.add_argument(
        "--fractional-correction-lamb",
        type=float,
        default=PAPER_FRACTIONAL_CORRECTION_LAMB,
        help="optional fSTRidge sparsity lamb for internal alpha/spatial correction terms; paper default is 0",
    )
    parser.add_argument(
        "--fractional-correction-tol-scale",
        type=float,
        default=PAPER_FRACTIONAL_CORRECTION_TOL_SCALE,
        help="STRidge threshold multiplier for alpha/spatial correction columns; 0.1 means tol/10",
    )
    parser.add_argument(
        "--alpha-correction-tol",
        type=float,
        default=None,
        help="order-correction clipping threshold for delta_alpha; default is case-specific",
    )
    parser.add_argument(
        "--beta-correction-tol",
        type=float,
        default=None,
        help="order-correction clipping threshold for delta_beta; default is case-specific",
    )
    parser.add_argument("--d-tol", type=float, default=None, help="initial STRidge threshold step")
    parser.add_argument("--delta-alpha-threshold", type=float, default=PAPER_DELTA_ALPHA_PRUNE_THRESHOLD)
    parser.add_argument("--delta-alpha-min", type=float, default=PAPER_DELTA_ALPHA_BOUNDS[0])
    parser.add_argument("--delta-alpha-max", type=float, default=PAPER_DELTA_ALPHA_BOUNDS[1])
    parser.add_argument("--delta-beta-threshold", type=float, default=PAPER_DELTA_BETA_PRUNE_THRESHOLD)
    parser.add_argument("--x-min", type=float, default=PAPER_X_MIN)
    parser.add_argument("--x-max", type=float, default=PAPER_X_MAX)
    parser.add_argument("--x-step", type=float, default=PAPER_X_STEP)
    parser.add_argument("--t-min", type=float, default=PAPER_T_MIN)
    parser.add_argument("--t-max", type=float, default=PAPER_T_MAX)
    parser.add_argument("--t-step", type=float, default=PAPER_T_STEP)
    parser.add_argument("--nx", type=int, default=200, help="analytic_tfade grid points in x")
    parser.add_argument("--nt", type=int, default=200, help="analytic_tfade grid points in t")
    parser.add_argument("--fit-x-min", type=float, default=PAPER_FIT_X_MIN, help="optional inclusive x lower bound for regression/Laplace fit window")
    parser.add_argument("--fit-x-max", type=float, default=PAPER_FIT_X_MAX, help="optional exclusive x upper bound for regression/Laplace fit window")
    parser.add_argument("--fit-t-min", type=float, default=PAPER_FIT_T_MIN, help="optional inclusive t lower bound for regression/Laplace fit window")
    parser.add_argument("--fit-t-max", type=float, default=PAPER_FIT_T_MAX, help="optional exclusive t upper bound for regression/Laplace fit window")
    parser.add_argument("--s-min", type=float, default=None, help="Laplace s-window lower bound; default is case-specific")
    parser.add_argument("--s-max", type=float, default=None, help="Laplace s-window upper bound; default is case-specific")
    parser.add_argument("--s-points", type=int, default=PAPER_LAPLACE_S_POINTS)
    parser.add_argument("--laguerre-nodes", type=int, default=PAPER_LAGUERRE_NODES)
    parser.add_argument("--enable-spatial-fractional", action="store_true", default=None)
    parser.add_argument("--disable-spatial-fractional", action="store_true", help="force-disable the spatial beta scan for a preset")
    parser.add_argument("--spatial-log-quad-points", type=int, default=SPATIAL_LOG_QUAD_POINTS)
    parser.add_argument(
        "--space-derivative-mode",
        choices=("fft", "autodiff"),
        default=SPACE_DERIVATIVE_MODE,
        help="backend for integer spatial derivatives on NN surrogates",
    )
    parser.add_argument(
        "--spatial-fractional-mode",
        choices=("fourier_taylor", "gj_richardson", "gl_pycaputo"),
        default=SPATIAL_FRACTIONAL_MODE,
        help="backend for spatial fractional columns and final D_x^beta H refit",
    )
    parser.add_argument(
        "--time-operator-mode",
        choices=("laplace_taylor", "l1_pycaputo"),
        default=TIME_OPERATOR_MODE,
        help="time-fractional discovery backend; l1_pycaputo is diagnostic only",
    )
    parser.add_argument(
        "--tsfade-route",
        choices=("gj_hybrid_taylor", "laplace_taylor_fourier"),
        default=TSFADE_DEFAULT_ROUTE,
        help="tsfade_fft discovery route; gj_hybrid_taylor is the default production path",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("augmented", "physical-refit", "physical-stridge"),
        default=PAPER_SELECTION_OBJECTIVE,
        help="candidate-selection objective after the augmented STRidge scan",
    )
    parser.add_argument(
        "--refit-mode",
        choices=("none", "linear", "nonlinear"),
        default=PAPER_REFIT_MODE,
        help="post-selection physical refit for G-J tsfade route; linear is retained as a diagnostic",
    )
    parser.add_argument(
        "--order-update-mode",
        choices=("grid", "iterative"),
        default=PAPER_ORDER_UPDATE_MODE,
        help="order search/update strategy; iterative repeatedly applies fractional correction from a local center",
    )
    parser.add_argument("--iter-start-alpha", type=float, default=PAPER_ITER_START_ALPHA)
    parser.add_argument("--iter-start-beta", type=float, default=PAPER_ITER_START_BETA)
    parser.add_argument(
        "--iter-start-beta-values",
        default=None,
        help="comma-separated beta0 multi-start values for iterative order updates, e.g. 2.0,1.8,1.7",
    )
    parser.add_argument("--iter-max-iters", type=int, default=PAPER_ITER_MAX_ITERS)
    parser.add_argument("--iter-order-tol", type=float, default=PAPER_ITER_ORDER_TOL)
    parser.add_argument("--iter-damping", type=float, default=PAPER_ITER_DAMPING)
    parser.add_argument("--iter-max-step-alpha", type=float, default=PAPER_ITER_MAX_STEP_ALPHA)
    parser.add_argument("--iter-max-step-beta", type=float, default=PAPER_ITER_MAX_STEP_BETA)
    parser.add_argument("--iter-delta-alpha-min", type=float, default=PAPER_ITER_DELTA_ALPHA_BOUNDS[0])
    parser.add_argument("--iter-delta-alpha-max", type=float, default=PAPER_ITER_DELTA_ALPHA_BOUNDS[1])
    parser.add_argument("--iter-delta-beta-min", type=float, default=PAPER_ITER_DELTA_BETA_BOUNDS[0])
    parser.add_argument("--iter-delta-beta-max", type=float, default=PAPER_ITER_DELTA_BETA_BOUNDS[1])
    parser.add_argument(
        "--iter-allow-nondecreasing-objective",
        action="store_true",
        help="continue iterative order updates even when the regularized objective does not decrease",
    )
    parser.add_argument("--iter-objective-min-delta", type=float, default=PAPER_ITER_OBJECTIVE_MIN_DELTA)
    parser.add_argument(
        "--iter-selection-mode",
        choices=("best_objective", "final"),
        default=PAPER_ITER_SELECTION_MODE,
        help="select the best valid iterative objective or the final valid/free-run iterate",
    )
    parser.add_argument(
        "--candidate-search",
        choices=("fixed", "generated"),
        default="fixed",
        help="RHS candidate source for the G-J route; generated enables EqGPT-style structure proposals",
    )
    parser.add_argument(
        "--allowed-physical-terms",
        default=None,
        help="comma-separated physical RHS terms to keep in the G-J fixed library, e.g. Hx or Hx,D_x^beta H",
    )
    parser.add_argument("--generated-candidates", type=int, default=100, help="number of EqGPT RHS samples per optimization epoch")
    parser.add_argument("--generated-max-terms", type=int, default=5, help="maximum physical RHS terms per generated structure")
    parser.add_argument("--generated-top-structures", type=int, default=10, help="reward top-k generated structures retained per EqGPT epoch")
    parser.add_argument("--generated-seed", type=int, default=7, help="deterministic seed for EqGPT sampling")
    parser.add_argument("--eqgpt-dictionary", type=Path, default=None, help="EqGPT dict_datas_0725.json path")
    parser.add_argument("--eqgpt-model-checkpoint", type=Path, default=None, help="pretrained EqGPT checkpoint, e.g. PDEGPT_KdV_equation.pt")
    parser.add_argument("--eqgpt-optimize-epochs", type=int, default=5, help="EqGPT reward optimization cycles; 0 runs one sampling/evaluation pass without fine-tuning")
    parser.add_argument("--eqgpt-finetune-epochs", type=int, default=5, help="fine-tuning epochs on reward top-k structures")
    parser.add_argument("--eqgpt-learning-rate", type=float, default=1.0e-5, help="learning rate for EqGPT reward top-k fine-tuning")
    parser.add_argument("--eqgpt-reward-sparsity-alpha", type=float, default=0.2, help="EqGPT reward sparsity coefficient alpha_0")
    parser.add_argument(
        "--eqgpt-selection-mode",
        choices=("stridge_objective", "eqgpt_reward"),
        default="stridge_objective",
        help="select generated structures by the local augmented STRidge objective or by the original EqGPT reward",
    )
    parser.add_argument("--eqgpt-random-exploration", type=float, default=0.2, help="probability of uniform random valid-token sampling")
    parser.add_argument(
        "--order-radius-mode",
        choices=("off", "penalty", "strict"),
        default=PAPER_ORDER_RADIUS_MODE,
        help="handle first-order Taylor effective-radius violations",
    )
    parser.add_argument("--order-radius-tolerance", type=float, default=PAPER_ORDER_RADIUS_TOLERANCE)
    parser.add_argument("--beta-min", type=float, default=BETA_BOUNDS[0])
    parser.add_argument("--beta-max", type=float, default=BETA_BOUNDS[1])
    parser.add_argument(
        "--beta-reference-orders",
        default=None,
        help="comma-separated beta0 grid for spatial Taylor expansion; default is case-specific",
    )
    parser.add_argument("--spatial-fractional-lower-bound", type=float, default=SPATIAL_FRACTIONAL_LOWER_BOUND)
    parser.add_argument(
        "--spatial-boundary-mode",
        choices=("none", "raw", "left_constant_subtract", "left_caputo"),
        default=SPATIAL_BOUNDARY_MODE,
        help="boundary treatment before applying the left spatial fractional operator",
    )
    parser.add_argument(
        "--spatial-boundary-value",
        type=float,
        default=SPATIAL_BOUNDARY_VALUE,
        help="optional constant value at x=0; inferred from the NN field when omitted",
    )
    return parser.parse_args()


def _explicit_cli_option(option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in sys.argv[1:])


def _apply_case_overrides(
    args: argparse.Namespace,
    preset: dict,
) -> tuple[float, float, float, float, float, float, float | None, float | None, float | None, float | None, object]:
    """Return (x_min, x_max, x_step, t_min, t_max, t_step, fit_x_min, fit_x_max, fit_t_min, fit_t_max, checkpoint_file)
    after applying case-specific defaults that override the paper-wide defaults."""

    x_min = args.x_min
    x_max = args.x_max
    x_step = args.x_step
    t_min = args.t_min
    t_max = args.t_max
    t_step = args.t_step
    fit_x_min = args.fit_x_min
    fit_x_max = args.fit_x_max
    fit_t_min = args.fit_t_min
    fit_t_max = args.fit_t_max
    checkpoint_file = args.checkpoint

    if args.case == "tsfade_fft" and args.tsfade_route == "gj_hybrid_taylor":
        if args.space_derivative_mode == SPACE_DERIVATIVE_MODE:
            args.space_derivative_mode = "autodiff"
        if args.spatial_fractional_mode == SPATIAL_FRACTIONAL_MODE:
            args.spatial_fractional_mode = "gj_richardson"
        if args.time_operator_mode == TIME_OPERATOR_MODE:
            args.time_operator_mode = "laplace_taylor"
        if args.laguerre_nodes == PAPER_LAGUERRE_NODES:
            args.laguerre_nodes = TSFADE_GJ_QUADRATURE_POINTS

    if args.case == "analytic_tfade":
        x_min = 0.0 if np.isclose(args.x_min, PAPER_X_MIN) else args.x_min
        x_max = float(np.pi) if np.isclose(args.x_max, PAPER_X_MAX) else args.x_max
        x_step = (x_max - x_min) / int(args.nx) if np.isclose(args.x_step, PAPER_X_STEP) else args.x_step
        t_min = 0.0 if np.isclose(args.t_min, PAPER_T_MIN) else args.t_min
        t_max = 10.0 if np.isclose(args.t_max, PAPER_T_MAX) else args.t_max
        t_step = (t_max - t_min) / int(args.nt) if np.isclose(args.t_step, PAPER_T_STEP) else args.t_step
        fit_x_min = 0.1 if args.fit_x_min == PAPER_FIT_X_MIN else args.fit_x_min
        fit_x_max = float(np.pi - 0.1) if args.fit_x_max == PAPER_FIT_X_MAX else args.fit_x_max
        fit_t_min = None if args.fit_t_min == PAPER_FIT_T_MIN else args.fit_t_min
        fit_t_max = None if args.fit_t_max == PAPER_FIT_T_MAX else args.fit_t_max
        checkpoint_file = Path("analytic_tfade")

    if args.case == "periodic_tfade_fft":
        if args.nx == 200:
            args.nx = 256
        x_min = 0.0 if np.isclose(args.x_min, PAPER_X_MIN) else args.x_min
        x_max = float(np.pi) if np.isclose(args.x_max, PAPER_X_MAX) else args.x_max
        x_step = (x_max - x_min) / int(args.nx) if np.isclose(args.x_step, PAPER_X_STEP) else args.x_step
        t_min = 0.0 if np.isclose(args.t_min, PAPER_T_MIN) else args.t_min
        t_max = 5.0 if np.isclose(args.t_max, PAPER_T_MAX) else args.t_max
        t_step = (t_max - t_min) / int(args.nt) if np.isclose(args.t_step, PAPER_T_STEP) else args.t_step
        fit_x_min = None if args.fit_x_min == PAPER_FIT_X_MIN else args.fit_x_min
        fit_x_max = None if args.fit_x_max == PAPER_FIT_X_MAX else args.fit_x_max
        fit_t_min = None if args.fit_t_min == PAPER_FIT_T_MIN else args.fit_t_min
        fit_t_max = None if args.fit_t_max == PAPER_FIT_T_MAX else args.fit_t_max
        if args.delta_alpha_min == PAPER_DELTA_ALPHA_BOUNDS[0] and args.delta_alpha_max == PAPER_DELTA_ALPHA_BOUNDS[1]:
            args.delta_alpha_min = -0.15
            args.delta_alpha_max = 0.15
        if checkpoint_file is None:
            checkpoint_file = Path("periodic_tfade_fft")

    if args.case == "tsfade_fft":
        if "x_min" in preset and np.isclose(args.x_min, PAPER_X_MIN):
            x_min = float(preset["x_min"])
        if "x_max" in preset and np.isclose(args.x_max, PAPER_X_MAX):
            x_max = float(preset["x_max"])
        if "x_step" in preset and np.isclose(args.x_step, PAPER_X_STEP):
            x_step = float(preset["x_step"])
        if "t_min" in preset and np.isclose(args.t_min, PAPER_T_MIN):
            t_min = float(preset["t_min"])
        if "t_max" in preset and np.isclose(args.t_max, PAPER_T_MAX):
            t_max = float(preset["t_max"])
        if "t_step" in preset and np.isclose(args.t_step, PAPER_T_STEP):
            t_step = float(preset["t_step"])
        if (
            "spatial_fractional_lower_bound" in preset
            and np.isclose(args.spatial_fractional_lower_bound, SPATIAL_FRACTIONAL_LOWER_BOUND)
        ):
            args.spatial_fractional_lower_bound = float(preset["spatial_fractional_lower_bound"])
        if checkpoint_file is None and not args.legacy_checkpoint_layout and preset.get("checkpoint_file") is not None:
            checkpoint_file = Path(preset["checkpoint_file"])

    return x_min, x_max, x_step, t_min, t_max, t_step, fit_x_min, fit_x_max, fit_t_min, fit_t_max, checkpoint_file


def _attach_metadata_to_result(
    result: object,
    *,
    display_example_name: str,
    paper_example_name: object,
    preset_example_name: str,
    explicit_checkpoint: bool,
    display_description: str,
    true_alpha: object,
    true_beta: object,
    preset: dict,
    tsfade_route: str | None,
) -> None:
    """Stamp example/ground-truth metadata onto result.model and every alpha_scan item."""

    shared = {
        "example": display_example_name,
        "paper_example": paper_example_name,
        "preset_example": preset_example_name,
        "example_source": "checkpoint" if explicit_checkpoint else "preset",
        "example_description": display_description,
        "true_alpha": true_alpha,
        "true_beta": true_beta,
        "gamma_shape": None if explicit_checkpoint else preset.get("gamma_shape"),
        "gamma_scale": None if explicit_checkpoint else preset.get("gamma_scale"),
        "tsfade_route": tsfade_route,
    }
    mechanistic_note = (
        "gamma_shape is a conductivity-field parameter, not an analytical PDE ground truth"
    )
    result.model.metadata.update(shared)
    if not explicit_checkpoint and preset.get("gamma_shape") is not None:
        result.model.metadata["mechanistic_reference_note"] = mechanistic_note
    for item in result.alpha_scan:
        item.model.metadata.update(shared)
        if not explicit_checkpoint and preset.get("gamma_shape") is not None:
            item.model.metadata["mechanistic_reference_note"] = mechanistic_note


def run_paper_task(args: argparse.Namespace) -> int:
    """Dispatch manuscript batch tasks through the single public entry point."""

    script = PAPER_TASK_SCRIPTS[args.paper_task]
    if not script.exists():
        print(f"paper task support script is missing: {script}", file=sys.stderr)
        return 2

    existing_data = {
        "generate-analytic-tfade": ROOT / "data" / "analytic_tfade_sine" / "analytic_tfade_sine.mat",
        "generate-periodic-tfade": ROOT / "data" / "periodic_tfade_fft" / "periodic_tfade_fft.mat",
    }.get(args.paper_task)
    if existing_data is not None and existing_data.exists() and not args.force_data_regeneration:
        print(f"paper task: {args.paper_task}")
        print(f"data already exists: {existing_data}")
        print("use --force-data-regeneration to overwrite it")
        return 0

    command = [sys.executable, str(script)]
    if args.paper_task == "legacy-de":
        command.extend(
            [
                "--noise-levels",
                args.legacy_noise_levels,
                "--maxiter",
                str(args.legacy_maxiter),
                "--stridge-mode",
                args.legacy_stridge_mode,
                "--out-dir",
                str(args.legacy_out_dir),
            ]
        )
        if args.legacy_quiet:
            command.append("--quiet")

    print("paper task:", args.paper_task)
    print("command:", " ".join(f'"{item}"' if " " in item else item for item in command))
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


def run_paper_reproduction(args: argparse.Namespace) -> int:
    """Run a named manuscript reproduction group using this public entry point."""

    commands = paper_reproduction_commands(sys.executable, args.paper_reproduce)
    print(f"paper reproduction group: {args.paper_reproduce}")
    for index, command in enumerate(commands, start=1):
        display = " ".join(f'"{item}"' if " " in item else item for item in command)
        print(f"[{index}/{len(commands)}] {display}")
        if args.paper_reproduce_dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def main() -> None:
    args = parse_args()
    if args.list_paper_examples:
        print(format_paper_examples())
        return
    if args.paper_reproduce is not None:
        raise SystemExit(run_paper_reproduction(args))
    if args.paper_task is not None:
        raise SystemExit(run_paper_task(args))
    paper_example_name = args.paper_example
    paper_example = apply_paper_example_overrides(args, _explicit_cli_option)
    if not CASES[args.case].enabled:
        raise NotImplementedError(f"{args.case} is not part of the paper-only workflow.")
    example_name = args.case if args.case in {"analytic_tfade", "periodic_tfade_fft"} else args.example
    preset_example_name = example_name
    preset = preset_for_case(args.case, example_name)
    model_alpha_tag = args.model_alpha_tag or preset["alpha_tag"]
    activation = args.activation or preset["activation"]
    hidden_layers = args.hidden_layers if args.hidden_layers is not None else int(preset["hidden_layers"])
    neurons = args.neurons if args.neurons is not None else int(preset["neurons"])
    trained_point = args.trained_point if args.trained_point is not None else int(preset["trained_point"])
    noise_level = args.noise_level if args.noise_level is not None else float(preset["noise_level"])
    sparsity_lamb = args.lamb if args.lamb is not None else float(preset.get("sparsity_lamb", PAPER_FSTRIDGE_LAMB))
    d_tol = args.d_tol if args.d_tol is not None else float(preset.get("d_tol", PAPER_D_TOL))
    laplace_s_min = float(args.s_min if args.s_min is not None else preset["laplace_s_min"])
    laplace_s_max = float(args.s_max if args.s_max is not None else preset["laplace_s_max"])
    alpha_correction_tol = float(
        args.alpha_correction_tol
        if args.alpha_correction_tol is not None
        else preset["alpha_correction_tol"]
    )
    beta_correction_tol = float(
        args.beta_correction_tol
        if args.beta_correction_tol is not None
        else preset["beta_correction_tol"]
    )
    beta_reference_orders = (
        tuple(float(item) for item in args.beta_reference_orders.split(",") if item.strip())
        if args.beta_reference_orders is not None
        else tuple(float(value) for value in preset["beta_reference_orders"])
    )
    iter_start_beta_values = (
        tuple(float(item) for item in args.iter_start_beta_values.split(",") if item.strip())
        if args.iter_start_beta_values is not None
        else None
    )
    allowed_physical_terms = (
        tuple(item.strip() for item in args.allowed_physical_terms.split(",") if item.strip())
        if args.allowed_physical_terms is not None
        else None
    )
    enable_spatial_fractional = bool(preset.get("enable_spatial_fractional", ENABLE_SPATIAL_FRACTIONAL))
    if args.enable_spatial_fractional is True:
        enable_spatial_fractional = True
    if args.disable_spatial_fractional:
        enable_spatial_fractional = False
    config_case_name = args.case
    (
        x_min, x_max, x_step,
        t_min, t_max, t_step,
        fit_x_min, fit_x_max, fit_t_min, fit_t_max,
        checkpoint_file,
    ) = _apply_case_overrides(args, preset)

    model_file = build_model_file(
        config_case_name,
        model_alpha_tag,
        activation,
        hidden_layers,
        neurons,
        noise_level,
    )
    if checkpoint_file is None and not args.legacy_checkpoint_layout:
        checkpoint_file = DEFAULT_MODEL_CHECKPOINT
    display_example_name = example_name
    display_description = preset.get("description", "")
    true_alpha = preset.get("true_alpha")
    true_beta = preset.get("true_beta")
    explicit_checkpoint = args.checkpoint is not None and not args.legacy_checkpoint_layout
    if explicit_checkpoint and checkpoint_file is not None:
        display_example_name = _checkpoint_derived_example_name(checkpoint_file)
        inferred_alpha, inferred_beta = _truth_from_checkpoint_label(display_example_name)
        true_alpha = inferred_alpha
        true_beta = inferred_beta
        display_description = (
            f"explicit checkpoint-derived run; preset defaults came from {preset_example_name}, "
            "but display name and ground-truth metadata are not inherited from that preset"
        )
    config = make_config(
        case_name=config_case_name,
        checkpoint_file=None if args.legacy_checkpoint_layout else checkpoint_file,
        model_root=args.model_root,
        model_alpha_tag=model_alpha_tag,
        trained_point=trained_point,
        noise_level=noise_level,
        activation=activation,
        hidden_layers=hidden_layers,
        neurons=neurons,
        checkpoint_iteration=args.checkpoint_iteration,
        sparsity_lamb=sparsity_lamb,
        fractional_correction_sparsity_lamb=args.fractional_correction_lamb,
        fractional_correction_tol_scale=args.fractional_correction_tol_scale,
        alpha_correction_tol=alpha_correction_tol,
        beta_correction_tol=beta_correction_tol,
        d_tol=d_tol,
        delta_alpha_prune_threshold=args.delta_alpha_threshold,
        delta_alpha_bounds=(args.delta_alpha_min, args.delta_alpha_max),
        delta_beta_prune_threshold=args.delta_beta_threshold,
        x_min=x_min,
        x_max=x_max,
        x_step=x_step,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        fit_x_min=fit_x_min,
        fit_x_max=fit_x_max,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        laplace_s_min=laplace_s_min,
        laplace_s_max=laplace_s_max,
        laplace_s_points=args.s_points,
        laguerre_nodes=args.laguerre_nodes,
        enable_spatial_fractional=enable_spatial_fractional,
        beta_bounds=(args.beta_min, args.beta_max),
        beta_reference_orders=beta_reference_orders,
        spatial_log_quad_points=args.spatial_log_quad_points,
        spatial_correction_method=args.spatial_fractional_mode,
        spatial_fractional_mode=args.spatial_fractional_mode,
        space_derivative_mode=args.space_derivative_mode,
        time_operator_mode=args.time_operator_mode,
        spatial_fractional_lower_bound=args.spatial_fractional_lower_bound,
        spatial_boundary_mode=args.spatial_boundary_mode,
        spatial_boundary_value=args.spatial_boundary_value,
        selection_objective=args.selection_objective,
        order_radius_mode=args.order_radius_mode,
        order_radius_tolerance=args.order_radius_tolerance,
        refit_mode=args.refit_mode,
        order_update_mode=args.order_update_mode,
        allowed_physical_terms=allowed_physical_terms,
        iter_start_alpha=args.iter_start_alpha,
        iter_start_beta=args.iter_start_beta,
        iter_start_beta_values=iter_start_beta_values,
        iter_max_iters=args.iter_max_iters,
        iter_order_tol=args.iter_order_tol,
        iter_damping=args.iter_damping,
        iter_max_step_alpha=args.iter_max_step_alpha,
        iter_max_step_beta=args.iter_max_step_beta,
        iter_delta_alpha_bounds=(args.iter_delta_alpha_min, args.iter_delta_alpha_max),
        iter_delta_beta_bounds=(args.iter_delta_beta_min, args.iter_delta_beta_max),
        iter_stop_on_non_decreasing_objective=(
            PAPER_ITER_STOP_ON_NON_DECREASING_OBJECTIVE and not args.iter_allow_nondecreasing_objective
        ),
        iter_objective_min_delta=args.iter_objective_min_delta,
        iter_selection_mode=args.iter_selection_mode,
        candidate_search=args.candidate_search,
        generated_candidates=args.generated_candidates,
        generated_max_terms=args.generated_max_terms,
        generated_top_structures=args.generated_top_structures,
        generated_seed=args.generated_seed,
        eqgpt_dictionary_path=args.eqgpt_dictionary,
        eqgpt_model_checkpoint=args.eqgpt_model_checkpoint,
        eqgpt_optimize_epochs=args.eqgpt_optimize_epochs,
        eqgpt_finetune_epochs=args.eqgpt_finetune_epochs,
        eqgpt_learning_rate=args.eqgpt_learning_rate,
        eqgpt_reward_sparsity_alpha=args.eqgpt_reward_sparsity_alpha,
        eqgpt_selection_mode=args.eqgpt_selection_mode,
        eqgpt_random_exploration=args.eqgpt_random_exploration,
    )
    if args.case == "tsfade_fft" and args.tsfade_route == "gj_hybrid_taylor":
        result = GJHybridDiscoverer(config).discover()
    else:
        result = FractionalPDEDiscoverer(config).discover()
    route_label = args.tsfade_route if args.case == "tsfade_fft" else args.time_operator_mode
    _attach_metadata_to_result(
        result,
        display_example_name=display_example_name,
        paper_example_name=paper_example_name,
        preset_example_name=preset_example_name,
        explicit_checkpoint=explicit_checkpoint,
        display_description=display_description,
        true_alpha=true_alpha,
        true_beta=true_beta,
        preset=preset,
        tsfade_route=args.tsfade_route if args.case == "tsfade_fft" else None,
    )
    detailed_text = format_detailed_result(result) + "\n"
    out_path: Path | None = None
    for candidate in (
        ROOT / "results" / f"{args.case}_{display_example_name}_{route_label}_result.txt",
        ROOT / "data" / "results" / f"{args.case}_{display_example_name}_{route_label}_result.txt",
        SRC / "results" / f"{args.case}_{display_example_name}_{route_label}_result.txt",
        ROOT / f"{args.case}_{display_example_name}_{route_label}_result.txt",
        Path(tempfile.gettempdir()) / "linear_fractional_results" / f"{args.case}_{display_example_name}_{route_label}_result.txt",
    ):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(detailed_text, encoding="utf-8")
            out_path = candidate
            break
        except PermissionError:
            continue

    print(format_console_result(result))
    if out_path is None:
        print("detailed report: skipped (permission denied)")
    else:
        print(f"detailed report: {out_path}")


if __name__ == "__main__":
    main()
