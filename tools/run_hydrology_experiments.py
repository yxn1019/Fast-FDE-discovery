"""Run hydrology NN surrogate training and equation discovery experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data" / "hydrology_experiments" / "hydrology_preparation_summary.json"


def _load_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing hydrology preparation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def _selected(items: list[dict[str, Any]], selector: str) -> list[dict[str, Any]]:
    if selector == "all":
        return items
    if ":" in selector:
        experiment, variant = selector.split(":", 1)
        return [item for item in items if item["experiment"] == experiment and item["variant"] == variant]
    return [item for item in items if item["experiment"] == selector]


def _run(command: list[str], *, dry_run: bool) -> tuple[int, str]:
    display = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(display, flush=True)
    if dry_run:
        return 0, ""
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, end="", flush=True)
    return int(completed.returncode), completed.stdout


def _report_path(stdout: str) -> str | None:
    match = re.search(r"^detailed report:\s*(.+)$", stdout, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _model_dir(item: dict[str, Any], *, hidden_layers: int, neurons: int, loss: str) -> Path:
    label = f"{item['experiment']}_{item['variant']}_tanh_{hidden_layers}x{neurons}_{loss}"
    return ROOT / "data" / "models" / "hydrology_experiments" / label


def _train_command(
    item: dict[str, Any],
    *,
    output_dir: Path,
    max_steps: int,
    hidden_layers: int,
    neurons: int,
    loss: str,
) -> list[str]:
    return [
        sys.executable,
        "src/train_nn_reconstruction.py",
        "--data-file",
        str(item["train_mat"]),
        "--output-dir",
        str(output_dir),
        "--train-points",
        str(int(item["train_points"])),
        "--val-points",
        str(int(item["val_points"])),
        "--max-steps",
        str(int(max_steps)),
        "--eval-every",
        "100",
        "--print-every",
        "1000",
        "--activation",
        "tanh",
        "--hidden-layers",
        str(int(hidden_layers)),
        "--neurons",
        str(int(neurons)),
        "--input-normalization",
        "unit_box",
        "--output-normalization",
        "unit_interval",
        "--data-loss",
        loss,
        "--allow-large-train-set",
    ]


def _discovery_command(item: dict[str, Any], *, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--case",
        "tsfade_fft",
        "--example",
        "tsfade_retrained_alpha078_beta183_noise5",
        "--checkpoint",
        str(checkpoint),
        "--x-min",
        str(float(item["x_min"])),
        "--x-max",
        str(float(item["discovery_x_max_exclusive"])),
        "--x-step",
        str(float(item["x_step"])),
        "--t-min",
        str(float(item["discovery_t_min"])),
        "--t-max",
        str(float(item["discovery_t_max_exclusive"])),
        "--t-step",
        str(float(item["discovery_t_step"])),
        "--fit-x-min",
        str(float(item["fit_x_min"])),
        "--fit-x-max",
        str(float(item["fit_x_max"])),
        "--fit-t-min",
        str(float(item["fit_t_min"])),
        "--fit-t-max",
        str(float(item["fit_t_max"])),
        "--s-min",
        str(float(item["laplace_s_min"])),
        "--s-max",
        str(float(item["laplace_s_max"])),
        "--s-points",
        "80",
        "--laguerre-nodes",
        "5",
        "--order-update-mode",
        "iterative",
        "--iter-start-beta-values",
        "2.0,1.8,1.7",
        "--selection-objective",
        "physical-stridge",
        "--refit-mode",
        "none",
        "--lamb",
        "0.0001",
        "--spatial-fractional-lower-bound",
        "0.0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--select",
        default="all",
        help="all, made2, north_loup, or experiment:variant such as made2:massnorm",
    )
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--neurons", type=int, default=50)
    parser.add_argument("--data-loss", choices=("mse", "log_mse", "hybrid_log_mse"), default="hybrid_log_mse")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = _selected(_load_summaries(args.summary), args.select)
    if not items:
        raise SystemExit(f"No hydrology variants matched --select {args.select!r}")

    run_summaries: list[dict[str, Any]] = []
    for item in items:
        output_dir = _model_dir(
            item,
            hidden_layers=args.hidden_layers,
            neurons=args.neurons,
            loss=args.data_loss,
        )
        checkpoint = output_dir / "best.pkl"
        row: dict[str, Any] = {
            "experiment": item["experiment"],
            "variant": item["variant"],
            "model_dir": str(output_dir),
            "checkpoint": str(checkpoint),
            "train_returncode": None,
            "discovery_returncode": None,
            "report": None,
        }
        if not args.discover_only:
            code, _stdout = _run(
                _train_command(
                    item,
                    output_dir=output_dir,
                    max_steps=args.max_steps,
                    hidden_layers=args.hidden_layers,
                    neurons=args.neurons,
                    loss=args.data_loss,
                ),
                dry_run=args.dry_run,
            )
            row["train_returncode"] = code
            if code != 0:
                run_summaries.append(row)
                continue
        if not args.train_only:
            code, stdout = _run(_discovery_command(item, checkpoint=checkpoint), dry_run=args.dry_run)
            row["discovery_returncode"] = code
            row["report"] = _report_path(stdout)
        run_summaries.append(row)

    safe_loss = str(args.data_loss).replace("/", "_")
    output = ROOT / "results" / f"hydrology_experiments_run_summary_{safe_loss}.json"
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
        print(f"run summary: {output}")


if __name__ == "__main__":
    main()
