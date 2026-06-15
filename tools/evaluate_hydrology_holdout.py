"""Evaluate hydrology surrogate profile quality on the held-out final snapshot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from plot_raw_vs_nn_surrogate import reconstruct_nn_on_raw_grid


ROOT = Path(__file__).resolve().parents[1]


def _normalize_mass(values: np.ndarray, dx: float) -> tuple[np.ndarray, float]:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    mass = float(np.sum(clipped) * dx)
    if mass <= 0.0 or not np.isfinite(mass):
        return np.zeros_like(clipped), mass
    return clipped / mass, mass


def _profile_metrics(x: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    dx = float(np.median(np.diff(x)))
    obs_norm, obs_mass = _normalize_mass(observed, dx)
    pred_norm, pred_mass = _normalize_mass(predicted, dx)
    denom = max(float(np.linalg.norm(obs_norm)), np.finfo(float).eps)
    peak = float(np.max(obs_norm)) if obs_norm.size else 0.0
    positive = obs_norm > 0.0
    support = obs_norm > 0.02 * max(peak, np.finfo(float).eps)
    eps = 1.0e-3 * max(peak, np.finfo(float).eps)
    if np.any(support):
        log_profile_mse = float(
            np.mean((np.log10(pred_norm[support] + eps) - np.log10(obs_norm[support] + eps)) ** 2)
        )
    else:
        log_profile_mse = float("nan")
    return {
        "observed_mass": obs_mass,
        "predicted_mass": pred_mass,
        "mass_error": pred_mass - obs_mass,
        "mass_normalized_relative_l2": float(np.linalg.norm(pred_norm - obs_norm) / denom),
        "positive_support_relative_l2": float(
            np.linalg.norm(pred_norm[positive] - obs_norm[positive])
            / max(float(np.linalg.norm(obs_norm[positive])), np.finfo(float).eps)
        )
        if np.any(positive)
        else float("nan"),
        "log_profile_mse_2pct": log_profile_mse,
        "peak_location_error": float(abs(x[int(np.argmax(pred_norm))] - x[int(np.argmax(obs_norm))])),
        "observed_peak_x": float(x[int(np.argmax(obs_norm))]),
        "predicted_peak_x": float(x[int(np.argmax(pred_norm))]),
    }


def _report_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    equation = re.search(r"^candidate equation:\s*(.+)$", text, flags=re.MULTILINE)
    alpha = re.search(r"^alpha:\s*(.+)$", text, flags=re.MULTILINE)
    return {
        "report": str(path),
        "candidate_equation": equation.group(1).strip() if equation else None,
        "alpha": alpha.group(1).strip() if alpha else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.full_npz, allow_pickle=True)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    c = np.asarray(data["c"], dtype=float)
    holdout_index = int(np.asarray(data["holdout_index"]).reshape(()))
    metadata = json.loads(str(np.asarray(data["metadata"]).reshape(())))
    prediction = reconstruct_nn_on_raw_grid(args.checkpoint, x, t)
    metrics = _profile_metrics(x, c[holdout_index, :], prediction[holdout_index, :])
    payload: dict[str, Any] = {
        "diagnostic": "surrogate_holdout_profile",
        "note": (
            "This compares the trained differentiable surrogate against the held-out final snapshot. "
            "A discovered-equation forecast is reported separately only when the selected equation has "
            "a supported closed-form profile predictor."
        ),
        "experiment": metadata.get("experiment"),
        "variant": metadata.get("variant"),
        "holdout_time": float(t[holdout_index]),
        "time_unit": metadata.get("time_unit"),
        "full_npz": str(args.full_npz),
        "checkpoint": str(args.checkpoint),
        **_report_summary(args.report),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        args.output.write_text("\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n", encoding="utf-8")
    print(f"holdout metrics: {args.output}")
    for key in ("mass_normalized_relative_l2", "positive_support_relative_l2", "log_profile_mse_2pct", "peak_location_error"):
        print(f"{key}: {payload[key]}")


if __name__ == "__main__":
    main()
