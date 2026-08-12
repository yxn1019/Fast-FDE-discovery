"""Revision diagnostics batch (run after the main regen chain completes).

1. Radius sensitivity sweep (Reviewer #1 comment 2): rerun the three tsfade
   cases with different (delta_alpha, delta_beta) caps.
2. Column-normalization check (Reviewer #2 comment 4): rerun with STRidge
   column normalization enabled (unit l2 norms, coefficients mapped back)
   and confirm supports/orders are unchanged.
3. Matrix conditioning diagnostics (Reviewer #2 comment 1) are handled by
   tools/run_matrix_conditioning_diagnostics.py.

Outputs: results/revision_diagnostics/*.csv
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_DIR = ROOT / "results" / "revision_diagnostics"
CASES = ["tsfade_clean", "tsfade_noise5", "tsfade_noise25"]

RADIUS_VALUES = {
    "delta_beta": [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    "delta_alpha": [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
}
PRESCRIBED_RADII = {"delta_alpha": 0.25, "delta_beta": 0.15}
PRESET_SPARSITY = {
    "tsfade_clean": 3.0e-6,
    "tsfade_noise5": 2.5e-5,
    "tsfade_noise25": 1.0e-4,
}


def radius_sweep_specs() -> list[dict]:
    specs = []
    for case in CASES:
        for radius_swept, values in RADIUS_VALUES.items():
            for value in values:
                da = value if radius_swept == "delta_alpha" else PRESCRIBED_RADII["delta_alpha"]
                db = value if radius_swept == "delta_beta" else PRESCRIBED_RADII["delta_beta"]
                sparsity_lamb = PRESET_SPARSITY[case]
                specs.append({
                    "case": case,
                    "radius_swept": radius_swept,
                    "value": value,
                    "delta_alpha_cap": da,
                    "delta_beta_cap": db,
                    "sparsity_lamb": sparsity_lamb,
                })
    return specs


def run_case(case: str, extra_args: list[str], env_extra: dict[str, str] | None = None) -> dict:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [PY, "main.py", "--paper-example", case, "--refit-mode", "none",
         "--iter-selection-mode", "best_objective", *extra_args],
        cwd=str(ROOT), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out = proc.stdout or ""
    eq = re.search(r"^candidate equation:\s*(.+)$", out, re.MULTILINE)
    orders = re.search(r"^alpha = ([0-9.eE+-]+), beta = ([0-9.eE+-]+)", out, re.MULTILINE)
    return {
        "returncode": proc.returncode,
        "equation": eq.group(1).strip() if eq else None,
        "alpha": float(orders.group(1)) if orders else None,
        "beta": float(orders.group(2)) if orders else None,
        "tail": out[-1500:] if proc.returncode != 0 else "",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    radius_rows = []
    for spec in radius_sweep_specs():
        da = spec["delta_alpha_cap"]
        db = spec["delta_beta_cap"]
        res = run_case(spec["case"], [
            "--lamb", str(spec["sparsity_lamb"]),
            "--iter-delta-alpha-min", str(-da), "--iter-delta-alpha-max", str(da),
            "--iter-delta-beta-min", str(-db), "--iter-delta-beta-max", str(db),
            "--iter-max-step-alpha", str(da), "--iter-max-step-beta", str(db),
        ])
        row = {**spec, **res}
        row.pop("tail", None)
        radius_rows.append(row)
        print(
            f"[radius] {spec['case']} {spec['radius_swept']}={spec['value']} "
            f"-> {res['equation']}"
        )
    with open(OUT_DIR / "radius_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(radius_rows[0].keys()))
        writer.writeheader()
        writer.writerows(radius_rows)

    norm_rows = []
    for case in CASES:
        for norm in ("0", "2"):
            res = run_case(case, [], env_extra={"GJ_STRIDGE_NORMALIZE": norm})
            row = {"case": case, "stridge_normalize": norm, **res}
            row.pop("tail", None)
            norm_rows.append(row)
            print(f"[normalize] {case} normalize={norm} -> {res['equation']}")
    with open(OUT_DIR / "normalization_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(norm_rows)

    print("diagnostics complete")


if __name__ == "__main__":
    main()
