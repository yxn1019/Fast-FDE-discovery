"""Collinearity / conditioning diagnostics of the augmented regression matrix.

CMAME revision, Reviewer #2 comment 1: report pairwise column correlations,
singular values, and condition numbers of the normalized augmented matrix
over the iterative order updates.

Runs the three tsfade mainline cases with GJ_MATRIX_DIAG_PATH enabled, then
summarizes the per-iteration records into a CSV and a console table.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_DIR = ROOT / "results" / "matrix_conditioning"
CASES = ["tsfade_clean", "tsfade_noise5", "tsfade_noise25"]

def _is_reference(name: str) -> bool:
    return name.startswith("D_x^") and name.endswith(" H")


def _is_alpha_correction(name: str) -> bool:
    return name == "alpha_correction"


def _is_spatial_correction(name: str) -> bool:
    return name.startswith("spatial_correction")


def _is_physical(name: str) -> bool:
    return not _is_alpha_correction(name) and not _is_spatial_correction(name)


def _maximum_pair(corr: dict[str, float], predicate) -> tuple[float | None, str | None]:
    matches = [(abs(value), key) for key, value in corr.items() if predicate(*key.split("|", 1))]
    return max(matches) if matches else (None, None)


def summarize(jsonl_path: Path, case: str, rows: list[dict]) -> None:
    with open(jsonl_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for idx, rec in enumerate(records):
        corr = rec["pairwise_correlations"]
        ref_corr = [v for k, v in corr.items() if k.count("D_x^") == 1 and "spatial_correction" in k]
        adv_corr = [
            v for k, v in corr.items()
            if "D_x^" in k and ("Hx|" in k or "|Hx" in k) and "Hxx" not in k and "H*Hx" not in k and "H^2*Hx" not in k
        ]
        alpha_corr = [v for k, v in corr.items() if "alpha_correction" in k and "D_x^" in k and "spatial" not in k]
        max_ref_physical, pair_ref_physical = _maximum_pair(
            corr,
            lambda a, b: (
                (_is_reference(a) and _is_physical(b) and not _is_reference(b))
                or (_is_reference(b) and _is_physical(a) and not _is_reference(a))
            ),
        )
        max_spatial_corr_physical, pair_spatial_corr_physical = _maximum_pair(
            corr,
            lambda a, b: (
                (_is_spatial_correction(a) and _is_physical(b) and not _is_reference(b))
                or (_is_spatial_correction(b) and _is_physical(a) and not _is_reference(a))
            ),
        )
        max_alpha_corr_physical, pair_alpha_corr_physical = _maximum_pair(
            corr,
            lambda a, b: (
                (_is_alpha_correction(a) and _is_physical(b))
                or (_is_alpha_correction(b) and _is_physical(a))
            ),
        )
        max_physical, pair_physical = _maximum_pair(
            corr, lambda a, b: _is_physical(a) and _is_physical(b)
        )
        max_all, pair_all = _maximum_pair(corr, lambda _a, _b: True)
        sv = rec["singular_values"]
        rows.append({
            "case": case,
            "solve_index": idx,
            "alpha0": rec["alpha0"],
            "beta0": rec["beta0"],
            "corr_ref_vs_lin": ref_corr[0] if ref_corr else None,
            "corr_ref_vs_Hx": adv_corr[0] if adv_corr else None,
            "corr_ref_vs_alpha_corr": alpha_corr[0] if alpha_corr else None,
            "max_abs_corr_ref_vs_other_physical": max_ref_physical,
            "pair_ref_vs_other_physical": pair_ref_physical,
            "max_abs_corr_spatial_corr_vs_physical": max_spatial_corr_physical,
            "pair_spatial_corr_vs_physical": pair_spatial_corr_physical,
            "max_abs_corr_alpha_corr_vs_physical": max_alpha_corr_physical,
            "pair_alpha_corr_vs_physical": pair_alpha_corr_physical,
            "max_abs_corr_physical_columns": max_physical,
            "pair_physical_columns": pair_physical,
            "max_abs_corr_all_columns": max_all,
            "pair_all_columns": pair_all,
            "sigma_max": sv[0],
            "sigma_min": sv[-1],
            "condition_normalized": rec["condition_normalized"],
        })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        jsonl_path = OUT_DIR / f"matrix_diag_{case}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        env = dict(os.environ)
        env["GJ_MATRIX_DIAG_PATH"] = str(jsonl_path)
        env["GJ_MATRIX_DIAG_ALL_PAIRS"] = "1"
        proc = subprocess.run(
            [PY, "main.py", "--paper-example", case, "--refit-mode", "none",
             "--iter-selection-mode", "best_objective"],
            cwd=str(ROOT), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        print(f"[{case}] rc={proc.returncode}")
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            continue
        summarize(jsonl_path, case, rows)

    csv_path = OUT_DIR / "matrix_conditioning_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} augmented solves)")

    for case in CASES:
        case_rows = [r for r in rows if r["case"] == case]
        if not case_rows:
            continue
        ref_lin = [abs(r["corr_ref_vs_lin"]) for r in case_rows if r["corr_ref_vs_lin"] is not None]
        conds = [r["condition_normalized"] for r in case_rows]
        print(
            f"{case}: solves={len(case_rows)}  |corr(ref,lin)| range="
            f"[{min(ref_lin):.3f}, {max(ref_lin):.3f}]  cond(normalized) range="
            f"[{min(conds):.3e}, {max(conds):.3e}]"
        )


if __name__ == "__main__":
    main()
