"""Measure the cost of every stage for the revision timing question (R1-6, R2-5).

Reports, for each noise level of the space-time-fractional benchmark:

  * surrogate training wall time (one-off, shared by both methods, excluded from
    every runtime reported in the manuscript);
  * surrogate load, operator-field construction, and order search for the
    proposed method;
  * the differential-evolution baseline runtime, whose timer starts only after
    the operator field is built.

Run on an otherwise idle machine: every number here is a wall-clock timing.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_DIR = ROOT / "results" / "timing_breakdown"

CASES = [
    ("clean", "tsfade_clean", "tsfade_alpha078_beta183_noise0.mat"),
    ("5%", "tsfade_noise5", "tsfade_alpha078_beta183_noise5.mat"),
    ("25%", "tsfade_noise25", "tsfade_alpha078_beta183_noise25.mat"),
]

TRAIN_COMMON = [
    "--case", "tsfade_fft", "--train-points", "2000", "--val-points", "2000",
    "--max-steps", "120000", "--noise-level", "0", "--noise-type", "none",
    "--activation", "tanh", "--hidden-layers", "8", "--neurons", "20",
    "--learning-rate", "0.001", "--weight-decay", "0.0001", "--seed", "525",
    "--eval-every", "200", "--print-every", "1000000",
    "--patience-steps", "30000", "--min-delta-rel", "0.0001",
    "--input-normalization", "unit_box", "--output-normalization", "unit_interval",
    "--selection-metric", "val",
]


def value(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_dir = ROOT / "data" / "tsfade_retrained_alpha078_beta183" / "raw_data"

    for label, example, mat in CASES:
        row: dict[str, object] = {"noise": label}

        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            proc = subprocess.run(
                [PY, str(ROOT / "src" / "train_nn_reconstruction.py"),
                 "--data-file", str(raw_dir / mat), "--output-dir", tmp, *TRAIN_COMMON],
                cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            row["surrogate_training_seconds"] = round(time.perf_counter() - started, 1)
            m = re.search(r"best_step=(\d+)", proc.stdout or "")
            row["surrogate_best_step"] = int(m.group(1)) if m else None
            row["surrogate_rc"] = proc.returncode

        proc = subprocess.run(
            [PY, "main.py", "--paper-example", example, "--refit-mode", "none",
             "--iter-selection-mode", "best_objective"],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out = proc.stdout or ""
        report = re.search(r"^detailed report:\s*(.+)$", out, re.MULTILINE)
        if report:
            text = Path(report.group(1).strip()).read_text(encoding="utf-8")
            for key, col in (
                ("end_to_end_seconds", "proposed_total_seconds"),
                ("timing_surrogate_load_seconds", "proposed_surrogate_load_seconds"),
                ("timing_operator_field_seconds", "proposed_operator_field_seconds"),
                ("timing_order_search_seconds", "proposed_order_search_seconds"),
            ):
                v = value(text, key)
                row[col] = round(float(v), 3) if v not in (None, "None") else None
        rows.append(row)
        print(f"[{label}] training={row['surrogate_training_seconds']}s "
              f"discovery={row.get('proposed_total_seconds')}s "
              f"(load {row.get('proposed_surrogate_load_seconds')}s + "
              f"field {row.get('proposed_operator_field_seconds')}s + "
              f"orders {row.get('proposed_order_search_seconds')}s)")

    csv_path = OUT_DIR / "timing_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "timing_breakdown.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
