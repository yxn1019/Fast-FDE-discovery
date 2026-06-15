"""Compare bounded and fully unbounded iterative order updates on tsfade."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUT_DIR = ROOT / "results" / "iterative_order_update"
CSV_PATH = OUT_DIR / "iter_radius_ablation_tsfade_normalized_tanh.csv"
JSON_PATH = OUT_DIR / "iter_radius_ablation_tsfade_normalized_tanh.json"

TSFADE_CASES = (
    ("clean", "tsfade_clean"),
    ("5%", "tsfade_noise5"),
    ("25%", "tsfade_noise25"),
)


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "nan", "inf", "-inf"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _report_path(stdout: str) -> Path | None:
    match = re.search(r"^detailed report:\s*(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    path_text = match.group(1).strip()
    if path_text == "skipped (permission denied)":
        return None
    return Path(path_text)


def _active_terms(text: str) -> dict[str, float]:
    lines = text.splitlines()
    try:
        start = lines.index("active terms:") + 1
    except ValueError:
        return {}
    terms: dict[str, float] = {}
    for line in lines[start:]:
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if stripped == "none" or ":" not in stripped:
            break
        name, value = stripped.rsplit(":", 1)
        number = _as_float(value)
        if number is not None:
            terms[name.strip()] = number
    return terms


def _iteration_trace(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    try:
        start = lines.index("iterative order update trace:")
    except ValueError:
        return []
    if start + 1 >= len(lines):
        return []
    header = [item.strip() for item in lines[start + 1].strip().split(",")]
    rows: list[dict[str, str]] = []
    for line in lines[start + 2:]:
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        parts = [item.strip() for item in stripped.split(",", maxsplit=len(header) - 1)]
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    return rows


def _max_abs(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    finite = [abs(value) for value in values if value is not None]
    return max(finite) if finite else None


def _support_check(support: list[str], beta: float | None) -> tuple[bool, str]:
    fractional = [name for name in support if name.startswith("D_x^")]
    if len(support) == 2 and "Hx" in support and len(fractional) == 1 and beta is not None and 1.0 < beta <= 2.0:
        return True, "support is exactly Hx plus one spatial fractional term"
    return False, f"expected Hx + one D_x^beta H term, got {support}"


def _relative_error(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None or abs(truth) < 1.0e-14:
        return None
    return abs(value - truth) / abs(truth)


def _run_variant(noise_label: str, paper_example: str, variant: str) -> dict[str, Any]:
    command = [
        str(PYTHON),
        "main.py",
        "--paper-example",
        paper_example,
        "--refit-mode",
        "none",
    ]
    if variant == "bounded":
        command.extend(["--iter-selection-mode", "best_objective"])
    elif variant == "unbounded":
        command.extend(
            [
                "--iter-selection-mode",
                "final",
                "--iter-allow-nondecreasing-objective",
                "--iter-delta-alpha-min",
                "-10",
                "--iter-delta-alpha-max",
                "10",
                "--iter-delta-beta-min",
                "-10",
                "--iter-delta-beta-max",
                "10",
                "--iter-max-step-alpha",
                "10",
                "--iter-max-step-beta",
                "10",
                "--iter-order-tol",
                "0",
            ]
        )
    else:
        raise ValueError(f"unknown variant {variant}")

    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = str(Path(tempfile.gettempdir()) / "linear_fractional_pycache")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report = _report_path(completed.stdout)
    report_text = report.read_text(encoding="utf-8") if report is not None and report.exists() else ""
    trace = _iteration_trace(report_text)
    terms = _active_terms(report_text)
    support = list(terms)
    alpha = _as_float(_value(report_text, "alpha"))
    beta = _as_float(_value(report_text, "spatial_beta"))
    true_alpha = _as_float(_value(report_text, "true_alpha"))
    true_beta = _as_float(_value(report_text, "true_beta"))
    support_correct, reason = _support_check(support, beta)
    if completed.returncode != 0:
        support_correct = False
        reason = f"command failed with return code {completed.returncode}"
    coef_hx = terms.get("Hx")
    coef_dbeta = next((value for name, value in terms.items() if name.startswith("D_x^")), None)
    first = trace[0] if trace else None
    last = trace[-1] if trace else None
    return {
        "noise": noise_label,
        "paper_example": paper_example,
        "variant": variant,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
        "equation": _value(report_text, "candidate equation"),
        "support_correct": support_correct,
        "support_reason": reason,
        "support": support,
        "coefficients": terms,
        "alpha": alpha,
        "beta": beta,
        "coef_hx": coef_hx,
        "coef_dbeta": coef_dbeta,
        "true_alpha": true_alpha,
        "true_beta": true_beta,
        "alpha_relative_error": _relative_error(alpha, true_alpha),
        "beta_relative_error": _relative_error(beta, true_beta),
        "coef_hx_relative_error": _relative_error(coef_hx, -1.0),
        "coef_dbeta_relative_error": _relative_error(coef_dbeta, 0.5),
        "refit_mode": _value(report_text, "refit_mode"),
        "physical_refit": _as_bool(_value(report_text, "physical_refit")),
        "order_update_mode": _value(report_text, "order_update_mode"),
        "iter_selection_mode": _value(report_text, "iter_selection_mode"),
        "iteration_count": _as_float(_value(report_text, "iteration_count")),
        "iteration_converged": _as_bool(_value(report_text, "iteration_converged")),
        "iteration_stop_reason": _value(report_text, "iteration_stop_reason"),
        "iteration_final_index": _as_float(_value(report_text, "iteration_final_index")),
        "iteration_best_objective_index": _as_float(_value(report_text, "iteration_best_objective_index")),
        "iteration_selected_index": _as_float(_value(report_text, "iteration_selected_index")),
        "selected_by": _value(report_text, "selected_by"),
        "mse": _as_float(_value(report_text, "mse")),
        "objective": _as_float(_value(report_text, "objective")),
        "sparsity_lamb": _as_float(_value(report_text, "sparsity_lamb")),
        "d_tol": _as_float(_value(report_text, "d_tol")),
        "fit_window_config": _value(report_text, "fit_window_config"),
        "max_abs_raw_delta_alpha": _max_abs(trace, "delta_alpha_raw"),
        "max_abs_raw_delta_beta": _max_abs(trace, "delta_beta_raw"),
        "max_abs_used_delta_alpha": _max_abs(trace, "delta_alpha"),
        "max_abs_used_delta_beta": _max_abs(trace, "delta_beta_used"),
        "first_trace_row": first,
        "last_trace_row": last,
        "trace": trace,
        "runtime_seconds": _as_float(_value(report_text, "end_to_end_seconds")),
        "report_path": str(report) if report is not None else None,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-8:]),
    }


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for key in ("support", "coefficients", "first_trace_row", "last_trace_row", "trace"):
            row[key] = json.dumps(record[key], sort_keys=True, ensure_ascii=False)
        rows.append(row)
    return rows


def _select_output_dir() -> None:
    global OUT_DIR, CSV_PATH, JSON_PATH
    for candidate in (
        ROOT / "results" / "iterative_order_update",
        Path(tempfile.gettempdir()) / "linear_fractional_results" / "iterative_order_update",
    ):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            OUT_DIR = candidate
            CSV_PATH = OUT_DIR / "iter_radius_ablation_tsfade_normalized_tanh.csv"
            JSON_PATH = OUT_DIR / "iter_radius_ablation_tsfade_normalized_tanh.json"
            return
        except PermissionError:
            continue
    raise PermissionError("No writable output directory found for iterative radius ablation.")


def main() -> int:
    _select_output_dir()
    records: list[dict[str, Any]] = []
    for noise_label, paper_example in TSFADE_CASES:
        for variant in ("bounded", "unbounded"):
            print(f"running {noise_label} {variant}", flush=True)
            record = _run_variant(noise_label, paper_example, variant)
            records.append(record)
            print(
                f"  {record['status']} support_correct={record['support_correct']} "
                f"selected={record['iteration_selected_index']} equation={record['equation']}",
                flush=True,
            )

    JSON_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "noise",
        "paper_example",
        "variant",
        "status",
        "support_correct",
        "support_reason",
        "equation",
        "support",
        "alpha",
        "beta",
        "coef_hx",
        "coef_dbeta",
        "true_alpha",
        "true_beta",
        "alpha_relative_error",
        "beta_relative_error",
        "coef_hx_relative_error",
        "coef_dbeta_relative_error",
        "refit_mode",
        "physical_refit",
        "order_update_mode",
        "iter_selection_mode",
        "iteration_count",
        "iteration_converged",
        "iteration_stop_reason",
        "iteration_final_index",
        "iteration_best_objective_index",
        "iteration_selected_index",
        "selected_by",
        "mse",
        "objective",
        "sparsity_lamb",
        "d_tol",
        "fit_window_config",
        "max_abs_raw_delta_alpha",
        "max_abs_raw_delta_beta",
        "max_abs_used_delta_alpha",
        "max_abs_used_delta_beta",
        "runtime_seconds",
        "report_path",
        "command",
        "coefficients",
        "first_trace_row",
        "last_trace_row",
        "trace",
        "returncode",
        "stdout_tail",
        "stderr_tail",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_rows(records))

    print(f"wrote {CSV_PATH}")
    print(f"wrote {JSON_PATH}")
    failed = [record for record in records if record["status"] != "ok"]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
