"""Run iterative order-update diagnostics for the paper G-J mainline cases."""

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
CSV_PATH = OUT_DIR / "iterative_paper_cases_normalized_tanh.csv"
JSON_PATH = OUT_DIR / "iterative_paper_cases_normalized_tanh.json"

NO_REFIT_CASES = (
    ("tsfade_clean", "tsfade"),
    ("tsfade_noise5", "tsfade"),
    ("tsfade_noise25", "tsfade"),
    ("dns_gamma_uniform_kmin1e6_oos", "dns_gamma"),
)


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text in {"", "None", "nan"}:
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
    terms: dict[str, float] = {}
    try:
        start = lines.index("active terms:") + 1
    except ValueError:
        return terms
    for line in lines[start:]:
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if stripped == "none":
            break
        if ":" not in stripped:
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


def _support_check(case_kind: str, support: list[str], beta: float | None) -> tuple[bool, str]:
    fractional = [name for name in support if name.startswith("D_x^")]
    if case_kind == "tsfade":
        if len(support) == 2 and "Hx" in support and len(fractional) == 1 and beta is not None and 1.0 < beta <= 2.0:
            return True, "tsfade support is exactly Hx plus one spatial fractional term"
        return False, f"expected Hx + one D_x^beta H term, got {support}"
    if case_kind == "dns_gamma":
        if support == ["Hx"]:
            return True, "dns_gamma support is pure convection Hx"
        return False, f"expected pure Hx support for advective DNS plume, got {support}"
    return False, f"unknown case kind {case_kind}"


def _run_case(paper_example: str, case_kind: str) -> dict[str, Any]:
    command = [
        str(PYTHON),
        "main.py",
        "--paper-example",
        paper_example,
    ]
    method = "gj_iterative_no_refit"
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
    terms = _active_terms(report_text)
    support = list(terms)
    alpha = _as_float(_value(report_text, "alpha"))
    beta = _as_float(_value(report_text, "spatial_beta"))
    support_correct, reason = _support_check(case_kind, support, beta)
    if completed.returncode != 0:
        support_correct = False
        reason = f"command failed with return code {completed.returncode}"
    trace = _iteration_trace(report_text)
    return {
        "paper_example": paper_example,
        "case_kind": case_kind,
        "method": method,
        "diagnostic_only": False,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
        "support_correct": support_correct,
        "pass_fail_reason": reason,
        "equation": _value(report_text, "candidate equation"),
        "support": support,
        "coefficients": terms,
        "alpha": alpha,
        "beta": beta,
        "coef_hx": terms.get("Hx"),
        "coef_dbeta": next((value for name, value in terms.items() if name.startswith("D_x^")), None),
        "true_alpha": _as_float(_value(report_text, "true_alpha")),
        "true_beta": _as_float(_value(report_text, "true_beta")),
        "alpha_relative_error": _as_float(_value(report_text, "alpha_relative_error")),
        "beta_relative_error": _as_float(_value(report_text, "beta_relative_error")),
        "refit_mode": _value(report_text, "refit_mode"),
        "physical_refit": _as_bool(_value(report_text, "physical_refit")),
        "order_update_mode": _value(report_text, "order_update_mode"),
        "iteration_count": _as_float(_value(report_text, "iteration_count")),
        "iteration_converged": _as_bool(_value(report_text, "iteration_converged")),
        "iteration_stop_reason": _value(report_text, "iteration_stop_reason"),
        "iteration_best_objective_index": _as_float(_value(report_text, "iteration_best_objective_index")),
        "iteration_selected_index": _as_float(_value(report_text, "iteration_selected_index")),
        "mse": _as_float(_value(report_text, "mse")),
        "objective": _as_float(_value(report_text, "objective")),
        "physical_refit_selection_objective": _as_float(_value(report_text, "physical_refit_selection_objective")),
        "sparsity_lamb": _as_float(_value(report_text, "sparsity_lamb")),
        "d_tol": _as_float(_value(report_text, "d_tol")),
        "fit_window_config": _value(report_text, "fit_window_config"),
        "runtime_seconds": _as_float(_value(report_text, "end_to_end_seconds")),
        "report_path": str(report) if report is not None else None,
        "trace_summary": {
            "first": trace[0] if trace else None,
            "last": trace[-1] if trace else None,
            "rows": trace,
        },
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-8:]),
    }


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["support"] = " + ".join(record["support"])
        row["coefficients"] = json.dumps(record["coefficients"], sort_keys=True)
        row["trace_summary"] = json.dumps(record["trace_summary"], sort_keys=True)
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
            CSV_PATH = OUT_DIR / "iterative_paper_cases_normalized_tanh.csv"
            JSON_PATH = OUT_DIR / "iterative_paper_cases_normalized_tanh.json"
            return
        except PermissionError:
            continue
    raise PermissionError("No writable output directory found for iterative diagnostics.")


def main() -> int:
    _select_output_dir()
    records: list[dict[str, Any]] = []
    for paper_example, case_kind in NO_REFIT_CASES:
        print(f"running {paper_example} [no refit]", flush=True)
        record = _run_case(paper_example, case_kind)
        records.append(record)
        print(f"  {record['status']} support_correct={record['support_correct']} equation={record['equation']}", flush=True)

    JSON_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "paper_example",
        "case_kind",
        "method",
        "diagnostic_only",
        "status",
        "support_correct",
        "pass_fail_reason",
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
        "refit_mode",
        "physical_refit",
        "order_update_mode",
        "iteration_count",
        "iteration_converged",
        "iteration_stop_reason",
        "iteration_best_objective_index",
        "iteration_selected_index",
        "mse",
        "objective",
        "physical_refit_selection_objective",
        "sparsity_lamb",
        "d_tol",
        "fit_window_config",
        "runtime_seconds",
        "report_path",
        "command",
        "coefficients",
        "trace_summary",
        "returncode",
        "stdout_tail",
        "stderr_tail",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_rows(records))

    gate_records = [record for record in records if not record["diagnostic_only"]]
    all_passed = all(record["support_correct"] for record in gate_records)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {JSON_PATH}")
    print(f"all_no_refit_gate_passed={all_passed}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
