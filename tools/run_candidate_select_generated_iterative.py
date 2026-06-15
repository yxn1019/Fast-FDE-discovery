"""Run EqGPT-style generated candidate diagnostics with iterative order updates.

The script is intentionally diagnostic: it runs the generated RHS layer on the
paper tsfade benchmark at clean, 5%, and 25% noise levels, records all outcomes,
and never promotes a failing high-noise result into the manuscript.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
import argparse


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUT_DIR = ROOT / "results" / "candidate_select"

CASES = (
    ("candidate_select_tsfade_clean", "clean"),
    ("candidate_select_tsfade_noise5", "5%"),
    ("candidate_select_tsfade_noise25", "25%"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit-mode",
        choices=("none",),
        default="none",
        help="paper candidate-selection runs use no final refit",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="optional output stem under results/candidate_select",
    )
    parser.add_argument(
        "--generated-top-structures",
        type=int,
        default=None,
        help="override the paper example generated_top_structures setting",
    )
    parser.add_argument("--generated-candidates", type=int, default=None)
    parser.add_argument(
        "--d-tol",
        type=float,
        default=None,
        help="override the initial STRidge threshold step passed to main.py",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=None,
        help="override the physical STRidge sparsity weight passed to main.py",
    )
    parser.add_argument("--eqgpt-model-checkpoint", type=Path, default=None)
    parser.add_argument("--eqgpt-optimize-epochs", type=int, default=None)
    parser.add_argument("--eqgpt-finetune-epochs", type=int, default=None)
    parser.add_argument(
        "--eqgpt-selection-mode",
        choices=("stridge_objective", "eqgpt_reward"),
        default=None,
        help="override generated-structure selection mode",
    )
    return parser.parse_args()


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
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
    path = match.group(1).strip()
    if path == "skipped (permission denied)":
        return None
    return Path(path)


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


def _generated_candidate_summaries(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    try:
        start = lines.index("generated candidate summary:")
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
        if not stripped or stripped == "none":
            break
        parts = [item.strip() for item in stripped.split(",", maxsplit=len(header) - 1)]
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
    return rows


def _support_is_clean_tsfade(support: list[str], beta: float | None) -> tuple[bool, str]:
    frac_terms = [term for term in support if term.startswith("D_x^")]
    if len(support) == 2 and "Hx" in support and len(frac_terms) == 1 and beta is not None and 1.0 < beta < 2.0:
        return True, "support is exactly Hx plus one spatial fractional term"
    return False, f"expected Hx + D_x^beta H, got {support}"


def _structure_terms(structure: str | None) -> set[str]:
    if not structure:
        return set()
    return {part.strip() for part in structure.split("+") if part.strip()}


def _is_target_like_structure(structure: str | None) -> bool:
    terms = _structure_terms(structure)
    return "Hx" in terms and "D_x^beta H" in terms


def _is_exact_target_structure(structure: str | None) -> bool:
    return _structure_terms(structure) == {"Hx", "D_x^beta H"}


def _run_case(
    paper_example: str,
    label: str,
    *,
    refit_mode: str,
    generated_top_structures: int | None,
    generated_candidates: int | None,
    eqgpt_model_checkpoint: Path | None,
    eqgpt_optimize_epochs: int | None,
    eqgpt_finetune_epochs: int | None,
    eqgpt_selection_mode: str | None,
    d_tol: float | None,
    lamb: float | None,
) -> dict[str, Any]:
    command = [str(PYTHON), "main.py", "--paper-example", paper_example]
    if generated_top_structures is not None:
        command.extend(["--generated-top-structures", str(generated_top_structures)])
    if generated_candidates is not None:
        command.extend(["--generated-candidates", str(generated_candidates)])
    if eqgpt_model_checkpoint is not None:
        command.extend(["--eqgpt-model-checkpoint", str(eqgpt_model_checkpoint)])
    if eqgpt_optimize_epochs is not None:
        command.extend(["--eqgpt-optimize-epochs", str(eqgpt_optimize_epochs)])
    if eqgpt_finetune_epochs is not None:
        command.extend(["--eqgpt-finetune-epochs", str(eqgpt_finetune_epochs)])
    if eqgpt_selection_mode is not None:
        command.extend(["--eqgpt-selection-mode", str(eqgpt_selection_mode)])
    if d_tol is not None:
        command.extend(["--d-tol", str(d_tol)])
    if lamb is not None:
        command.extend(["--lamb", str(lamb)])
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
    beta = _as_float(_value(report_text, "spatial_beta"))
    support_correct, support_reason = _support_is_clean_tsfade(support, beta)
    if completed.returncode != 0:
        support_correct = False
        support_reason = f"command failed with return code {completed.returncode}"

    copied_report = None
    copy_note = None
    if report is not None and report.exists():
        copied_report = OUT_DIR / f"{paper_example}_report.txt"
        try:
            shutil.copy2(report, copied_report)
        except PermissionError as exc:
            copy_note = f"report copy skipped: {exc}"
            copied_report = report

    trace = _iteration_trace(report_text)
    generated_summaries = _generated_candidate_summaries(report_text)
    target_like_count = sum(1 for row in generated_summaries if _is_target_like_structure(row.get("structure")))
    exact_target_count = sum(1 for row in generated_summaries if _is_exact_target_structure(row.get("structure")))
    selected_target_like = any(
        str(row.get("selected", "")).lower() == "true" and _is_target_like_structure(row.get("structure"))
        for row in generated_summaries
    )
    return {
        "paper_example": paper_example,
        "noise_label": label,
        "method": f"eqgpt_generated_iterative_{_value(report_text, 'eqgpt_selection_mode') or eqgpt_selection_mode or 'configured'}_{refit_mode}_refit",
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "support_correct": support_correct,
        "support_reason": support_reason,
        "equation": _value(report_text, "candidate equation"),
        "support": support,
        "coefficients": terms,
        "alpha": _as_float(_value(report_text, "alpha")),
        "beta": beta,
        "coef_hx": terms.get("Hx"),
        "coef_dbeta": next((value for name, value in terms.items() if name.startswith("D_x^")), None),
        "true_alpha": _as_float(_value(report_text, "true_alpha")),
        "true_beta": _as_float(_value(report_text, "true_beta")),
        "alpha_relative_error": _as_float(_value(report_text, "alpha_relative_error")),
        "beta_relative_error": _as_float(_value(report_text, "beta_relative_error")),
        "candidate_search": _value(report_text, "candidate_search"),
        "generated_structure": _value(report_text, "generated_structure"),
        "generated_raw_structure": _value(report_text, "generated_raw_structure"),
        "generated_normalized_from": _value(report_text, "generated_normalized_from"),
        "generated_structure_has_fractional": _as_bool(_value(report_text, "generated_structure_has_fractional")),
        "generated_structure_rank": _as_float(_value(report_text, "generated_structure_rank")),
        "eqgpt_checkpoint": _value(report_text, "eqgpt_checkpoint"),
        "eqgpt_epoch": _as_float(_value(report_text, "eqgpt_epoch")),
        "eqgpt_sample_count": _as_float(_value(report_text, "eqgpt_sample_count")),
        "eqgpt_unique_structure_count": _as_float(_value(report_text, "eqgpt_unique_structure_count")),
        "eqgpt_optimize_epochs": _as_float(_value(report_text, "eqgpt_optimize_epochs")),
        "eqgpt_selection_mode": _value(report_text, "eqgpt_selection_mode"),
        "eqgpt_reward": _as_float(_value(report_text, "eqgpt_reward")),
        "eqgpt_r2": _as_float(_value(report_text, "eqgpt_r2")),
        "eqgpt_source_equation": _value(report_text, "eqgpt_source_equation"),
        "generated_complexity_score": _as_float(_value(report_text, "generated_complexity_score")),
        "generated_structure_objective_penalty": _as_float(_value(report_text, "generated_structure_objective_penalty")),
        "generated_structure_count": _as_float(_value(report_text, "generated_structure_count")),
        "target_like_candidate_present": target_like_count > 0,
        "target_like_candidate_count": target_like_count,
        "exact_target_structure_present": exact_target_count > 0,
        "exact_target_structure_count": exact_target_count,
        "selected_structure_target_like": selected_target_like,
        "refit_mode": _value(report_text, "refit_mode"),
        "physical_refit": _as_bool(_value(report_text, "physical_refit")),
        "nonlinear_refit_success": _as_bool(_value(report_text, "nonlinear_refit_success")),
        "nonlinear_refit_cost": _as_float(_value(report_text, "nonlinear_refit_cost")),
        "nonlinear_refit_nfev": _as_float(_value(report_text, "nonlinear_refit_nfev")),
        "nonlinear_refit_status": _value(report_text, "nonlinear_refit_status"),
        "nonlinear_refit_message": _value(report_text, "nonlinear_refit_message"),
        "order_update_mode": _value(report_text, "order_update_mode"),
        "iteration_count": _as_float(_value(report_text, "iteration_count")),
        "iteration_converged": _as_bool(_value(report_text, "iteration_converged")),
        "iteration_stop_reason": _value(report_text, "iteration_stop_reason"),
        "iteration_selected_index": _as_float(_value(report_text, "iteration_selected_index")),
        "objective": _as_float(_value(report_text, "objective")),
        "augmented_stridge_objective_raw": _as_float(_value(report_text, "augmented_stridge_objective_raw")),
        "sparsity_lamb": _as_float(_value(report_text, "sparsity_lamb")),
        "d_tol": _as_float(_value(report_text, "d_tol")),
        "fit_window_config": _value(report_text, "fit_window_config"),
        "runtime_seconds": _as_float(_value(report_text, "end_to_end_seconds")),
        "report_path": str(copied_report) if copied_report is not None else None,
        "report_copy_note": copy_note,
        "command": " ".join(command),
        "trace_summary": {
            "first": trace[0] if trace else None,
            "last": trace[-1] if trace else None,
            "row_count": len(trace),
        },
        "generated_candidate_summaries": generated_summaries,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-10:]),
    }


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["support"] = " + ".join(record["support"])
        row["coefficients"] = json.dumps(record["coefficients"], sort_keys=True, ensure_ascii=False)
        row["trace_summary"] = json.dumps(record["trace_summary"], sort_keys=True, ensure_ascii=False)
        row["generated_candidate_summaries"] = json.dumps(
            record["generated_candidate_summaries"], sort_keys=True, ensure_ascii=False
        )
        rows.append(row)
    return rows


def _writable_output_paths(stem: str) -> tuple[Path, Path]:
    candidates = [
        OUT_DIR,
        Path(tempfile.gettempdir()) / "linear_fractional_results" / "candidate_select",
    ]
    last_error: Exception | None = None
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return directory / f"{stem}.csv", directory / f"{stem}.json"
        except PermissionError as exc:
            last_error = exc
            continue
    raise PermissionError(f"No writable candidate-select output directory found: {last_error}")


def main() -> int:
    args = parse_args()
    cases = CASES
    output_stem = args.output_stem or (
        "generated_iterative_top10_tsfade_normalized_tanh"
        if args.generated_top_structures in (None, 10)
        else "generated_iterative_tsfade_normalized_tanh"
    )
    csv_path, json_path = _writable_output_paths(output_stem)
    records = []
    structure_records: list[dict[str, Any]] = []
    for paper_example, label in cases:
        print(f"running {paper_example}", flush=True)
        record = _run_case(
            paper_example,
            label,
            refit_mode=args.refit_mode,
            generated_top_structures=args.generated_top_structures,
            generated_candidates=args.generated_candidates,
            eqgpt_model_checkpoint=args.eqgpt_model_checkpoint,
            eqgpt_optimize_epochs=args.eqgpt_optimize_epochs,
            eqgpt_finetune_epochs=args.eqgpt_finetune_epochs,
            eqgpt_selection_mode=args.eqgpt_selection_mode,
            d_tol=args.d_tol,
            lamb=args.lamb,
        )
        records.append(record)
        for row in record["generated_candidate_summaries"]:
            row["target_like_structure"] = _is_target_like_structure(row.get("structure"))
            row["exact_target_structure"] = _is_exact_target_structure(row.get("structure"))
            structure_records.append(
                {
                    "paper_example": paper_example,
                    "noise_label": label,
                    "winner_equation": record["equation"],
                    **row,
                }
            )
        print(
            f"  {record['status']} support_correct={record['support_correct']} "
            f"equation={record['equation']}",
            flush=True,
        )

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "paper_example",
        "noise_label",
        "method",
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
        "candidate_search",
        "generated_structure",
        "generated_raw_structure",
        "generated_normalized_from",
        "generated_structure_has_fractional",
        "generated_structure_rank",
        "eqgpt_checkpoint",
        "eqgpt_epoch",
        "eqgpt_sample_count",
        "eqgpt_unique_structure_count",
        "eqgpt_optimize_epochs",
        "eqgpt_selection_mode",
        "eqgpt_reward",
        "eqgpt_r2",
        "eqgpt_source_equation",
        "generated_complexity_score",
        "generated_structure_objective_penalty",
        "generated_structure_count",
        "target_like_candidate_present",
        "target_like_candidate_count",
        "exact_target_structure_present",
        "exact_target_structure_count",
        "selected_structure_target_like",
        "refit_mode",
        "physical_refit",
        "nonlinear_refit_success",
        "nonlinear_refit_cost",
        "nonlinear_refit_nfev",
        "nonlinear_refit_status",
        "nonlinear_refit_message",
        "order_update_mode",
        "iteration_count",
        "iteration_converged",
        "iteration_stop_reason",
        "iteration_selected_index",
        "objective",
        "augmented_stridge_objective_raw",
        "sparsity_lamb",
        "d_tol",
        "fit_window_config",
        "runtime_seconds",
        "report_path",
        "report_copy_note",
        "command",
        "coefficients",
        "trace_summary",
        "generated_candidate_summaries",
        "returncode",
        "stdout_tail",
        "stderr_tail",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_rows(records))

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    if structure_records:
        structure_csv_path = csv_path.with_name(f"{csv_path.stem}_structures.csv")
        structure_json_path = json_path.with_name(f"{json_path.stem}_structures.json")
        structure_json_path.write_text(json.dumps(structure_records, indent=2, ensure_ascii=False), encoding="utf-8")
        structure_fields = [
            "paper_example",
            "noise_label",
            "winner_equation",
            "epoch",
            "rank",
            "selected",
            "structure",
            "raw_structure",
            "eqgpt_source_equation",
            "eqgpt_reward",
            "eqgpt_selection_mode",
            "eqgpt_r2",
            "complexity_score",
            "structure_penalty",
            "endpoint_penalty",
            "iter_start_beta",
            "alpha",
            "beta",
            "coef_hx",
            "coef_dbeta",
            "objective",
            "augmented_objective",
            "validation_residual_norm",
            "condition",
            "support",
            "target_like_structure",
            "exact_target_structure",
        ]
        with structure_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=structure_fields)
            writer.writeheader()
            writer.writerows(structure_records)
        print(f"wrote {structure_csv_path}")
        print(f"wrote {structure_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
