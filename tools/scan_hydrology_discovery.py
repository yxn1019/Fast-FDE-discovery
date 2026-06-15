"""Coarse discovery scan for the hydrology NN-surrogate experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "data" / "hydrology_experiments" / "hydrology_preparation_summary.json"
MODEL_ROOT = ROOT / "data" / "models" / "hydrology_experiments"
RESULT_ROOT = ROOT / "results" / "hydrology_discovery_scan"

LAMB_VALUES = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
D_TOL_VALUES = (1.0e-3, 5.0e-3, 2.0e-2)
STAGE_A_X_MAX = {
    "made2": (110.4, 128.8, 147.2, 165.6, 174.8),
    "north_loup": (347.5, 399.8, 452.0, 493.8, 504.31),
}
STAGE_B_T_MIN = {
    "made2": (27.05, 49.0, 126.0),
    "north_loup": (33.0, 44.0, 70.9),
}

SUMMARY_FIELDS = (
    "run_id",
    "stage",
    "experiment",
    "variant",
    "x_max",
    "fit_x_min",
    "fit_x_max",
    "fit_t_min",
    "fit_t_max",
    "lamb",
    "d_tol",
    "returncode",
    "runtime_seconds",
    "report_path",
    "archived_report",
    "candidate_equation",
    "support",
    "active_terms",
    "alpha",
    "beta",
    "objective",
    "validation_residual_norm",
    "physical_refit_selection_residual_norm",
    "condition",
    "best_tolerance",
    "selected_by",
    "rank_class",
    "rank_score",
    "ade_like",
    "zero_model",
    "correction_only",
    "nonlinear_only",
    "command",
)


@dataclass(frozen=True)
class ScanRun:
    run_id: str
    stage: str
    item: dict[str, Any]
    x_max: float
    fit_t_min: float
    lamb: float
    d_tol: float


def _load_summaries(path: Path) -> list[dict[str, Any]]:
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


def _model_dir(item: dict[str, Any], *, hidden_layers: int, neurons: int, loss: str) -> Path:
    label = f"{item['experiment']}_{item['variant']}_tanh_{hidden_layers}x{neurons}_{loss}"
    return MODEL_ROOT / label


def _checkpoint(item: dict[str, Any], *, hidden_layers: int, neurons: int, loss: str) -> Path:
    return _model_dir(item, hidden_layers=hidden_layers, neurons=neurons, loss=loss) / "best.pkl"


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _float_token(value: float) -> str:
    text = f"{float(value):.6g}".replace("+", "")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).replace(".", "p").replace("-", "m")


def _command_for_run(run: ScanRun, *, hidden_layers: int, neurons: int, loss: str) -> list[str]:
    item = run.item
    checkpoint = _checkpoint(item, hidden_layers=hidden_layers, neurons=neurons, loss=loss)
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
        _format_float(float(item["x_min"])),
        "--x-max",
        _format_float(run.x_max),
        "--x-step",
        _format_float(float(item["x_step"])),
        "--t-min",
        _format_float(float(item["discovery_t_min"])),
        "--t-max",
        _format_float(float(item["discovery_t_max_exclusive"])),
        "--t-step",
        _format_float(float(item["discovery_t_step"])),
        "--fit-x-min",
        _format_float(float(item["fit_x_min"])),
        "--fit-x-max",
        _format_float(run.x_max),
        "--fit-t-min",
        _format_float(run.fit_t_min),
        "--fit-t-max",
        _format_float(float(item["fit_t_max"])),
        "--s-min",
        _format_float(float(item["laplace_s_min"])),
        "--s-max",
        _format_float(float(item["laplace_s_max"])),
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
        _format_float(run.lamb),
        "--d-tol",
        _format_float(run.d_tol),
        "--spatial-fractional-lower-bound",
        "0.0",
    ]


def _display_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _parse_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = text.strip()
    if stripped in {"", "None"}:
        return None
    if stripped.lower() == "inf":
        return float("inf")
    if stripped.lower() == "-inf":
        return float("-inf")
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_report_path(stdout: str) -> Path | None:
    match = re.search(r"^detailed report:\s*(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value.startswith("skipped"):
        return None
    return Path(value)


def _parse_active_terms(text: str) -> dict[str, float | None]:
    terms: dict[str, float | None] = {}
    match = re.search(r"^active terms:\s*$", text, flags=re.MULTILINE)
    if not match:
        return terms
    for line in text[match.end() :].splitlines():
        if not line.strip():
            continue
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if not stripped or stripped == "none":
            continue
        if ":" not in stripped:
            break
        name, value = stripped.split(":", 1)
        terms[name.strip()] = _parse_float(value.strip())
    return terms


def _parse_trace_condition(text: str, *, objective: float | None, support: list[str]) -> float | None:
    marker = "iter,alpha0,beta0,delta_alpha_raw,delta_beta_raw,delta_alpha,delta_beta_used"
    if marker not in text:
        marker = "iter,alpha0,beta0,delta_alpha_raw,delta_beta_raw,delta_alpha,delta_beta"
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if marker in line), None)
    if header_index is None:
        return None
    header = [part.strip() for part in lines[header_index].strip().split(",")]
    if "condition" not in header:
        return None
    condition_index = header.index("condition")
    objective_index = header.index("objective") if "objective" in header else None
    support_index = header.index("support") if "support" in header else None
    candidates: list[tuple[int, float]] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("alpha0/beta0 scan"):
            break
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) <= condition_index:
            continue
        condition = _parse_float(parts[condition_index])
        if condition is None:
            continue
        score = 0
        if objective is not None and objective_index is not None and len(parts) > objective_index:
            row_objective = _parse_float(parts[objective_index])
            if row_objective is not None and _same_float(row_objective, objective):
                score += 4
        if support and support_index is not None and len(parts) > support_index:
            row_support = set(part for part in parts[support_index:] if part)
            if set(support).issubset(row_support):
                score += 2
        candidates.append((score, condition))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _same_float(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0)
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= 1.0e-6 * scale


def _is_correction(name: str) -> bool:
    return name == "alpha_correction" or name.startswith("spatial_correction")


def _is_nonlinear(name: str) -> bool:
    return "*" in name or "^2" in name


def _classify(row: dict[str, Any]) -> None:
    support = list(row["support"])
    physical = [name for name in support if not _is_correction(name)]
    zero_model = len(support) == 0 or str(row.get("candidate_equation", "")).endswith("= 0")
    correction_only = bool(support) and not physical
    nonlinear_only = bool(physical) and all(_is_nonlinear(name) for name in physical)
    ade_like = any(name in {"Hx", "Hxx"} or name.startswith("D_x^") for name in physical)
    objective = row.get("objective")
    residual = row.get("validation_residual_norm")
    condition = row.get("condition")
    finite_objective = isinstance(objective, float) and math.isfinite(objective)
    finite_residual = isinstance(residual, float) and math.isfinite(residual)
    finite_condition = isinstance(condition, float) and math.isfinite(condition)
    if row.get("returncode") != 0:
        rank_class = "failed"
    elif zero_model:
        rank_class = "zero"
    elif correction_only:
        rank_class = "correction_only"
    elif not physical:
        rank_class = "empty"
    elif not finite_objective:
        rank_class = "infinite_objective"
    elif nonlinear_only:
        rank_class = "nonlinear_only"
    elif ade_like:
        rank_class = "ade_like"
    else:
        rank_class = "physical_other"
    class_penalty = {
        "ade_like": 0,
        "physical_other": 1,
        "nonlinear_only": 2,
        "infinite_objective": 3,
        "correction_only": 4,
        "zero": 5,
        "empty": 5,
        "failed": 9,
    }[rank_class]
    residual_score = float(residual) if finite_residual else 1.0e99
    objective_score = float(objective) if finite_objective else 1.0e99
    condition_score = math.log10(max(float(condition), 1.0)) if finite_condition else 99.0
    rank_score = class_penalty * 1.0e100 + objective_score + 1.0e-3 * residual_score + 1.0e-6 * condition_score
    row["rank_class"] = rank_class
    row["rank_score"] = rank_score
    row["ade_like"] = ade_like
    row["zero_model"] = zero_model
    row["correction_only"] = correction_only
    row["nonlinear_only"] = nonlinear_only


def _parse_report(text: str) -> dict[str, Any]:
    active_terms = _parse_active_terms(text)
    support = list(active_terms)
    objective = _parse_float(_parse_value(text, "objective"))
    row: dict[str, Any] = {
        "candidate_equation": _parse_value(text, "candidate equation"),
        "support": support,
        "active_terms": active_terms,
        "alpha": _parse_float(_parse_value(text, "alpha")),
        "beta": _parse_float(_parse_value(text, "spatial_beta")),
        "objective": objective,
        "validation_residual_norm": _parse_float(_parse_value(text, "validation_residual_norm")),
        "physical_refit_selection_residual_norm": _parse_float(
            _parse_value(text, "physical_refit_selection_residual_norm")
        ),
        "condition": _parse_trace_condition(text, objective=objective, support=support),
        "best_tolerance": _parse_float(_parse_value(text, "best tolerance")),
        "selected_by": _parse_value(text, "selected_by"),
    }
    _classify(row)
    return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _csv_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "inf" if value > 0 else "-inf"
    return value


def _write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(_json_safe(rows), indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in SUMMARY_FIELDS})


def _run_one(
    run: ScanRun,
    *,
    output_dir: Path,
    hidden_layers: int,
    neurons: int,
    loss: str,
) -> dict[str, Any]:
    command = _command_for_run(run, hidden_layers=hidden_layers, neurons=neurons, loss=loss)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    runtime = time.perf_counter() - started
    stdout = completed.stdout
    reports_dir = output_dir / "reports"
    stdout_dir = output_dir / "stdout"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_dir / f"{run.run_id}.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    report_path = _parse_report_path(stdout)
    archived_report: Path | None = None
    parsed: dict[str, Any] = {
        "candidate_equation": None,
        "support": [],
        "active_terms": {},
        "alpha": None,
        "beta": None,
        "objective": None,
        "validation_residual_norm": None,
        "physical_refit_selection_residual_norm": None,
        "condition": None,
        "best_tolerance": None,
        "selected_by": None,
    }
    if report_path is not None and report_path.exists():
        archived_report = reports_dir / f"{run.run_id}.txt"
        shutil.copyfile(report_path, archived_report)
        parsed = _parse_report(archived_report.read_text(encoding="utf-8"))
    else:
        _classify(parsed)
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "stage": run.stage,
        "experiment": run.item["experiment"],
        "variant": run.item["variant"],
        "x_max": run.x_max,
        "fit_x_min": float(run.item["fit_x_min"]),
        "fit_x_max": run.x_max,
        "fit_t_min": run.fit_t_min,
        "fit_t_max": float(run.item["fit_t_max"]),
        "lamb": run.lamb,
        "d_tol": run.d_tol,
        "returncode": int(completed.returncode),
        "runtime_seconds": runtime,
        "report_path": str(report_path) if report_path is not None else None,
        "archived_report": str(archived_report) if archived_report is not None else None,
        "command": _display_command(command),
        **parsed,
    }
    _classify(row)
    return row


def _stage_a_runs(items: list[dict[str, Any]]) -> list[ScanRun]:
    runs: list[ScanRun] = []
    for item in items:
        experiment = str(item["experiment"])
        for x_max in STAGE_A_X_MAX[experiment]:
            for lamb in LAMB_VALUES:
                for d_tol in D_TOL_VALUES:
                    runs.append(
                        ScanRun(
                            run_id="",
                            stage="stage_a",
                            item=item,
                            x_max=float(x_max),
                            fit_t_min=float(item["fit_t_min"]),
                            lamb=float(lamb),
                            d_tol=float(d_tol),
                        )
                    )
    return _assign_run_ids(runs)


def _assign_run_ids(runs: list[ScanRun], *, start_index: int = 1) -> list[ScanRun]:
    assigned: list[ScanRun] = []
    for offset, run in enumerate(runs, start=start_index):
        run_id = (
            f"{offset:04d}_{run.stage}_{run.item['experiment']}_{run.item['variant']}"
            f"_x{_float_token(run.x_max)}_tmin{_float_token(run.fit_t_min)}"
            f"_lamb{_float_token(run.lamb)}_dtol{_float_token(run.d_tol)}"
        )
        assigned.append(
            ScanRun(
                run_id=run_id,
                stage=run.stage,
                item=run.item,
                x_max=run.x_max,
                fit_t_min=run.fit_t_min,
                lamb=run.lamb,
                d_tol=run.d_tol,
            )
        )
    return assigned


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    objective = row.get("objective")
    residual = row.get("validation_residual_norm")
    condition = row.get("condition")
    return (
        float(row.get("rank_score", 1.0e300)),
        float(objective) if isinstance(objective, float) and math.isfinite(objective) else 1.0e99,
        float(residual) if isinstance(residual, float) and math.isfinite(residual) else 1.0e99,
        math.log10(max(float(condition), 1.0))
        if isinstance(condition, float) and math.isfinite(condition)
        else 99.0,
    )


def _stage_b_runs(
    stage_a_rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    top_k: int,
    start_index: int,
) -> list[ScanRun]:
    item_by_key = {(item["experiment"], item["variant"]): item for item in items}
    runs: list[ScanRun] = []
    seen: set[tuple[str, str, float, float, float, float]] = set()
    for key in sorted(item_by_key):
        experiment, variant = key
        rows = [
            row
            for row in stage_a_rows
            if row.get("stage") == "stage_a"
            and row.get("experiment") == experiment
            and row.get("variant") == variant
            and row.get("returncode") == 0
        ]
        top_rows = sorted(rows, key=_sort_key)[: max(0, int(top_k))]
        for source in top_rows:
            for fit_t_min in STAGE_B_T_MIN[experiment]:
                signature = (
                    experiment,
                    variant,
                    float(source["x_max"]),
                    float(fit_t_min),
                    float(source["lamb"]),
                    float(source["d_tol"]),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                runs.append(
                    ScanRun(
                        run_id="",
                        stage="stage_b",
                        item=item_by_key[key],
                        x_max=float(source["x_max"]),
                        fit_t_min=float(fit_t_min),
                        lamb=float(source["lamb"]),
                        d_tol=float(source["d_tol"]),
                    )
                )
    return _assign_run_ids(runs, start_index=start_index)


def _write_readme(output_dir: Path, rows: list[dict[str, Any]], *, dry_run: bool = False) -> None:
    if dry_run:
        return
    lines = [
        "# Hydrology Discovery Coarse Scan",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs completed: {len(rows)}",
        "",
        "## Rank Counts",
    ]
    for (experiment, variant), grouped in _group_rows(rows).items():
        counts: dict[str, int] = {}
        for row in grouped:
            rank_class = str(row.get("rank_class"))
            counts[rank_class] = counts.get(rank_class, 0) + 1
        count_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        lines.append(f"- {experiment}:{variant}: {count_text}")
    lines.extend(["", "## Best Per Variant"])
    best_by_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for (experiment, variant), grouped in _group_rows(rows).items():
        valid = [row for row in grouped if row.get("returncode") == 0]
        if not valid:
            lines.append(f"- {experiment}:{variant}: no successful runs")
            continue
        best = sorted(valid, key=_sort_key)[0]
        best_by_variant[(experiment, variant)] = best
        lines.append(
            "- "
            f"{experiment}:{variant}: {best.get('candidate_equation')} "
            f"(stage={best.get('stage')}, x_max={best.get('x_max')}, "
            f"t_min={best.get('fit_t_min')}, lamb={best.get('lamb')}, d_tol={best.get('d_tol')}, "
            f"rank={best.get('rank_class')}, objective={best.get('objective')})"
        )
    lines.extend(["", "## Top Physical Candidates"])
    for (experiment, variant), grouped in _group_rows(rows).items():
        physical = [
            row
            for row in grouped
            if row.get("returncode") == 0
            and row.get("rank_class") in {"ade_like", "physical_other", "nonlinear_only"}
        ]
        lines.append(f"### {experiment}:{variant}")
        if not physical:
            lines.append("- no non-zero physical candidates in this grid")
            continue
        for row in sorted(physical, key=_sort_key)[:5]:
            lines.append(
                "- "
                f"{row.get('candidate_equation')} "
                f"[{row.get('stage')}, x_max={row.get('x_max')}, t_min={row.get('fit_t_min')}, "
                f"lamb={row.get('lamb')}, d_tol={row.get('d_tol')}, alpha={row.get('alpha')}, "
                f"beta={row.get('beta')}, objective={row.get('objective')}, "
                f"residual={row.get('validation_residual_norm')}, condition={row.get('condition')}]"
            )
    lines.extend(["", "## Raw/Massnorm Stability"])
    experiments = sorted({str(row.get("experiment")) for row in rows})
    for experiment in experiments:
        raw = best_by_variant.get((experiment, "raw"))
        massnorm = best_by_variant.get((experiment, "massnorm"))
        if raw is None or massnorm is None:
            lines.append(f"- {experiment}: missing raw or massnorm result")
            continue
        raw_support = set(raw.get("support") or [])
        mass_support = set(massnorm.get("support") or [])
        raw_physical = {name for name in raw_support if not _is_correction(str(name))}
        mass_physical = {name for name in mass_support if not _is_correction(str(name))}
        if not raw_physical or not mass_physical:
            lines.append(
                f"- {experiment}: not stable across variants; raw rank={raw.get('rank_class')} "
                f"support={sorted(raw_physical)}, massnorm rank={massnorm.get('rank_class')} "
                f"support={sorted(mass_physical)}"
            )
            continue
        alpha_diff = _abs_float_diff(raw.get("alpha"), massnorm.get("alpha"))
        beta_diff = _abs_float_diff(raw.get("beta"), massnorm.get("beta"))
        lines.append(
            f"- {experiment}: raw support={sorted(raw_physical)}, massnorm support={sorted(mass_physical)}, "
            f"alpha_diff={alpha_diff}, beta_diff={beta_diff}"
        )
    lines.extend(["", "## Recommended Fine Scan Centers"])
    for (experiment, variant), best in best_by_variant.items():
        if best.get("rank_class") not in {"ade_like", "physical_other", "nonlinear_only"}:
            lines.append(f"- {experiment}:{variant}: no fine-scan center from this coarse grid")
            continue
        lines.append(
            "- "
            f"{experiment}:{variant}: center around x_max={best.get('x_max')}, "
            f"fit_t_min={best.get('fit_t_min')}, lamb={best.get('lamb')}, d_tol={best.get('d_tol')}; "
            "scan neighboring x_max/t_min values and one decade around lamb."
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _abs_float_diff(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return None
    return abs(float(left) - float(right))


def _reparse_existing(output_dir: Path) -> list[dict[str, Any]]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing existing summary: {summary_path}")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {summary_path}")
    reparsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        archived = row.get("archived_report")
        report_path = Path(str(archived)) if archived else None
        if report_path is not None and not report_path.is_absolute():
            report_path = ROOT / report_path
        if report_path is not None and report_path.exists():
            parsed = _parse_report(report_path.read_text(encoding="utf-8"))
            row.update(parsed)
        else:
            _classify(row)
        _classify(row)
        reparsed.append(row)
    _write_summaries(reparsed, output_dir)
    _write_readme(output_dir, reparsed)
    return reparsed


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["experiment"]), str(row["variant"])), []).append(row)
    return dict(sorted(grouped.items()))


def _planned_summary(runs: list[ScanRun], *, hidden_layers: int, neurons: int, loss: str) -> None:
    print(f"planned_runs: {len(runs)}")
    counts: dict[tuple[str, str, str], int] = {}
    for run in runs:
        key = (run.stage, run.item["experiment"], run.item["variant"])
        counts[key] = counts.get(key, 0) + 1
        checkpoint = _checkpoint(run.item, hidden_layers=hidden_layers, neurons=neurons, loss=loss)
        if not checkpoint.exists():
            print(f"missing checkpoint: {checkpoint}")
    for key, count in sorted(counts.items()):
        print(f"{key[0]} {key[1]}:{key[2]} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--select", default="all", help="all, made2, north_loup, or experiment:variant")
    parser.add_argument("--loss", default="hybrid_log_mse")
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--neurons", type=int, default=50)
    parser.add_argument("--stage", choices=("stage-a", "all"), default="all")
    parser.add_argument("--stage-a-limit", type=int, default=None, help="truncate Stage A for smoke tests")
    parser.add_argument("--stage-b-top-k", type=int, default=3)
    parser.add_argument("--stage-b-limit", type=int, default=None, help="truncate Stage B for diagnostics")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reparse-existing", action="store_true", help="rebuild summary files from archived reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    if args.reparse_existing:
        if args.output_dir is None:
            raise SystemExit("--reparse-existing requires --output-dir")
        rows = _reparse_existing(args.output_dir)
        print(f"reparsed rows: {len(rows)}")
        print(f"summary csv: {args.output_dir / 'summary.csv'}")
        print(f"summary json: {args.output_dir / 'summary.json'}")
        return

    items = _selected(_load_summaries(args.summary), args.select)
    if not items:
        raise SystemExit(f"No hydrology variants matched --select {args.select!r}")
    stage_a = _stage_a_runs(items)
    if args.stage_a_limit is not None:
        stage_a = stage_a[: max(0, int(args.stage_a_limit))]

    if args.dry_run:
        _planned_summary(stage_a, hidden_layers=args.hidden_layers, neurons=args.neurons, loss=args.loss)
        if args.print_commands:
            for run in stage_a:
                print(_display_command(_command_for_run(run, hidden_layers=args.hidden_layers, neurons=args.neurons, loss=args.loss)))
        return

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULT_ROOT / f"coarse_{args.loss}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(stage_a, start=1):
        print(f"[stage_a {index}/{len(stage_a)}] {run.item['experiment']}:{run.item['variant']} x={run.x_max} lamb={run.lamb:g} d_tol={run.d_tol:g}", flush=True)
        row = _run_one(run, output_dir=output_dir, hidden_layers=args.hidden_layers, neurons=args.neurons, loss=args.loss)
        rows.append(row)
        print(
            f"  -> {row['rank_class']} alpha={row.get('alpha')} beta={row.get('beta')} eq={row.get('candidate_equation')}",
            flush=True,
        )
        _write_summaries(rows, output_dir)
        _write_readme(output_dir, rows)

    if args.stage == "all":
        stage_b = _stage_b_runs(stage_a_rows=rows, items=items, top_k=args.stage_b_top_k, start_index=len(rows) + 1)
        if args.stage_b_limit is not None:
            stage_b = stage_b[: max(0, int(args.stage_b_limit))]
        for index, run in enumerate(stage_b, start=1):
            print(f"[stage_b {index}/{len(stage_b)}] {run.item['experiment']}:{run.item['variant']} x={run.x_max} t_min={run.fit_t_min} lamb={run.lamb:g} d_tol={run.d_tol:g}", flush=True)
            row = _run_one(run, output_dir=output_dir, hidden_layers=args.hidden_layers, neurons=args.neurons, loss=args.loss)
            rows.append(row)
            print(
                f"  -> {row['rank_class']} alpha={row.get('alpha')} beta={row.get('beta')} eq={row.get('candidate_equation')}",
                flush=True,
            )
            _write_summaries(rows, output_dir)
            _write_readme(output_dir, rows)

    _write_summaries(rows, output_dir)
    _write_readme(output_dir, rows)
    print(f"scan output: {output_dir}")
    print(f"summary csv: {output_dir / 'summary.csv'}")
    print(f"summary json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
