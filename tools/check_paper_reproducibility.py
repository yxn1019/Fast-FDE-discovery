"""Check paper settings in fresh processes without overwriting paper outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CASES = ("tsfade_clean", "tsfade_noise5", "tsfade_noise25")


def worker(deterministic: bool, cases: list[str]) -> dict:
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import numpy as np
    import scipy
    import torch
    import main as entry
    from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(deterministic)
    records = []
    for case in cases:
        # Obtain exactly the config built by the public CLI, and stop before
        # its discovery/reporting stages can overwrite manuscript artifacts.
        captured = []

        class ConfigCaptured(Exception):
            pass

        def capture(config):
            captured.append(config)
            raise ConfigCaptured()

        with patch.object(entry, "GJHybridDiscoverer", capture), patch.object(
            sys, "argv", [str(ROOT / "main.py"), "--paper-example", case]
        ):
            try:
                entry.main()
            except ConfigCaptured:
                pass
        if len(captured) != 1:
            raise RuntimeError(f"Could not resolve the CLI configuration for {case}")
        cli_runtime_restored = not torch.are_deterministic_algorithms_enabled()
        # The optional diagnostic deliberately bypasses the CLI runtime after
        # config capture, to document the effect of the non-paper setting.
        torch.use_deterministic_algorithms(deterministic)
        config = captured[0]
        result = GJHybridDiscoverer(config).discover()
        active = [(name, float(c)) for name, c in zip(
            result.model.term_names, result.model.coefficients.reshape(-1)
        ) if abs(float(c)) > 1e-12]
        support = ["D_x^beta H" if name.startswith("D_x^") else name for name, _ in active]
        records.append(dict(
            case=case, alpha=result.alpha, beta=result.model.metadata["spatial_beta"],
            support=support, coefficients=[c for _, c in active],
            objective=result.model.information_criterion,
            correct_support=set(support) == {"Hx", "D_x^beta H"},
            cli_runtime_restored=cli_runtime_restored,
            config=asdict(config),
            checkpoint_sha256=hashlib.sha256(config.checkpoint_path.read_bytes()).hexdigest(),
        ))
    return dict(deterministic=deterministic, python=sys.version, torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, cases=records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-deterministic-diagnostic", action="store_true",
                        help="Also test the non-paper deterministic=True runtime; failures are retained")
    parser.add_argument("--worker", choices=("default", "deterministic"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # The CLI can announce runtime restoration during config capture.
        # Keep the final worker record unambiguous for the parent process.
        print(json.dumps(worker(args.worker == "deterministic", args.cases), default=str))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new file to preserve previous checks")

    runs = []
    for repeat in range(args.repeats):
        modes = ("default", "deterministic") if args.include_deterministic_diagnostic else ("default",)
        for mode in modes:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                 "--cases", *args.cases], cwd=ROOT, capture_output=True, text=True, check=True,
            )
            record = json.loads(completed.stdout.strip().splitlines()[-1])
            record["repeat"] = repeat
            runs.append(record)
            print(json.dumps(dict(repeat=repeat, mode=mode, results=[{
                key: row[key] for key in ("case", "alpha", "beta", "support", "correct_support")
            } for row in record["cases"]])), flush=True)

    import numpy as np
    comparisons = []
    for case in args.cases:
        rows = [row for run in runs for row in run["cases"] if row["case"] == case]
        reference = rows[0]
        consistent = all(
            row["support"] == reference["support"]
            and np.allclose([row["alpha"], row["beta"], *row["coefficients"], row["objective"]],
                            [reference["alpha"], reference["beta"], *reference["coefficients"], reference["objective"]],
                            rtol=1e-8, atol=1e-10)
            for row in rows
        )
        comparisons.append(dict(case=case, runs=len(rows), consistent=consistent,
                                all_correct_support=all(row["correct_support"] for row in rows)))
    passed = all(row["consistent"] and row["all_correct_support"] for row in comparisons)
    report = dict(passed=passed,
                  structure_passed=all(row["all_correct_support"] for row in comparisons),
                  parameter_consistency_passed=all(row["consistent"] for row in comparisons),
                  comparisons=comparisons, runs=runs,
                  note="Fixed supplied models and observations; this is not a retraining or resampling study.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    print(json.dumps(dict(passed=passed, comparisons=comparisons)), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
