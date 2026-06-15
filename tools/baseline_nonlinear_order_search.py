"""Neutral non-self-citing baseline: generic nonlinear optimization of the
fractional orders, alternating with the same STRidge sparse regression, on the
same surrogate / library as the paper.

Purpose for the manuscript: show that the order objective J(alpha,beta) -- the
STRidge validation score after support selection -- is piecewise/discontinuous
in the orders (support flips as the order moves), so gradient and quasi-Newton
optimizers (L-BFGS-B, Nelder-Mead) cannot reliably navigate it. This is the
neutral comparison a reviewer asks for, beyond the authors' own DE baseline.

Run the cheap `sweep` smoke test first (1-D landscape); only then `optimize`.

    python tools/baseline_nonlinear_order_search.py sweep
    python tools/baseline_nonlinear_order_search.py optimize
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for p in (ROOT, SRC, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_legacy_de_normalized_tsfade_benchmark import (  # noqa: E402
    config_for_paper_example,
    fit_values_for_orders,
    mainline_stridge_fit,
    support_is_correct,
)
from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer  # noqa: E402

TRUE_ALPHA = 0.78
TRUE_BETA = 1.83
OUT_FIG = ROOT / "figures" / "baseline_nonlinear"
OUT_RES = ROOT / "results" / "baseline_nonlinear"
CASE_PAPER = {"clean": "tsfade_clean", "noise5": "tsfade_noise5", "noise25": "tsfade_noise25"}
TAG = ""  # output filename suffix: "" for clean, "_noise5"/"_noise25" otherwise


class OrderObjective:
    """J(alpha,beta) on the clean tsfade benchmark, identical pipeline to paper."""

    def __init__(self, paper_example: str = "tsfade_clean") -> None:
        self.config = config_for_paper_example(paper_example)
        self.discoverer = GJHybridDiscoverer(self.config)
        self.torch, self.net, self.metadata = self.discoverer._load_network()
        self.operator_field = self.discoverer._build_base_field(self.torch, self.net, self.metadata)
        self.fit_field = self.discoverer._fit_window_field(self.operator_field)
        self.alpha_cache: dict[float, np.ndarray] = {}
        self.beta_cache: dict[float, np.ndarray] = {}
        self.n_eval = 0

    def evaluate(self, alpha: float, beta: float) -> dict[str, Any]:
        self.n_eval += 1
        try:
            matrix, target, term_names = fit_values_for_orders(
                self.discoverer, self.torch, self.net,
                self.operator_field, self.fit_field,
                self.alpha_cache, self.beta_cache, float(alpha), float(beta),
            )
            fit = mainline_stridge_fit(self.discoverer, matrix, target, term_names)
            return {"alpha": float(alpha), "beta": float(beta),
                    "objective": float(fit.objective), "mse": float(fit.mse),
                    "support": tuple(sorted(fit.support)), "n_support": len(fit.support),
                    "coef_dbeta": (float(fit.coef_dbeta) if fit.coef_dbeta is not None else None),
                    "correct": bool(support_is_correct(fit.support, float(beta), fit.coef_dbeta)),
                    "status": "ok"}
        except Exception as exc:  # operators undefined at some orders
            return {"alpha": float(alpha), "beta": float(beta),
                    "objective": float("nan"), "mse": float("nan"),
                    "support": (), "n_support": 0, "status": f"fail:{exc}"}

    def loss(self, params) -> float:
        rec = self.evaluate(params[0], params[1])
        return rec["objective"] if np.isfinite(rec["objective"]) else 1.0e30


# ---------------------------------------------------------------- smoke: 1D sweep
def _support_jumps(records: list[dict[str, Any]]) -> tuple[int, int, float]:
    """Return (#distinct supports, #support transitions, max relative J jump)."""
    ok = [r for r in records if r["status"] == "ok"]
    supports = [r["support"] for r in ok]
    distinct = len(set(supports))
    transitions = sum(1 for a, b in zip(supports, supports[1:]) if a != b)
    max_jump = 0.0
    for a, b in zip(ok, ok[1:]):
        if a["support"] != b["support"] and a["objective"] > 0:
            max_jump = max(max_jump, abs(b["objective"] - a["objective"]) / max(a["objective"], 1e-12))
    return distinct, transitions, max_jump


def sweep(obj: OrderObjective, n: int = 50) -> dict[str, Any]:
    alphas = np.linspace(0.55, 0.98, n)
    betas = np.linspace(1.05, 1.97, n)
    rec_a = [obj.evaluate(a, TRUE_BETA) for a in alphas]
    rec_b = [obj.evaluate(TRUE_ALPHA, b) for b in betas]
    da, ta, ja = _support_jumps(rec_a)
    db, tb, jb = _support_jumps(rec_b)
    print(f"[sweep] alpha-line (beta={TRUE_BETA}): {da} distinct supports, "
          f"{ta} support transitions, max relative J jump = {ja:.3g}")
    print(f"[sweep] beta-line  (alpha={TRUE_ALPHA}): {db} distinct supports, "
          f"{tb} support transitions, max relative J jump = {jb:.3g}")
    _write_csv(OUT_RES / f"landscape_alpha{TAG}.csv", rec_a)
    _write_csv(OUT_RES / f"landscape_beta{TAG}.csv", rec_b)
    _plot_landscape(rec_a, rec_b)
    return {"alpha": {"distinct": da, "transitions": ta, "max_jump": ja},
            "beta": {"distinct": db, "transitions": tb, "max_jump": jb}}


def _plot_landscape(rec_a: list[dict[str, Any]], rec_b: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[sweep] matplotlib unavailable: {exc}")
        return
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    correct_color, wrong_color = "#2e8b57", "#bdbdbd"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, rec, xlabel, truth, key in (
        (axes[0], rec_a, r"temporal order $\alpha$ (at $\beta=1.83$)", TRUE_ALPHA, "alpha"),
        (axes[1], rec_b, r"spatial order $\beta$ (at $\alpha=0.78$)", TRUE_BETA, "beta"),
    ):
        ok = [r for r in rec if r["status"] == "ok" and np.isfinite(r["objective"])]
        xs = np.array([r[key] for r in ok])
        js = np.array([r["objective"] for r in ok])
        good = np.array([bool(r.get("correct")) for r in ok])
        # neutral connecting line shows the discontinuous J trajectory
        ax.plot(xs, js, "-", lw=0.7, color="0.8", zorder=1)
        ax.scatter(xs[good], js[good], s=11, c=correct_color, zorder=3,
                   edgecolors="none", label="correct structure")
        ax.scatter(xs[~good], js[~good], s=9, c=wrong_color, zorder=2,
                   edgecolors="none", label="incorrect structure")
        ax.axvline(truth, color="k", ls="--", lw=0.9, alpha=0.7)
        ax.text(truth, ax.get_ylim()[1], "true", fontsize=6, va="top", ha="center")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("order objective $J$ (STRidge score)")
        ax.tick_params(width=0.8, length=3)
    axes[0].legend(loc="upper right", fontsize=6.0, handletextpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"order_objective_landscape{TAG}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"[sweep] wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------- baseline optimizers
# admissible order ranges; the spatial operator is defined only for 1 < beta < 2,
# so beta is constrained to (1, 2) rather than (0, 2).
A_LO, A_HI, B_LO, B_HI = 0.0, 1.0, 1.0, 2.0


def _to_orders(u: np.ndarray) -> tuple[float, float]:
    a = A_LO + (A_HI - A_LO) / (1.0 + np.exp(-u[0]))
    b = B_LO + (B_HI - B_LO) / (1.0 + np.exp(-u[1]))
    return float(a), float(b)


def _to_unconstrained(a: float, b: float) -> np.ndarray:
    pa = min(max((a - A_LO) / (A_HI - A_LO), 1e-6), 1.0 - 1e-6)
    pb = min(max((b - B_LO) / (B_HI - B_LO), 1e-6), 1.0 - 1e-6)
    return np.array([np.log(pa / (1.0 - pa)), np.log(pb / (1.0 - pb))])


def optimize(obj: OrderObjective) -> dict[str, Any]:
    from scipy.optimize import minimize
    bounds = [(0.02, 0.98), (1.02, 1.98)]
    starts = [(1.0 - 1e-3, 2.0 - 1e-3), (0.5, 1.5), (0.9, 1.9)]
    rows: list[dict[str, Any]] = []
    for method in ("L-BFGS-B", "Nelder-Mead"):
        for s in starts:
            obj.n_eval = 0
            t0 = time.perf_counter()
            if method == "Nelder-Mead":
                # Nelder-Mead has no box constraints; a sigmoid reparametrization
                # keeps the orders inside the admissible ranges during the search.
                res = minimize(lambda u: obj.loss(_to_orders(u)),
                               _to_unconstrained(*s), method="Nelder-Mead")
                a, b = _to_orders(res.x)
            else:
                res = minimize(obj.loss, np.array(s, float), method=method, bounds=bounds)
                a, b = float(res.x[0]), float(res.x[1])
            dt = time.perf_counter() - t0
            final = obj.evaluate(a, b)
            rows.append({"method": method, "start_alpha": s[0], "start_beta": s[1],
                         "alpha": a, "beta": b,
                         "err_alpha": abs(a - TRUE_ALPHA), "err_beta": abs(b - TRUE_BETA),
                         "objective": final["objective"], "structure_correct": bool(final["correct"]),
                         "support": str(final["support"]),
                         "n_eval": obj.n_eval, "seconds": dt, "success": bool(res.success)})
            print(f"[opt] {method:11s} start=({s[0]:.2f},{s[1]:.2f}) -> "
                  f"alpha={a:.3f} beta={b:.3f} (err {abs(a-TRUE_ALPHA):.3f},{abs(b-TRUE_BETA):.3f}) "
                  f"struct={'Y' if final['correct'] else 'N'} "
                  f"J={final['objective']:.4g} evals={obj.n_eval} {dt:.1f}s")
    _write_csv(OUT_RES / f"optimizer_baseline{TAG}.csv", rows)
    return {"rows": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (str(v) if isinstance(v, tuple) else v) for k, v in r.items()})
    print(f"  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sweep", "optimize", "all"], default="sweep", nargs="?")
    parser.add_argument("--n", type=int, default=50, help="sweep resolution per axis")
    parser.add_argument("--case", choices=list(CASE_PAPER), default="clean",
                        help="benchmark noise case")
    args = parser.parse_args()
    global TAG
    TAG = "" if args.case == "clean" else f"_{args.case}"
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    obj = OrderObjective(CASE_PAPER[args.case])
    print(f"loaded {args.case} tsfade objective (true alpha={TRUE_ALPHA}, beta={TRUE_BETA})")
    if args.mode in ("sweep", "all"):
        sweep(obj, n=args.n)
    if args.mode in ("optimize", "all"):
        optimize(obj)
    print(f"done in {time.perf_counter() - t0:.1f} s ({obj.n_eval} final evals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
