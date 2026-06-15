"""Numerical verification of the order-linearization analysis in the paper.

Three experiments, all using the *actual* paper Gauss--Jacobi operators
(GJHybridDiscoverer._compute_halpha / _compute_hbeta) on the clean tsfade
surrogate, cropped to the interior fitting window x in [4,26), t in [3,14):

  E1  remainder scaling: verify ||D_{g0+dg} - D_{g0} - dg * d_g D_{g0}|| / ||D_{g0}||
      scales as O(dg^2) over dg in [0.01, 0.3]. The order derivative d_g D_{g0}
      uses the same centered finite difference (step h) as the discovery loop;
      a second curve at h/10 checks finite-difference contamination at small dg.

  E2  Gauss--Jacobi convergence in the number of quadrature points N_q
      (config.laguerre_nodes), at fixed alpha=0.78, beta=1.83, vs an N_q=80
      reference. Marks the paper value N_q=5.

  E3  order finite-difference step sensitivity: vary the centered-difference
      step h and compare d_g D_{g0} against a Richardson reference, to show the
      paper steps (h_alpha=0.05, h_beta=0.02) sit in the stable plateau.

No fabricated numbers: every value is computed from the operators and written
to results/verification/. If a slope is not ~2 it is reported as-is.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for p in (ROOT, SRC, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_case_params import make_config  # noqa: E402
from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer  # noqa: E402

# ---- paper benchmark constants (clean tsfade surrogate) ----------------------
CHECKPOINT = (
    ROOT / "data" / "models" / "tsfade_retrained_alpha078_beta183_normalized_tanh"
    / "draft-2000-0" / "best.pkl"
)
TRUE_ALPHA = 0.78
TRUE_BETA = 1.83
FIT_X_MIN, FIT_X_MAX = 4.0, 26.0
FIT_T_MIN, FIT_T_MAX = 3.0, 14.0
PAPER_NQ = 5            # main.py forces laguerre_nodes -> TSFADE_GJ_QUADRATURE_POINTS = 5 for tsfade
H_ALPHA = 0.05          # _alpha_step(0.78)  = min(0.05, ...) = 0.05
H_BETA = 0.02           # _beta_step(1.80)   = min(0.02, ...) = 0.02

OUT_FIG = ROOT / "figures" / "verification"
OUT_RES = ROOT / "results" / "verification"


def make_cfg(laguerre_nodes: int = PAPER_NQ) -> Any:
    return make_config(
        case_name="tsfade_fft",
        checkpoint_file=CHECKPOINT,
        model_root=CHECKPOINT.parent.parent,
        model_alpha_tag="retrained_alpha078_beta183_normalized_tanh",
        trained_point=2000,
        noise_level=0.0,
        activation="tanh",
        hidden_layers=8,
        neurons=20,
        x_min=0.0, x_max=30.0, x_step=0.25,
        t_min=0.0, t_max=15.0, t_step=0.1,
        fit_x_min=FIT_X_MIN, fit_x_max=FIT_X_MAX,
        fit_t_min=FIT_T_MIN, fit_t_max=FIT_T_MAX,
        laguerre_nodes=int(laguerre_nodes),
        enable_spatial_fractional=True,
        beta_reference_orders=(2.0, 1.9, 1.8, 1.7, 1.6, 1.5),
        space_derivative_mode="autodiff",
        spatial_fractional_mode="gj_richardson",
        spatial_correction_method="gj_richardson",
        time_operator_mode="laplace_taylor",
        refit_mode="none",
    )


class Operators:
    """Loads the surrogate once; exposes cached D_t(alpha) and D_x(beta)."""

    def __init__(self, laguerre_nodes: int = PAPER_NQ) -> None:
        self.base_cfg = make_cfg(laguerre_nodes)
        self.runner = GJHybridDiscoverer(self.base_cfg)
        self.torch, self.net, self.metadata = self.runner._load_network()
        self.field = self.runner._build_base_field(self.torch, self.net, self.metadata)
        self.x = np.asarray(self.field.position, dtype=float).reshape(-1)
        self.t = np.asarray(self.field.time, dtype=float).reshape(-1)
        self._xmask = (self.x >= FIT_X_MIN) & (self.x < FIT_X_MAX)
        self._tmask = (self.t >= FIT_T_MIN) & (self.t < FIT_T_MAX)
        self._runner_cache: dict[int, GJHybridDiscoverer] = {laguerre_nodes: self.runner}
        self._cache: dict[tuple[str, int, float], np.ndarray] = {}

    def _runner_for(self, nq: int) -> GJHybridDiscoverer:
        if nq not in self._runner_cache:
            self._runner_cache[nq] = GJHybridDiscoverer(replace(self.base_cfg, laguerre_nodes=int(nq)))
        return self._runner_cache[nq]

    def crop(self, values: np.ndarray) -> np.ndarray:
        return values[np.ix_(self._tmask, self._xmask)]

    def D(self, kind: str, gamma: float, nq: int = PAPER_NQ) -> np.ndarray:
        key = (kind, int(nq), round(float(gamma), 10))
        if key in self._cache:
            return self._cache[key]
        runner = self._runner_for(nq)
        if kind == "time":
            val = runner._compute_halpha(self.torch, self.net, self.field, float(gamma))
        elif kind == "space":
            val = runner._compute_hbeta(self.torch, self.net, self.field, float(gamma))
        else:
            raise ValueError(kind)
        val = np.asarray(val, dtype=float)
        self._cache[key] = val
        return val

    def order_derivative(self, kind: str, gamma0: float, h: float, nq: int = PAPER_NQ) -> np.ndarray:
        """Centered finite difference in the order, matching the discovery loop."""
        plus = self.D(kind, gamma0 + h, nq)
        minus = self.D(kind, gamma0 - h, nq)
        return (plus - minus) / (2.0 * h)


def relnorm(num: np.ndarray, den_norm: float, op: Operators) -> float:
    c = op.crop(num)
    return float(np.linalg.norm(c) / den_norm)


def crop_norm(values: np.ndarray, op: Operators) -> float:
    return float(np.linalg.norm(op.crop(values)))


# ----------------------------------------------------------------------------- E1
def experiment_e1(op: Operators) -> dict[str, Any]:
    cases = [
        ("time_alpha0_0.78", "time", 0.78, H_ALPHA),
        ("time_alpha0_0.60", "time", 0.60, H_ALPHA),
        ("space_beta0_1.80", "space", 1.80, H_BETA),
        ("space_beta0_1.50", "space", 1.50, H_BETA),
    ]
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for label, kind, g0, h_paper in cases:
        base = op.D(kind, g0)
        base_norm = crop_norm(base, op)
        # valid +dg range so g0+dg stays inside the admissible interval
        if kind == "time":
            dg_max = min(0.30, 0.98 - g0)
        else:
            dg_max = min(0.30, 1.98 - g0)
        dgs = np.unique(np.round(np.logspace(math.log10(0.004), math.log10(dg_max), 15), 6))
        for h in (h_paper, h_paper / 10.0):
            dderiv = op.order_derivative(kind, g0, h)
            for dg in dgs:
                R = op.D(kind, g0 + dg) - base - dg * dderiv
                r = relnorm(R, base_norm, op)
                rows.append({"case": label, "kind": kind, "gamma0": g0,
                             "h_fd": h, "is_paper_h": int(math.isclose(h, h_paper)),
                             "delta_gamma": float(dg), "rel_remainder": r})
        # slope fit on paper-h curve over dg in [0.01, min(0.3,dg_max)]
        sub = [row for row in rows if row["case"] == label and row["is_paper_h"] == 1
               and 0.01 <= row["delta_gamma"] <= min(0.30, dg_max)]
        xs = np.log(np.array([s["delta_gamma"] for s in sub]))
        ys = np.log(np.array([s["rel_remainder"] for s in sub]))
        slope, intercept = np.polyfit(xs, ys, 1)
        # remainder at the actual cap (alpha 0.25 / beta 0.15) if reachable
        cap = 0.25 if kind == "time" else 0.15
        r_at_cap = None
        if cap <= dg_max:
            Rc = op.D(kind, g0 + cap) - base - cap * op.order_derivative(kind, g0, h_paper)
            r_at_cap = relnorm(Rc, base_norm, op)
        summary.append({"case": label, "kind": kind, "gamma0": g0, "h_paper": h_paper,
                        "fitted_slope": float(slope), "r2_window": "[0.01, %.2f]" % min(0.30, dg_max),
                        "rel_remainder_at_cap": r_at_cap,
                        "cap": cap})
        print(f"[E1] {label}: fitted log-log slope = {slope:.3f}"
              + (f", rel.remainder at cap {cap} = {r_at_cap:.3e}" if r_at_cap is not None else ""))
    _write_csv(OUT_RES / "e1_remainder_scaling.csv", rows)
    _plot_e1(rows, summary)
    return {"rows": rows, "summary": summary}


def _plot_e1(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[E1] matplotlib unavailable, skipping plot: {exc}")
        return
    # publication style: sans-serif, editable vector text, no top/right spines,
    # no background grid, no on-figure title (the caption carries the message).
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })
    # two method families: temporal operator (blue), spatial operator (orange).
    style = {
        "time_alpha0_0.78": dict(color="#1f4e79", marker="o", label=r"$D_t^{\alpha}$, $\alpha_0=0.78$"),
        "time_alpha0_0.60": dict(color="#6fa8dc", marker="o", label=r"$D_t^{\alpha}$, $\alpha_0=0.60$"),
        "space_beta0_1.80": dict(color="#c55a11", marker="s", label=r"$D_x^{\beta}$, $\beta_0=1.80$"),
        "space_beta0_1.50": dict(color="#f0a868", marker="s", label=r"$D_x^{\beta}$, $\beta_0=1.50$"),
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for case, st in style.items():
        pts = sorted([r for r in rows if r["case"] == case and r["is_paper_h"] == 1],
                     key=lambda r: r["delta_gamma"])
        if not pts:
            continue
        dg = np.array([p["delta_gamma"] for p in pts])
        rr = np.array([p["rel_remainder"] for p in pts])
        ax.loglog(dg, rr, marker=st["marker"], ms=3.2, mfc="white", mew=0.8,
                  lw=1.0, color=st["color"], label=st["label"], zorder=3)
        pts10 = sorted([r for r in rows if r["case"] == case and r["is_paper_h"] == 0],
                       key=lambda r: r["delta_gamma"])
        if pts10:
            dg10 = np.array([p["delta_gamma"] for p in pts10])
            rr10 = np.array([p["rel_remainder"] for p in pts10])
            ax.loglog(dg10, rr10, ls=":", lw=0.8, color=st["color"], alpha=0.6, zorder=2)
    # slope-2 guide anchored through the data cloud at dg=0.05
    paper = [r for r in rows if r["is_paper_h"] == 1]
    near = [r["rel_remainder"] for r in paper if 0.04 <= r["delta_gamma"] <= 0.06]
    y_at = (float(np.median(near)) if near else float(np.median([r["rel_remainder"] for r in paper]))) * 2.2
    gx = np.array([0.008, 0.30])
    ax.loglog(gx, y_at * (gx / 0.05) ** 2, ls="--", lw=0.9, color="0.45", zorder=1)
    ax.text(0.30, y_at * (0.30 / 0.05) ** 2, r"slope $2$", fontsize=6.2,
            color="0.4", ha="right", va="bottom")
    ax.set_xlabel(r"order correction $|\Delta\gamma|$")
    ax.set_ylabel("normalized linearization remainder")
    ax.tick_params(width=0.8, length=3)
    ax.legend(loc="lower right", handlelength=1.5, labelspacing=0.28,
              borderaxespad=0.4, fontsize=6.3)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"e1_remainder_scaling.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"[E1] wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------------------- E2
def experiment_e2(op: Operators) -> dict[str, Any]:
    nqs = [3, 4, 5, 6, 8, 10, 15, 20, 30, 40]
    ref_nq = 80
    ref_t = op.D("time", TRUE_ALPHA, ref_nq)
    ref_x = op.D("space", TRUE_BETA, ref_nq)
    ref_t_norm = crop_norm(ref_t, op)
    ref_x_norm = crop_norm(ref_x, op)
    rows: list[dict[str, Any]] = []
    for nq in nqs:
        et = relnorm(op.D("time", TRUE_ALPHA, nq) - ref_t, ref_t_norm, op)
        ex = relnorm(op.D("space", TRUE_BETA, nq) - ref_x, ref_x_norm, op)
        rows.append({"n_quad": nq, "rel_err_time_Dt": et, "rel_err_space_Dx": ex})
        print(f"[E2] N_q={nq:>2}: time rel.err={et:.3e}  space rel.err={ex:.3e}"
              + ("   <-- paper value" if nq == PAPER_NQ else ""))
    _write_csv(OUT_RES / "e2_gj_convergence.csv", rows)
    _plot_e2(rows)
    return {"rows": rows, "ref_nq": ref_nq}


def _plot_e2(rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[E2] matplotlib unavailable: {exc}")
        return
    nq = np.array([r["n_quad"] for r in rows])
    et = np.array([r["rel_err_time_Dt"] for r in rows])
    ex = np.array([r["rel_err_space_Dx"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.semilogy(nq, et, "o-", label=r"time $D_t^{0.78}$")
    ax.semilogy(nq, ex, "s-", label=r"space $D_x^{1.83}$")
    ax.axvline(PAPER_NQ, color="r", ls="--", lw=1, label=f"paper $N_q={PAPER_NQ}$")
    ax.set_xlabel(r"number of Gauss--Jacobi points $N_q$")
    ax.set_ylabel(r"relative $L_2$ error vs $N_q=80$")
    ax.set_title("E2: Gauss--Jacobi quadrature convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_FIG / "e2_gj_convergence.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[E2] wrote {out}")


# ----------------------------------------------------------------------------- E3
def experiment_e3(op: Operators) -> dict[str, Any]:
    cases = [("time_alpha0_0.78", "time", 0.78, H_ALPHA),
             ("space_beta0_1.80", "space", 1.80, H_BETA)]
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for label, kind, g0, h_paper in cases:
        # admissible step room
        if kind == "time":
            room = min(g0, 1.0 - g0)
        else:
            room = min(g0 - 1.0, 2.0 - g0)
        h_hi = min(0.1, 0.45 * room)
        hs = np.unique(np.round(np.logspace(-5, math.log10(h_hi), 12), 8))
        # Richardson reference from the paper step
        d_h = op.order_derivative(kind, g0, h_paper)
        d_h2 = op.order_derivative(kind, g0, h_paper / 2.0)
        ref = (4.0 * d_h2 - d_h) / 3.0
        ref_norm = crop_norm(ref, op)
        for h in hs:
            d = op.order_derivative(kind, g0, float(h))
            err = relnorm(d - ref, ref_norm, op)
            rows.append({"case": label, "kind": kind, "gamma0": g0,
                         "h_fd": float(h), "is_paper_h": int(math.isclose(h, h_paper, rel_tol=1e-6)),
                         "rel_err_vs_richardson": err})
        err_at_paper = relnorm(op.order_derivative(kind, g0, h_paper) - ref, ref_norm, op)
        summary.append({"case": label, "kind": kind, "gamma0": g0, "h_paper": h_paper,
                        "rel_err_at_paper_h": err_at_paper})
        print(f"[E3] {label}: rel.err of order-derivative at paper h={h_paper} "
              f"(vs Richardson) = {err_at_paper:.3e}")
    _write_csv(OUT_RES / "e3_order_fd_step.csv", rows)
    _plot_e3(rows, cases)
    return {"rows": rows, "summary": summary}


def _plot_e3(rows: list[dict[str, Any]], cases) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[E3] matplotlib unavailable: {exc}")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for label, kind, g0, h_paper in cases:
        pts = [r for r in rows if r["case"] == label]
        h = np.array([p["h_fd"] for p in pts])
        e = np.array([p["rel_err_vs_richardson"] for p in pts])
        line, = ax.loglog(h, e, "o-", ms=3, label=label)
        ax.axvline(h_paper, color=line.get_color(), ls="--", lw=1, alpha=0.7)
    ax.set_xlabel(r"order finite-difference step $h_\gamma$")
    ax.set_ylabel("rel. error of order derivative (vs Richardson)")
    ax.set_title("E3: order finite-difference step sensitivity (dashed: paper $h_\\gamma$)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = OUT_FIG / "e3_order_fd_step.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[E3] wrote {out}")


# ----------------------------------------------------------------------------- io
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def write_summary(op: Operators, e1, e2, e3, cmd: str, elapsed: float) -> None:
    OUT_RES.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Verification of the order-linearization analysis\n")
    lines.append(f"- command: `{cmd}`")
    lines.append(f"- python: `{sys.executable}`")
    lines.append(f"- checkpoint: `{CHECKPOINT}`")
    lines.append(f"- surrogate grid: x in [0,30] step 0.25, t in [0,15] step 0.1; "
                 f"interior fit window x in [4,26), t in [3,14)")
    lines.append(f"- paper N_q (laguerre_nodes) = {PAPER_NQ}; "
                 f"order FD steps h_alpha = {H_ALPHA}, h_beta = {H_BETA}")
    lines.append(f"- wall time: {elapsed:.1f} s\n")
    if e1:
        lines.append("## E1 -- remainder O(|dgamma|^2) scaling\n")
        lines.append("| case | gamma0 | fitted log-log slope | rel. remainder at cap | cap |")
        lines.append("|---|---|---|---|---|")
        for s in e1["summary"]:
            rc = "-" if s["rel_remainder_at_cap"] is None else f"{s['rel_remainder_at_cap']:.3e}"
            lines.append(f"| {s['case']} | {s['gamma0']} | {s['fitted_slope']:.3f} "
                         f"(window {s['r2_window']}) | {rc} | {s['cap']} |")
        lines.append("\nExpected slope ~ 2.0. Dotted (h/10) curves in the figure show "
                     "finite-difference contamination only at the smallest |dgamma|.\n")
    if e2:
        lines.append("## E2 -- Gauss--Jacobi convergence in N_q\n")
        lines.append(f"Reference: N_q = {e2['ref_nq']}.\n")
        lines.append("| N_q | rel.err time Dt^0.78 | rel.err space Dx^1.83 |")
        lines.append("|---|---|---|")
        for r in e2["rows"]:
            mark = "  (paper)" if r["n_quad"] == PAPER_NQ else ""
            lines.append(f"| {r['n_quad']}{mark} | {r['rel_err_time_Dt']:.3e} | {r['rel_err_space_Dx']:.3e} |")
        lines.append("")
    if e3:
        lines.append("## E3 -- order finite-difference step sensitivity\n")
        lines.append("| case | paper h | rel.err of order derivative at paper h (vs Richardson) |")
        lines.append("|---|---|---|")
        for s in e3["summary"]:
            lines.append(f"| {s['case']} | {s['h_paper']} | {s['rel_err_at_paper_h']:.3e} |")
        lines.append("")
    out = OUT_RES / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=["all"],
                        help="any of: e1 e2 e3 all")
    args = parser.parse_args()
    chosen = set(args.experiments) or {"all"}
    if "all" in chosen:
        chosen = {"e1", "e2", "e3"}
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    op = Operators(PAPER_NQ)
    print(f"loaded surrogate: x={op.x.size} pts, t={op.t.size} pts, N_q={PAPER_NQ}")
    e1 = experiment_e1(op) if "e1" in chosen else None
    e2 = experiment_e2(op) if "e2" in chosen else None
    e3 = experiment_e3(op) if "e3" in chosen else None
    elapsed = time.perf_counter() - t0
    write_summary(op, e1, e2, e3, "python tools/verify_linearization_analysis.py " + " ".join(sorted(chosen)), elapsed)
    print(f"done in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
