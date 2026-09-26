"""Evolution of the estimated fractional orders under bounded updates.

CMAME revision, Reviewer #1 comments 1 and 4.

Claim defended by the figure: the bounded linearization update carries each
order from the near-integer initialization to its estimate within a few steps,
and the reported model is the trial with the lowest validation score across all
multi-start branches.

What is plotted is the *estimated* order at each trial, alpha_0 + Delta alpha
and beta_0 + Delta beta, which is the quantity reported in the manuscript. The
internal next reference point is a different quantity and is not shown.

Trajectories come from ``main.py`` runs, so the figure is produced by exactly
the configuration behind the reported models. All multi-start branches are
dumped through GJ_TRACE_DUMP_PATH; the report alone keeps only the selected
branch.

Outputs SVG, PDF, TIFF and PNG.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "iterative_order_update" / "order_iteration_trajectories"

CASES = [("tsfade_clean", "clean"), ("tsfade_noise5", "5% noise"), ("tsfade_noise25", "25% noise")]
TRUE_ALPHA, TRUE_BETA = 0.78, 1.83
START_ALPHA = 0.99

OTHER = "#b8c4cc"      # branches not selected
PICKED = "#08519c"     # selected branch
TRUTH = "#525252"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
})


def run_case(example: str):
    """Return (branches, reported_alpha, reported_beta) for one paper example."""
    with tempfile.TemporaryDirectory(dir=r"D:\\") as tmp:
        dump = Path(tmp) / "trace.jsonl"
        env = dict(os.environ, GJ_TRACE_DUMP_PATH=str(dump))
        proc = subprocess.run(
            [PY, "main.py", "--paper-example", example, "--refit-mode", "none",
             "--iter-selection-mode", "best_objective"],
            cwd=str(ROOT), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out = proc.stdout or ""
        m = re.search(r"^alpha = ([\d.eE+-]+), beta = ([\d.eE+-]+)", out, re.MULTILINE)
        if m is None:
            raise RuntimeError(f"{example}: no reported orders\n{out[-1200:]}")
        branches = []
        for line in dump.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows = rec["trace"]
            branches.append({
                "beta_start": float(rec["iter_start_beta"]),
                "alpha": [r["alpha0"] + r["delta_alpha"] for r in rows],
                "beta": [r["beta0"] + r["delta_beta"] for r in rows],
                "objective": [r["objective"] for r in rows],
                # A trial whose support carries no spatial fractional column
                # discards the ratio, so its beta is not an order estimate. The
                # support string is the field the dump carries for this.
                "beta_defined": ["D_x^" in str(r["support"]) for r in rows],
            })
    return branches, float(m.group(1)), float(m.group(2))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(183 / 25.4, 125 / 25.4), sharex="col")
    for row in (0, 1):
        for col in (1, 2):
            axes[row, col].sharey(axes[row, 0])

    for col, (example, label) in enumerate(CASES):
        branches, rep_a, rep_b = run_case(example)
        best = min(
            ((bi, ti, b["objective"][ti]) for bi, b in enumerate(branches)
             for ti in range(len(b["objective"]))),
            key=lambda item: item[2],
        )
        print(f"{label}: {len(branches)} branches, reported ({rep_a:.4f}, {rep_b:.4f}), "
              f"lowest J at branch {best[0]} trial {best[1]}")

        # The admissible ranges 0 < alpha < 1 and 1 < beta <= 2 are the natural
        # limits: a narrow window would exaggerate small order deviations. The
        # score row is left to autoscale per case, since J is not comparable
        # across noise levels.
        for row, (key, truth, ylabel, ylim) in enumerate((
            ("alpha", TRUE_ALPHA, r"estimated $\alpha$", (0.0, 1.2)),
            ("beta", TRUE_BETA, r"estimated $\beta$", (1.0, 2.2)),
            ("objective", None, r"validation score $\mathcal{J}$", None),
        )):
            ax = axes[row, col]
            if truth is not None:
                ax.axhline(truth, color=TRUTH, ls=(0, (4, 2)), lw=0.8, zorder=1)
            # Trials are indexed from 1; k = 0 is the reference order the branch
            # starts from, which is an input and carries no model, so it appears
            # on the order rows only.
            for bi, br in enumerate(branches):
                picked = bi == best[0]
                start = START_ALPHA if key == "alpha" else br["beta_start"]
                xs = list(range(1, len(br[key]) + 1))
                if key != "objective":
                    xs, ys = [0] + xs, [start] + br[key]
                else:
                    ys = br[key]
                ax.plot(xs, ys,
                        marker="o", markersize=3.0 if picked else 2.4,
                        lw=1.3 if picked else 0.9,
                        color=PICKED if picked else OTHER,
                        zorder=4 if picked else 2,
                        label=(r"selected branch" if picked else "other starts")
                        if (row == 0 and col == 0 and bi in (best[0], (best[0] + 1) % len(branches)))
                        else None)
                if key != "objective":
                    ax.plot([0], [start], linestyle="none", marker="s",
                            markersize=4.0 if picked else 3.4,
                            color=PICKED if picked else OTHER,
                            zorder=5 if picked else 3,
                            label="reference order"
                            if (row == 0 and col == 0 and picked) else None)
                if key == "beta":
                    ks = [k + 1 for k, ok in enumerate(br["beta_defined"]) if not ok]
                    ax.plot(ks, [br[key][k - 1] for k in ks], linestyle="none", marker="o",
                            markersize=5.4 if picked else 4.4, markerfacecolor="white",
                            markeredgecolor=PICKED if picked else OTHER,
                            markeredgewidth=1.0, zorder=5 if picked else 3,
                            label="no fractional term selected"
                            if (col == 0 and bi == best[0]) else None)
            ax.plot([best[1] + 1], [branches[best[0]][key][best[1]]], marker="o",
                    markersize=7.0, markerfacecolor="none", markeredgecolor=PICKED,
                    markeredgewidth=1.1, zorder=6,
                    label="reported model" if (row == 0 and col == 0) else None)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(ylabel)
            elif ylim is not None:
                # Rows 1 and 2 share a scale across columns, so the repeated tick
                # labels are redundant ink. The score row keeps its own labels.
                ax.tick_params(labelleft=False)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_xticks(range(max(len(b["objective"]) for b in branches) + 1))
            ax.margins(x=0.25, y=0.18)
            if row == 0:
                ax.set_title(label, fontsize=7.5)
            elif row == 2:
                ax.set_xlabel("iteration $k$")

    axes[0, 0].annotate("true order", xy=(0.03, TRUE_ALPHA), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", color=TRUTH, fontsize=6.2)
    axes[1, 0].annotate("true order", xy=(0.03, TRUE_BETA), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", color=TRUTH, fontsize=6.2)
    # Legend sits above the panel row so it never overlaps the traces.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for handle, label in zip(*axes[1, 0].get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.2,
               handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.045), frameon=False)
    for (row, col), tag in zip(((r, c) for r in range(3) for c in range(3)), "abcdefghi"):
        # Columns 1 and 2 of the order rows carry no tick labels, so their panel
        # letters sit closer to the axes.
        dx = -0.18 if (col == 0 or row == 2) else -0.07
        axes[row, col].text(dx, 1.06, tag, transform=axes[row, col].transAxes,
                            fontsize=8, fontweight="bold")

    fig.tight_layout(pad=0.5)
    # SVG and PDF keep editable text (svg.fonttype none, pdf.fonttype 42); the
    # TIFF is the 600 dpi LZW raster journals ask for.
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {OUT}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
