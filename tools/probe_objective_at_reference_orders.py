"""Evaluate the STRidge selection objective at specific fractional orders.

CMAME revision diagnostic. The differential-evolution baseline reaches a lower
validation objective than the proposed method while recovering a spatial order
far from the ground truth. This script evaluates the *same* objective (same
physical library, same _fit_fde_stridge core) at

  * the ground-truth orders,
  * the orders recovered by the proposed method,
  * the orders recovered by the DE baseline,

so the comparison can be reported honestly: whether the objective's minimum is
attained at the true orders at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from transporteq_discovery.gj_hybrid_discoverer import GJHybridDiscoverer  # noqa: E402

TRUE_ALPHA, TRUE_BETA = 0.78, 1.83

PROBES = {
    "tsfade_clean": [
        ("ground truth", TRUE_ALPHA, TRUE_BETA),
        ("proposed", 0.77889694, 1.85),
        ("DE baseline", 0.74824204, 1.32184),
    ],
    "tsfade_noise5": [
        ("ground truth", TRUE_ALPHA, TRUE_BETA),
        ("proposed", 0.79319089, 1.677661),
        ("DE baseline", 0.75290639, 1.31859),
    ],
    "tsfade_noise25": [
        ("ground truth", TRUE_ALPHA, TRUE_BETA),
        ("proposed", 0.84765988, 1.80),
        ("DE baseline", 0.81044555, 1.33621),
    ],
}


def main() -> None:
    from run_legacy_de_normalized_tsfade_benchmark import config_for_paper_example

    for example, probes in PROBES.items():
        print(f"\n=== {example} ===")
        cfg = config_for_paper_example(example)
        disc = GJHybridDiscoverer(cfg)
        torch, net, meta = disc._load_network()
        ofield = disc._build_base_field(torch, net, meta)
        ffield = disc._fit_window_field(ofield)
        acache: dict[float, np.ndarray] = {}
        bcache: dict[float, np.ndarray] = {}

        for label, alpha, beta in probes:
            ak, bk = round(float(alpha), 10), round(float(beta), 10)
            if ak not in acache:
                acache[ak] = disc._compute_halpha(torch, net, ofield, float(alpha))
            if bk not in bcache:
                bcache[bk] = disc._compute_hbeta(torch, net, ofield, float(beta))
            halpha = disc._restrict_values(acache[ak], ofield, ffield)
            hbeta = disc._restrict_values(bcache[bk], ofield, ffield)
            terms, order = disc._build_physical_terms(ffield, hbeta, float(beta))
            matrix = np.column_stack([terms[n].reshape(-1) for n in order])
            target = halpha.reshape(-1, 1)
            coef, obj, tol, l0, details = disc._fde._fit_fde_stridge(
                matrix, target, protected_indices=(), term_names=tuple(order)
            )
            coef = coef.reshape(-1)
            support = tuple(order[i] for i, v in enumerate(coef) if abs(float(v)) > 1e-12)
            resid = target - matrix @ coef.reshape(-1, 1)
            print(
                f"  {label:13s} a={alpha:.5f} b={beta:.5f}  J={obj:9.4f}  "
                f"val_res={float(details['validation_residual_norm']):8.4f}  "
                f"mse={float(np.mean(resid**2)):.3e}  support={'+'.join(support)}"
            )


if __name__ == "__main__":
    main()
