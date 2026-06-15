# Neutral nonlinear order-search baseline (non-self-citing)

Same surrogate / candidate library / STRidge as the paper; only the fractional-order
search is replaced by a generic nonlinear optimizer. Clean tsfade benchmark
(true alpha = 0.78, beta = 1.83). Reproduce:

    python tools/baseline_nonlinear_order_search.py sweep    --n 50
    python tools/baseline_nonlinear_order_search.py optimize

## Why this baseline exists

A reviewer asks for a neutral comparison beyond the authors' own DE baseline.
fPINNs and similar require a known equation structure, so they are not order-search
competitors. The neutral choice is a generic nonlinear optimizer (L-BFGS-B,
Nelder-Mead) over (alpha, beta), alternating with the same STRidge support
selection. The result below shows why this does not work, and therefore why a
derivative-free global search (DE) was the prior choice and why the proposed
order linearization is preferable.

## E1. The order objective is discontinuous (1-D landscape sweep, n = 50)

J(gamma) is the STRidge validation score after support selection. As the order
moves, STRidge re-selects the active terms, so J is piecewise with frequent jumps.

| line | distinct supports | support transitions | max relative J jump |
|---|---|---|---|
| alpha (beta = 1.83) | ~16-19 / 50 points | ~24-26 | ~25% |
| beta  (alpha = 0.78) | ~16-17 / 50 points | ~20 | ~36-45% |

Exact counts vary by a few across runs because the operator/STRidge pipeline has
floating-point nondeterminism (threading); the pattern is stable. The manuscript
text therefore states "more than 20 of the 50 sampled orders along each axis".
The correct advection--fractional-diffusion structure is selected at only a few
isolated orders, and the beta landscape has long flat plateaus (zero gradient)
between spikes. See `figures/baseline_nonlinear/order_objective_landscape.pdf`
(points colored green where the correct structure is selected, gray otherwise).

## E2. Generic optimizers fail on this landscape

| optimizer | start (alpha,beta) | recovered alpha | recovered beta | err alpha | err beta | evals | time (s) |
|---|---|---|---|---|---|---|---|
| L-BFGS-B    | (1.00, 2.00) | 0.747 | 1.980 | 0.033 | 0.150 | 133 | 23.0 |
| L-BFGS-B    | (0.50, 1.50) | 0.722 | 1.372 | 0.058 | 0.458 | 118 | 52.7 |
| L-BFGS-B    | (0.90, 1.90) | 0.747 | 1.738 | 0.033 | 0.092 | 88  | 34.7 |
| Nelder-Mead | (1.00, 2.00) | 0.772 | 2.297 | 0.008 | 0.467 | 67  | 11.7 |
| Nelder-Mead | (0.50, 1.50) | 0.746 | 0.779 | 0.034 | 1.051 | 126 | 74.2 |
| Nelder-Mead | (0.90, 1.90) | 0.772 | 2.086 | 0.008 | 0.256 | 75  | 11.8 |

Findings:

- alpha is recovered to within 0.01-0.06 (its landscape has a clear basin).
- beta fails: errors range 0.09-1.05, strongly start-dependent. L-BFGS-B stalls on
  flat plateaus (beta stuck near the starting 2.0); Nelder-Mead wanders out of the
  admissible interval (beta -> 2.30 or collapses to 0.78).
- The finite-difference gradient of L-BFGS-B is meaningless across the support
  jumps, and Nelder-Mead's simplex contracts onto a plateau or a spike.

## E3. Noise robustness (5% and 25%)

Same protocol on the noisy surrogates (true alpha = 0.78, beta = 1.83). Reproduce:

    python tools/baseline_nonlinear_order_search.py all --case noise5
    python tools/baseline_nonlinear_order_search.py all --case noise25

Outputs are suffixed `_noise5` / `_noise25` so the clean artifacts are preserved.

| noise | support transitions (alpha / beta, of 50) | max relative J jump (alpha / beta) | struct correct (of 6) | alpha err range | beta err range |
|---|---|---|---|---|---|
| clean | 24-26 / 20 | ~25% / ~45% | 3/6 | 0.02-0.11 | 0.07-0.62 |
| 5%    | 25 / 23   | 567% / 285% | 3/6 | 0.015-0.21 | 0.05-0.57 |
| 25%   | 30 / 34   | 219% / 112% | 6/6 | 0.002-0.20 | 0.03-0.17 |

Notes:

- The objective is discontinuous at every noise level, and the jumps grow under
  noise (clean ~45% -> 5% 567%).
- The generic optimizers recover the orders unreliably and start-dependently at
  every level: alpha error up to 0.21, beta error up to 0.62.
- "struct correct" is non-monotonic (3, 3, 6) and misleading at 25%: there the
  two-term structure is selected over a wide order band, but the orders inside it
  are ambiguous (L-BFGS-B lands at alpha ~ 0.96-0.98). Report ORDER ERROR, not
  the structure count, as the discriminating metric.
- Manuscript decision: the clean landscape figure and clean optimizer table are
  shown in Section 3.1; the 5%/25% results are summarized in one sentence
  (order errors up to 0.21 / 0.57) and kept here, not as extra main-text floats.

## Takeaway for the manuscript

Generic local / quasi-Newton optimization of the fractional order is unreliable
because the order objective is discontinuous and partly flat. This is the neutral
baseline a reviewer expects; it also explains the prior derivative-free global DE
search and motivates the order linearization, which replaces the non-smooth order
search with a sequence of linear sparse regressions and recovers beta to within
about 0.03-0.13 across noise levels (Table in Section 4).
