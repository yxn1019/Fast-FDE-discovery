# Paper artifact manifest

This manifest defines the scope of the public repository.

## Numerical examples

- Space--time fractional advection--dispersion benchmark: clean, 5%, and 25% noise.
- Periodized-initial-condition verification used in the revision response.
- Space--time fractional Fisher--KPP benchmark: clean, 5%, and 25% noise.
- MODFLOW--MODPATH heterogeneous-plume example and out-of-sample prediction.
- MADE field-tracer discovery and fixed-support parameter optimization.
- Analytical single-mode example used to demonstrate parsimony bias.

## Comparisons and diagnostics

- Differential-evolution baseline.
- L-BFGS-B and Nelder--Mead order-search baselines.
- Derivative-approximation robustness.
- EqGPT-10 and EqGPT-80 candidate generation.
- Bounded/unbounded update ablation and radius-sensitivity scan.
- Augmented-matrix conditioning diagnostics.
- Order-iteration trajectories.
- Linearization, quadrature, and order-difference verification.
- Surrogate-training and discovery timing records.

## Authoritative artifact locations

- `data/`: paper data and trained checkpoints only.
- `figures/hydrology_fft_discovery/made2_row252/`: frozen MADE direct-discovery bundle.
- `figures/hydrology_fft_discovery/made2_row252_parameter_polish_local/`: MADE optimized parameters and predictions.
- `results/`: curated result tables, diagnostics, and per-case reports.
- `tools/`: scripts required to generate the listed paper artifacts.

Files from exploratory sweeps and examples not reported in the manuscript or
revision response are intentionally excluded from the public branch.
