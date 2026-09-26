# Fast Fractional Equation Discovery

This repository contains the manuscript-facing implementation and archived
artifacts for the Laplace--Taylor fractional PDE discovery study. It has been
trimmed to the numerical examples, baselines, diagnostics, and figures reported
in the manuscript or its revision response.

## Setup

Python 3.10 or later is recommended.

```powershell
python -m pip install -r requirements.txt
git lfs pull
$py = 'python'
```

The EqGPT checkpoint is stored with Git LFS. All other data and trained neural
surrogates needed by the paper examples are included directly in the repository.

## Paper artifact map

| Manuscript item | Command |
|---|---|
| Space--time benchmark, clean/5%/25% | `& $py main.py --paper-example tsfade_clean` / `tsfade_noise5` / `tsfade_noise25` |
| Differential-evolution baseline | `& $py main.py --paper-task legacy-de --legacy-maxiter 100 --legacy-stridge-mode same_stridge_core --legacy-quiet` |
| Generic local-optimizer baseline | `& $py main.py --paper-task local-optimizers` |
| Fisher--KPP benchmark, clean/5%/25% | `& $py main.py --paper-example fisher_clean` / `fisher_noise5` / `fisher_noise25` |
| MODFLOW--MODPATH plume | `& $py main.py --paper-example dns_gamma_uniform_kmin1e6_oos` |
| MADE field tracer | `& $py main.py --paper-example made2_raw_field` |
| Derivative robustness | `& $py main.py --paper-task derivative-robustness` |
| EqGPT-10 and EqGPT-80 | `& $py main.py --paper-task candidate-select` |
| Bounded/unbounded update ablation | `& $py main.py --paper-task iter-radius-ablation` |
| Radius-sensitivity scan | `& $py main.py --paper-task radius-sensitivity` |
| Matrix conditioning | `& $py main.py --paper-task matrix-conditioning` |
| Order-iteration trajectories | `& $py main.py --paper-task iteration-trajectories` |
| Linearization and operator verification | `& $py main.py --paper-task linearization-verification` |
| Parsimony-bias analytic example | `& $py main.py --paper-example analytic_limitation` |

List every paper-facing example together with its resolved parameters:

```powershell
& $py main.py --list-paper-examples
```

Preview a reproduction group without starting the calculations:

```powershell
& $py main.py --paper-reproduce diagnostics --paper-reproduce-dry-run
```

## Space--time benchmark

```powershell
& $py main.py --paper-example tsfade_clean
& $py main.py --paper-example tsfade_noise5
& $py main.py --paper-example tsfade_noise25
```

The shared settings are an 8-by-20 tanh surrogate, 2000 training points,
five-point Gauss--Jacobi quadrature, ridge parameter 2.0, ten inner STRidge
iterations, update radii `(delta_alpha, delta_beta) = (0.25, 0.15)`, and spatial
reference orders `beta0 in {2.0, 1.8, 1.7}`. The sparsity weights are `3e-6`,
`5e-5`, and `1e-4` for clean, 5%, and 25% noise, respectively.

The periodized-initial-condition runs used to address the periodic-compatibility
question in the revision response are retained as:

```powershell
& $py main.py --paper-example tsfade_pic_clean
& $py main.py --paper-example tsfade_pic_noise5
& $py main.py --paper-example tsfade_pic_noise25
```

## Fisher--KPP benchmark

```powershell
& $py main.py --paper-example fisher_clean
& $py main.py --paper-example fisher_noise5
& $py main.py --paper-example fisher_noise25
& $py tools\plot_fisher_benchmark_figure.py
```

The benchmark data, 8-by-20 tanh checkpoints, discovered equations, prediction
arrays, and manuscript figure outputs are stored under `data/fisher_tfr_alpha07/`,
`data/models/fisher_tfr_alpha07_tanh/`, and `results/fisher_tfr/`.

## MODFLOW--MODPATH plume

```powershell
& $py main.py --paper-example dns_gamma_uniform_kmin1e6_oos
& $py tools\plot_dns_forecast_profiles.py
& $py tools\plot_dns_full_domain_prediction_heatmaps.py
```

The paper case uses uniform inlet injection, `K_min=1e-6`, discovery over the
first 80% of the observation interval, and out-of-sample prediction on later
snapshots. Its archived equation is

```text
D_t^0.83349955 H = -0.6202 Hx.
```

## MADE field tracer

The direct discovery uses the raw MADE data and the 5-by-50 tanh surrogate:

```powershell
& $py main.py --paper-example made2_raw_field
```

The manuscript's fixed-support parameter optimization starts from the frozen
discovery bundle:

```powershell
& $py tools\refit_made2_reaction_term.py --mode full --full-bounds local `
    --output-dir figures\hydrology_fft_discovery\made2_row252_parameter_polish_local
& $py tools\plot_made2_fft_normalized_paper_figures.py `
    --input-npz figures\hydrology_fft_discovery\made2_row252_parameter_polish_local\made2_row252_parameter_polish_full_prediction.npz
```

The authoritative direct-discovery bundle is
`figures/hydrology_fft_discovery/made2_row252/`.

## EqGPT candidate generation

The manuscript compares ten and eighty sampled sequences per optimization round:

```powershell
& $py main.py --paper-example candidate_select_tsfade_clean_10
& $py main.py --paper-example candidate_select_tsfade_clean_80
& $py main.py --paper-task candidate-select
```

The checkpoint `data/eqgpt/PDEGPT_KdV_equation.pt` and its dictionary are stored
locally so the candidate-generation layer does not depend on an external model
download.

## Archived evidence

- `results/baseline_nonlinear/`: local-optimizer table and objective landscapes.
- `results/candidate_select/`: EqGPT-10 and EqGPT-80 outputs used by the paper.
- `results/derivative_robustness/`: derivative-approximation comparison.
- `results/iterative_order_update/`: mainline, radius ablation, and trajectories.
- `results/legacy_de_normalized_tsfade_alpha078_beta183/`: DE baseline outputs.
- `results/matrix_conditioning/`: saved augmented-matrix diagnostics.
- `results/revision_diagnostics/radius_sensitivity.csv`: complete radius scan.
- `results/timing_breakdown/`: discovery and surrogate-training timing records.
- `results/verification/` and `figures/verification/`: Appendix verification.

## Scope of the public repository

Exploratory parameter sweeps, alternative hydrology sites, unused MADE
preprocessing variants, temporary manuscript renders, and obsolete standalone
benchmarks are intentionally excluded. The Git history preserves the previous
public snapshot; the development archive is maintained separately.
