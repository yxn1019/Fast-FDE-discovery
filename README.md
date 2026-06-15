# Laplace-Taylor Fractional Discovery

This repository contains the manuscript-facing implementation for the
Laplace-Taylor fractional PDE discovery paper. The public reproduction path is
centered on `main.py`; support scripts under `tools/` are only for paper tables,
figures, and the DNS / MADE post-processing chains.

Interpreter used throughout:

```powershell
$py = 'D:\Miniconda\envs\sr\python.exe'
```

## Paper Artifact -> Command Map

Each compiled manuscript result and the single command that reproduces it.

| Manuscript item | Section / label | Command |
|---|---|---|
| Space-time benchmark, our method (clean/5%/25%) | Sec. 3.1, Tab. `tab:tsfade` | `& $py main.py --paper-example tsfade_clean` / `tsfade_noise5` / `tsfade_noise25` |
| Space-time benchmark, batch mainline | Sec. 3.1, Tab. `tab:tsfade` | `& $py main.py --paper-task mainline` |
| DE baseline | Sec. 3.1, Tab. `tab:tsfade` | `& $py main.py --paper-task legacy-de --legacy-maxiter 100 --legacy-stridge-mode same_stridge_core --legacy-quiet` |
| Surrogate heatmaps | Fig. `fig:tsfa_retr_surr_heat` | `& $py main.py --paper-task surrogate-heatmaps` |
| MODFLOW/MODPATH plume, discovery | Sec. 3.2, Eq. `eq:dns_gamm_disc` | `& $py main.py --paper-example dns_gamma_uniform_kmin1e6_oos` |
| MODFLOW/MODPATH prediction figures | Sec. 3.2, Figs. forecast / full-domain | `& $py tools\plot_dns_forecast_profiles.py` ; `& $py tools\plot_dns_full_domain_prediction_heatmaps.py` |
| MADE field, discovery | Sec. 3.3, Eq. `eq:made2_disc` | `& $py main.py --paper-example made2_raw_field` (see MADE note below) |
| MADE field, parameter polish + errors + figures | Sec. 3.3, Eq. `eq:made2_polish` | `& $py tools\refit_made2_reaction_term.py --mode full --full-bounds local` |
| Derivative robustness | Sec. 4.1, Tab. `tab:deri_appr_comp` | `& $py main.py --paper-task derivative-robustness` |
| EqGPT candidate generation | Sec. 4.2, Tab. `tab:eqgpt_cand_sele` | `& $py main.py --paper-task candidate-select` |
| Iterative-radius ablation | Sec. 4.3, Tab. `tab:iter_radi_abla` | `& $py main.py --paper-task iter-radius-ablation` |
| Parsimony-bias analytic example | Sec. 4.4, Eq. `eq:pars_disc_comp` | `& $py main.py --paper-example analytic_limitation` |

List all paper-facing examples and their resolved parameters:

```powershell
& $py main.py --list-paper-examples
```

## Space-time Benchmark (Sec. 3.1)

```powershell
& $py main.py --paper-example tsfade_clean
& $py main.py --paper-example tsfade_noise5
& $py main.py --paper-example tsfade_noise25
```

Shared settings (G--J hybrid Taylor route): `8x20` tanh surrogate, `2000`
training points, Gauss--Jacobi `Nq=5`, ridge parameter `2.0`, `10` inner STRidge
iterations, bounded iterative order update with radii
`(delta_alpha, delta_beta) = (0.25, 0.15)`, multi-start `beta0 in {2.0, 1.8, 1.7}`.
Per-case sparsity weight `lambda`: `3e-6` (clean), `2e-4` (5%), `1e-4` (25%);
threshold step `d_tol`: `0.005` (clean and 5%), `0.006` (25%); fit window
`x in [4, 26]` (clean, 5%) and `x in [5, 28]` (25%), `t in [3, 14]`.

## MODFLOW/MODPATH Plume (Sec. 3.2)

```powershell
& $py main.py --paper-example dns_gamma_uniform_kmin1e6_oos
& $py tools\plot_dns_forecast_profiles.py
& $py tools\plot_dns_full_domain_prediction_heatmaps.py
& $py tools\update_paper_dns_uniform.py
```

Uniform inlet injection, `K_min=1e-6`, sparse discovery on the first 80% of the
observation interval, prediction on later snapshots. Paper equation:

```text
D_t^0.83349955 H = -0.6202*Hx
```

Active artifacts:

- `data/dns_gamma075_lc1_uniform_kmin1e6/`
- `data/models/dns_gamma075_lc1_uniform_kmin1e6_tanh_5x50_clean_4000_gj_rawcoords/`
- `results/tsfade_fft_dns_gamma075_lc1_uniform_kmin1e6_gj_hybrid_taylor_result.txt`
- `figures/dns_gamma075_lc1_uniform_kmin1e6_forecast_t100_{linear,loglog}.png`
- `figures/dns_gamma075_lc1_uniform_kmin1e6_full_domain_t80split_heatmaps.png`

## MADE Field Tracer (Sec. 3.3)

The MADE result has two stages: a direct discovery and a fixed-support
parameter polish. The manuscript reports the `raw` variant with the
`hybrid_log_mse` `5x50` tanh surrogate.

**Discovery (Eq. `eq:made2_disc`).** Re-run discovery through the dedicated
paper example:

```powershell
& $py main.py --paper-example made2_raw_field
```

Settings: fit window `x in [9.2, 174.8]`, `t in [49, 370]` d, Laplace
`s in [0.0162, 0.667]`, `lamb=1e-6`, `d_tol=1e-3`, bounded iterative update with
`beta0 in {2.0, 1.8, 1.7}`.

> MADE note: the discovered support `{1, H, Hx, D_x^beta H}` and temporal order
> `alpha=0.999` are stable, but the spatial order sits near the `delta_beta=0.15`
> first-order bound from `beta0=2.0`, so the order selection is sensitive to
> numerical/library drift (a bare re-run can land on `beta~1.98` rather than the
> archived `beta=1.93`). The **exact** manuscript equations are reproduced from
> the frozen discovery bundle in the polish step below, which is the authoritative
> MADE artifact.

**Polish (Eq. `eq:made2_polish`) + figures + errors.** Operates on the frozen
discovery bundle `figures/hydrology_fft_discovery/made2_row252/made2_row252_prediction.npz`:

```powershell
& $py tools\refit_made2_reaction_term.py --mode full --full-bounds local
& $py tools\plot_made2_fft_normalized_paper_figures.py
```

This reproduces, bit-for-bit, the manuscript values:

```text
direct   : D_t^0.999 H = 0.0382 - 0.00890 H - 0.01998 Hx + 0.02502 D_x^1.93 H
polished : D_t^0.983 H = 0.00109 - 0.01453 H - 0.02885 Hx + 0.03224 D_x^1.70 H
```

Active artifacts:

- `data/hydrology_experiments/made2/raw/`
- `data/models/hydrology_experiments/made2_raw_tanh_5x50_hybrid_log_mse/`
- `figures/hydrology_fft_discovery/made2_row252/` (frozen discovery bundle)
- `results/tsfade_fft_hydrology_experiments_made2_raw_tanh_5x50_hybrid_log_mse_gj_hybrid_taylor_result.txt`

## EqGPT Candidate Selection (Sec. 4.2)

EqGPT assets are local to this repository:

- `data/eqgpt/PDEGPT_KdV_equation.pt`
- `data/eqgpt/dict_datas_0725.json`

```powershell
& $py main.py --paper-example candidate_select_tsfade_clean
& $py main.py --paper-example candidate_select_tsfade_noise5
& $py main.py --paper-example candidate_select_tsfade_noise25
& $py main.py --paper-task candidate-select   # full table batch
```

The `EqGPT-10` and `EqGPT-80` rows of Tab. `tab:eqgpt_cand_sele` correspond to
`generated_candidates = 10` and `80` per optimization round.

## Diagnostics and Tables

```powershell
& $py main.py --paper-task derivative-robustness   # Tab. tab:deri_appr_comp
& $py main.py --paper-task iter-radius-ablation     # Tab. tab:iter_radi_abla
& $py main.py --paper-task mainline                 # Tab. tab:tsfade (batch)
```

Curated reproduction groups:

```powershell
& $py main.py --paper-reproduce diagnostics --paper-reproduce-dry-run
& $py main.py --paper-reproduce diagnostics
& $py main.py --paper-reproduce eqgpt
```

The legacy DE task is intentionally slow. It is retained for the manuscript
benchmark table and uses the same normalized tanh surrogate, fit windows,
fractional derivative approximation, and STRidge core as the proposed method.

> **Parameter history note.** The DE rows reported in Tab. `tab:tsfade` were
> produced with an earlier parameter set (noise5 `lambda=1e-4`, clean
> `d_tol=0.025`). The current `--paper-task legacy-de` command inherits the
> updated paper-example parameters (`lambda=2e-4` for noise5, `d_tol=0.005` for
> clean) and gives different equations; those results are archived in
> `results/legacy_de_normalized_tsfade_alpha078_beta183/`. The table values are
> retained as published because the Score column 鈥?which depended on the
> regularisation weight and was not comparable across methods 鈥?has been removed.

## Artifact Layout

- `data/tsfade_retrained_alpha078_beta183/`: space-time FADE raw/noisy data (alpha=0.78, beta=1.83).
- `data/models/tsfade_retrained_alpha078_beta183_normalized_tanh/`: manuscript tsfade checkpoints.
- `data/dns_gamma075_lc1_uniform_kmin1e6/`: MODFLOW/MODPATH plume data and prediction arrays.
- `data/hydrology_experiments/made2/raw/`: MADE field tracer data (raw variant).
- `data/models/hydrology_experiments/made2_raw_tanh_5x50_hybrid_log_mse/`: MADE surrogate checkpoint.
- `data/analytic_tfade_sine/`: single-mode analytic time-fractional ADE data (Sec. 4.4 parsimony example).
- `data/eqgpt/`: local EqGPT checkpoint and dictionary.
- `figures/hydrology_fft_discovery/made2_row252/`: frozen MADE discovery bundle (authoritative for Eq. made2_disc / made2_polish).
- `results/`: manuscript diagnostics, benchmark outputs, and per-case result reports.

> Non-paper material removed during slimming: the `north_loup` hydrology site, the
> MADE `massnorm`/`mse` variants, the 877 MB DNS work directory, and the coarse-scan
> report set under `results/hydrology_discovery_scan/`. The periodic time-fractional
> example is retained (its surrogate panel is still produced by
> `--paper-task surrogate-heatmaps`). See `CODE_SLIMMING_PLAN.md`.
