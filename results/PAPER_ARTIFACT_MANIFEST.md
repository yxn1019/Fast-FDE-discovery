# Paper Artifact Manifest

Generated at: 2026-05-11T16:56:42

Current paper artifacts after iterative mainline, generated iterative candidate-select diagnostics, repo-local heatmap regeneration, and non-paper archive cleanup.

| Case | Label | Kind | Path | Status | Note |
|---|---|---|---|---|---|
| tsfade_st | `sec:tsfade_st;tab:tsfade;fig:tsfade_retrained_surrogate_heatmaps` | raw_data_dir | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\data\tsfade_retrained_alpha078_beta183\raw_data` | repo-local | alpha=0.78 beta=1.83 raw/noisy data for clean, 5%, and 25% tsfade benchmark |
| tsfade_st | `sec:tsfade_st;tab:tsfade;fig:tsfade_retrained_surrogate_heatmaps` | nn_checkpoint_dir | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\data\models\tsfade_retrained_alpha078_beta183_tanh` | repo-local | tanh 8x20 surrogate checkpoints for draft-2000-{0,5,25} |
| tsfade_st | `tab:tsfade` | iterative_mainline_results | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\results\iterative_order_update\iterative_paper_cases.csv` | regenerated | bounded iterative no-refit paper mainline diagnostics |
| candidate_select | `sec:candidate_select` | generated_iterative_results | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\results\candidate_select\generated_iterative_tsfade.csv` | new | EqGPT-style generated RHS + bounded iterative correction diagnostics for 0/5/25% tsfade |
| derivative_robustness | `tab:derivative_approximation_comparison` | robustness_results_dir | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\results\derivative_robustness` | repo-local | self-consistent derivative robustness CSV/JSON artifacts |
| dns_gamma_plume | `sec:dns_gamma_plume;fig:dns_gamma075_fields` | dns_data_results | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\data\dns_gamma075_lc1_flux` | repo-local | Gamma conductivity DNS plume data, model inputs, and result report |
| surrogate_heatmaps | `fig:tsfade_retrained_surrogate_heatmaps;fig:tfade_surrogate_heatmaps` | paper_figures | `D:\OneDrive - HHU\My paper\11Laplace-Taylor discovery\figures\tsfade_retrained_surrogate_heatmaps.pdf` | regenerated_from_repo_local_inputs | metrics JSON contains no external scratch tsfade source paths |
| cleanup | `repository hygiene` | cleanup_manifest | `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\results\CLEANUP_MANIFEST.json` | new | non-paper sweep/refit/long-train diagnostics archived under _archive_nonpaper_20260511; two OneDrive-locked files remain noted |


