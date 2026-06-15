# Verification of the order-linearization analysis

- command: `python tools/verify_linearization_analysis.py e1`
- python: `C:\Python314\python.exe`
- checkpoint: `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\data\models\tsfade_retrained_alpha078_beta183_normalized_tanh\draft-2000-0\best.pkl`
- surrogate grid: x in [0,30] step 0.25, t in [0,15] step 0.1; interior fit window x in [4,26), t in [3,14)
- paper N_q (laguerre_nodes) = 5; order FD steps h_alpha = 0.05, h_beta = 0.02
- wall time: 11.3 s

## E1 -- remainder O(|dgamma|^2) scaling

| case | gamma0 | fitted log-log slope | rel. remainder at cap | cap |
|---|---|---|---|---|
| time_alpha0_0.78 | 0.78 | 1.932 (window [0.01, 0.20]) | - | 0.25 |
| time_alpha0_0.60 | 0.6 | 1.926 (window [0.01, 0.30]) | 1.380e-01 | 0.25 |
| space_beta0_1.80 | 1.8 | 1.960 (window [0.01, 0.18]) | 6.560e-02 | 0.15 |
| space_beta0_1.50 | 1.5 | 1.944 (window [0.01, 0.30]) | 6.703e-02 | 0.15 |

Expected slope ~ 2.0. Dotted (h/10) curves in the figure show finite-difference contamination only at the smallest |dgamma|.
