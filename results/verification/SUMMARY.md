# Verification of the order-linearization analysis

- command: `python tools/verify_linearization_analysis.py e1 e2 e3`
- python: `D:\Miniconda\envs\sr\python.exe`
- checkpoint: `D:\OneDrive - HHU\ML codes\DL-PDE\Linear_fractional\data\models\tsfade_retrained_alpha078_beta183_normalized_tanh\draft-2000-0\best.pkl`
- surrogate grid: x in [0,30] step 0.25, t in [0,15] step 0.1; interior fit window x in [4,26), t in [3,14)
- paper N_q (laguerre_nodes) = 5; order FD steps h_alpha = 0.05, h_beta = 0.02
- wall time: 32.5 s

## E1 -- remainder O(|dgamma|^2) scaling

| case | gamma0 | fitted log-log slope | rel. remainder at cap | cap |
|---|---|---|---|---|
| time_alpha0_0.78 | 0.78 | 1.932 (window [0.01, 0.20]) | - | 0.25 |
| time_alpha0_0.60 | 0.6 | 1.926 (window [0.01, 0.30]) | 1.380e-01 | 0.25 |
| space_beta0_1.80 | 1.8 | 1.960 (window [0.01, 0.18]) | 6.560e-02 | 0.15 |
| space_beta0_1.50 | 1.5 | 1.944 (window [0.01, 0.30]) | 6.703e-02 | 0.15 |

Expected slope ~ 2.0. Dotted (h/10) curves in the figure show finite-difference contamination only at the smallest |dgamma|.

## E2 -- Gauss--Jacobi convergence in N_q

Reference: N_q = 80.

| N_q | rel.err time Dt^0.78 | rel.err space Dx^1.83 |
|---|---|---|
| 3 | 2.557e-03 | 3.781e-02 |
| 4 | 3.793e-04 | 1.886e-02 |
| 5  (paper) | 5.218e-05 | 6.326e-03 |
| 6 | 9.639e-06 | 1.904e-03 |
| 8 | 3.204e-07 | 2.678e-04 |
| 10 | 2.071e-07 | 2.947e-05 |
| 15 | 1.811e-07 | 2.500e-07 |
| 20 | 1.620e-07 | 1.963e-07 |
| 30 | 1.484e-07 | 1.786e-07 |
| 40 | 1.376e-07 | 1.674e-07 |

## E3 -- order finite-difference step sensitivity

| case | paper h | rel.err of order derivative at paper h (vs Richardson) |
|---|---|---|
| time_alpha0_0.78 | 0.05 | 1.663e-03 |
| space_beta0_1.80 | 0.02 | 4.004e-04 |
