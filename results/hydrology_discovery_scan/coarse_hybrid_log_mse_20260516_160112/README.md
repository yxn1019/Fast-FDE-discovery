# Hydrology Discovery Coarse Scan

Generated: 2026-05-16T16:30:18
Runs completed: 276

## Rank Counts
- made2:massnorm: zero=69
- made2:raw: ade_like=27, physical_other=6, zero=36
- north_loup:massnorm: zero=69
- north_loup:raw: ade_like=33, zero=36

## Best Per Variant
- made2:massnorm: D_t^0.999 H = 0 (stage=stage_b, x_max=165.6, t_min=126.0, lamb=1e-06, d_tol=0.005, rank=zero, objective=inf)
- made2:raw: D_t^0.999 H = 0.01317*1 + -0.005685*H + -0.01839*Hx + 0.02142 D_x^1.928585 H (stage=stage_b, x_max=174.8, t_min=126.0, lamb=1e-06, d_tol=0.005, rank=ade_like, objective=7.824275157715002)
- north_loup:massnorm: D_t^0.999 H = 0 (stage=stage_b, x_max=399.8, t_min=70.9, lamb=1e-06, d_tol=0.005, rank=zero, objective=inf)
- north_loup:raw: D_t^0.999 H = -0.9742*Hx (stage=stage_a, x_max=347.5, t_min=33.0, lamb=1e-06, d_tol=0.02, rank=ade_like, objective=0.5924514957814551)

## Top Physical Candidates
### made2:massnorm
- no non-zero physical candidates in this grid
### made2:raw
- D_t^0.999 H = 0.01317*1 + -0.005685*H + -0.01839*Hx + 0.02142 D_x^1.928585 H [stage_b, x_max=174.8, t_min=126.0, lamb=1e-06, d_tol=0.005, alpha=0.999, beta=1.928585249232164, objective=7.824275157715002, residual=3.571265370979, condition=1789000.0]
- D_t^0.999 H = 0.01317*1 + -0.005685*H + -0.01839*Hx + 0.02142 D_x^1.928585 H [stage_b, x_max=174.8, t_min=126.0, lamb=1e-06, d_tol=0.001, alpha=0.999, beta=1.928585249232164, objective=7.824275157715009, residual=3.5712653709789985, condition=1789000.0]
- D_t^0.999 H = 0.01412*1 + -0.005695*H + -0.01839*Hx + 0.02136 D_x^1.928569 H [stage_b, x_max=165.6, t_min=126.0, lamb=1e-06, d_tol=0.001, alpha=0.999, beta=1.9285687843583763, objective=7.96377677483789, residual=3.8003347854992153, condition=1789000.0]
- D_t^0.999 H = 0.04106*1 + -0.008924*H + -0.01939*Hx + 0.02569 D_x^1.969391 H [stage_b, x_max=165.6, t_min=49.0, lamb=1e-06, d_tol=0.001, alpha=0.999, beta=1.9693906682163513, objective=32.77647566855896, residual=11.041903758465232, condition=6201000.0]
- D_t^0.999 H = 0.03821*1 + -0.008901*H + -0.01998*Hx + 0.02502 D_x^1.933515 H [stage_b, x_max=174.8, t_min=49.0, lamb=1e-06, d_tol=0.001, alpha=0.999, beta=1.9335153968156105, objective=33.382358454997046, residual=11.993006208263306, condition=5923000.0]
### north_loup:massnorm
- no non-zero physical candidates in this grid
### north_loup:raw
- D_t^0.999 H = -0.9742*Hx [stage_a, x_max=347.5, t_min=33.0, lamb=1e-06, d_tol=0.02, alpha=0.999, beta=1.8, objective=0.5924514957814551, residual=0.24179839428982777, condition=342800.0]
- D_t^0.999 H = -0.9742*Hx [stage_b, x_max=347.5, t_min=33.0, lamb=1e-06, d_tol=0.02, alpha=0.999, beta=1.8, objective=0.5924514957814551, residual=0.24179839428982777, condition=342800.0]
- D_t^0.999 H = -0.9747*Hx [stage_a, x_max=452.0, t_min=33.0, lamb=1e-06, d_tol=0.02, alpha=0.999, beta=2.0, objective=0.661343394811122, residual=0.2924899907196112, condition=391600.0]
- D_t^0.999 H = -0.9747*Hx [stage_b, x_max=452.0, t_min=33.0, lamb=1e-06, d_tol=0.02, alpha=0.999, beta=2.0, objective=0.661343394811122, residual=0.2924899907196112, condition=391600.0]
- D_t^0.999 H = -0.6881*Hx + -0.4088*H*Hx [stage_a, x_max=347.5, t_min=33.0, lamb=1e-06, d_tol=0.005, alpha=0.999, beta=1.7, objective=0.9226463740183478, residual=0.2387624138436353, condition=342600.0]

## Raw/Massnorm Stability
- made2: not stable across variants; raw rank=ade_like support=['1', 'D_x^1.928585 H', 'H', 'Hx'], massnorm rank=zero support=[]
- north_loup: not stable across variants; raw rank=ade_like support=['Hx'], massnorm rank=zero support=[]

## Recommended Fine Scan Centers
- made2:massnorm: no fine-scan center from this coarse grid
- made2:raw: center around x_max=174.8, fit_t_min=126.0, lamb=1e-06, d_tol=0.005; scan neighboring x_max/t_min values and one decade around lamb.
- north_loup:massnorm: no fine-scan center from this coarse grid
- north_loup:raw: center around x_max=347.5, fit_t_min=33.0, lamb=1e-06, d_tol=0.02; scan neighboring x_max/t_min values and one decade around lamb.
