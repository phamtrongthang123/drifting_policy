# Scores Tracker (append-only — never delete rows)

## Can Image (Drifting Policy) — Target: 0.99 (paper), minimum 0.80

| Date | Job ID | Config | Epoch | MSE | Score | Notes |
|------|--------|--------|-------|-----|-------|-------|
| 2026-03-02 | 178465 | old code (wrong S_j, conditional eye, range norm) | 1600 | 0.229 | 0.02 (fluke at ep1350, else 0.0) | Complete failure |
| 2026-03-03 | 178597 | 4 fixes + per_timestep=FALSE | 750 | 0.47 plateau | 0.0 at all rollouts | per_timestep=false kills learning |
| 2026-03-03 | 178656 | 5 fixes + per_timestep=TRUE | 861 | 0.239 (declining but slow) | 0.0 at all rollouts | CANCELLED — drifting alone cannot learn obs-conditional |
| 2026-03-04 | 178684 | 5 fixes + bc_coeff=10.0 | 50 | 0.015 | **0.96** | bc_coeff=10.0 WORKS — matches DDPM (0.96@ep50) exactly |

## Can Image (DDPM Baseline) — confirms env/pipeline correct

| Epoch | Score | Paper |
|-------|-------|-------|
| 50 | 0.960 | 0.97 |
| 100 | 0.960 | — |
| 150 | 0.980 | — |

## Other Tasks

| Task | Best Score | Epoch | Notes |
|------|-----------|-------|-------|
| pusht_image | 0.78 | — | batch=256 |
| pusht_lowdim | 0.819 | 700 | close to 0.86 target |
| lift_image | 0.92 | 100 | paper target 1.00; declined to 0.78 by ep150 |
| tool_hang_lowdim | N/A | — | diverging, deprioritized |
