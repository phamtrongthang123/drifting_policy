# Scores Tracker (append-only — never delete rows)

## Can Image (Drifting Policy) — Target: 0.99 (paper), minimum 0.80

| Date | Job ID | Config | Epoch | MSE | Score | Notes |
|------|--------|--------|-------|-----|-------|-------|
| 2026-03-02 | 178465 | old code (wrong S_j, conditional eye, range norm) | 1600 | 0.229 | 0.02 (fluke at ep1350, else 0.0) | Complete failure |
| 2026-03-03 | 178597 | 4 fixes + per_timestep=FALSE | 750 | 0.47 plateau | 0.0 at all rollouts | per_timestep=false kills learning |
| 2026-03-03 | 178656 | 5 fixes + per_timestep=TRUE | 861 | 0.239 (declining but slow) | 0.0 at all rollouts | CANCELLED — drifting alone cannot learn obs-conditional |
| 2026-03-04 | 178684 | 5 fixes + bc_coeff=10.0 | 50 | 0.015 | **0.96** | bc_coeff=10.0 WORKS — matches DDPM (0.96@ep50) exactly |
| 2026-03-22 | 195490 | official port, bc_coeff=0, gen_per_label=8 | 50 | 0.0025 | **0.98** | FAITHFUL PORT — exceeds 0.90 target, bc_coeff=0 ✅ |
| 2026-03-22 | 195490 | official port, bc_coeff=0, gen_per_label=8 | 100 | — | **0.98** | Stable at 0.98 |
| 2026-03-22 | 195490 | official port, bc_coeff=0, gen_per_label=8 | 150 | — | **0.98** | Stable. Job CANCELLED@ep151 (cluster). Final. |

## Can Image (DDPM Baseline) — confirms env/pipeline correct

| Epoch | Score | Paper |
|-------|-------|-------|
| 50 | 0.960 | 0.97 |
| 100 | 0.960 | — |
| 150 | 0.980 | — |

## Other Tasks

| Task | Best Score | Epoch | Notes |
|------|-----------|-------|-------|
| pusht_image | **0.803** | 50 | official port, gen_per_label=8 (job 195496, still running 300ep) |
| pusht_image | **0.862** | 100 | faithful port, gen_per_label=8 (job 195496) |
| pusht_image | **0.862** | 100 | matches paper target ✅. Job 195496 CANCELLED@ep127 (cluster). Final. |
| pusht_lowdim | 0.819 | 700 | close to 0.86 target |
| lift_image | 0.92 | 100 | paper target 1.00; declined to 0.78 by ep150 |
| tool_hang_lowdim | N/A | — | diverging, deprioritized |

## Faithful Port Evaluation (2026-03-22)

All 6 untested configs submitted. Scores will be recorded here as checkpoints arrive.

| Task | Best Score | Epoch | Notes |
|------|-----------|-------|-------|
| lift_image | **1.000** | 50 | faithful port, gen_per_label=8 (job 195504) ✅ matches paper |
| tool_hang_image | **0.260** | 50 | faithful port, gen_per_label=8 (job 195514, running) |
| can_lowdim | **0.980** | 50 | faithful port, gen_per_label=8 (job 195505) ✅ matches paper |
| lift_lowdim | **1.000** | 50 | faithful port, gen_per_label=8 (job 195501) ✅ matches paper |
| pusht_lowdim | **0.871** | 50 | faithful port, gen_per_label=8 (job 195506) ✅ exceeds paper (0.86) |
| tool_hang_lowdim | **0.840** | 50 | faithful port, gen_per_label=8 (job 195515) ✅ far exceeds paper (0.38) |

## Multi-Stage Tasks (State Observation)

Paper targets (Drifting column):
- BlockPush: p1=0.56, p2=0.16
- Kitchen: p_1=1.00, p_2=1.00, p_3=0.99, p_4=0.96

| Task | Job ID | Epoch | p1/p_1 | p2/p_2 | p3/p_3 | p4/p_4 | mean_score | Notes |
|------|--------|-------|--------|--------|--------|--------|------------|-------|
| blockpush_lowdim | 195524 | 525 | 0.30 | 0.12 | — | — | 0.129 | gen_per_label=8 plateaued. CANCELLED. Resubmitting with gen_per_label=32. |
| kitchen_lowdim | 195523 | 87 | **1.00** | **1.00** | **1.00** | **1.00** | 0.569 | ✅ exceeds all paper targets. Best ep43: p4=1.00. CANCELLED. |
