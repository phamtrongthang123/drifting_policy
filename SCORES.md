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
| tool_hang_image | **0.260** | 50 | faithful port, gen_per_label=8 (job 195514) |
| tool_hang_image | **0.520** | 100 | gen_per_label=8 (job 195514). Peaked here, declined to 0.34@ep150. Job finished ep299. |
| tool_hang_image | **0.740** | 25 | gen_per_label=4 (job 195552) ✅ exceeds paper (0.67). Sweep winner. |
| tool_hang_image | 0.700 | 50 | gen_per_label=4 (job 195552, running) |
| tool_hang_image | 0.560 | 50 | gen_per_label=2 (job 195553, running) |
| tool_hang_image | 0.200 | 25 | gen_per_label=16 (job 195551, running) |
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
| blockpush_lowdim | 195524 | 525 | 0.30 | 0.12 | — | — | 0.129 | gen_per_label=8 plateaued. CANCELLED. |
| blockpush_lowdim | 195525 | 200 | 0.12 | 0.06 | — | — | 0.069 | gen_per_label=32. Worse than gpl=8. CANCELLED. |
| blockpush_lowdim | 195545 | 1917 | 0.20 | 0.10 | — | — | 0.149 | gen_per_label=2. Plateaued ~p1=0.16-0.20. Running. |
| kitchen_lowdim | 195523 | 87 | **1.00** | **1.00** | **1.00** | **1.00** | 0.569 | ✅ exceeds all paper targets. Best ep43: p4=1.00. CANCELLED. |

## BlockPush Sweep (2026-03-24)

Paper targets: p1=0.56, p2=0.16. All use `drifting_blockpush_lowdim.yaml` base config (batch_size=256, embed_dim=128, lr_scheduler=constant_with_warmup, per_timestep_loss=true, bc_coeff=10.0). Jobs still running (5000 epochs max).

### Abs action sweep (abs_action=true, zarr=multimodal_push_seed_abs.zarr)

| Job ID | gen_per_label | Epoch | Best p1 | Best p2 | Avg p1 (last 20) | Notes |
|--------|---------------|-------|---------|---------|-------------------|-------|
| 195634_0 | 2 | 247 | **0.56** | **0.30** | 0.257 | ✅ p1 matches paper target, p2 far exceeds. Running. |
| 195634_3 | 4 | 143 | 0.44 | 0.20 | **0.307** | Best avg p1. Running. |
| 195634_2 | 8 | 85 | 0.36 | 0.18 | 0.265 | Running. |
| 195634_1 | 16 | 46 | 0.30 | 0.14 | 0.183 | Running. |

### Delta action sweep (abs_action=false, zarr=multimodal_push_seed.zarr)

| Job ID | Config | Epoch | Best p1 | Best p2 | Avg p1 (last 20) | Notes |
|--------|--------|-------|---------|---------|-------------------|-------|
| 195620 | gpl=4 | 713 | 0.38 | 0.18 | 0.232 | Running. |
| 195621 | gpl=16 | 59 | 0.18 | 0.06 | 0.108 | Running. |
| 195623_0 | bs256/e256/cosine (gpl=8) | 412 | 0.26 | 0.12 | 0.168 | Running. |
| 195623_1 | bs512/e256/cosine (gpl=8) | 101 | 0.24 | 0.08 | 0.147 | Running. |
| 195623_2 | bs256/e256/warmup (gpl=8) | 96 | 0.24 | 0.10 | 0.152 | Running. |
| 195623_3 | bs256/e128/cosine (gpl=8) | 97 | 0.26 | 0.10 | 0.128 | Running. |
| 195623_4 | bs128/e256/cosine (gpl=8) | 90 | 0.24 | 0.12 | 0.147 | Running. |
| 195623_5 | bs512/e256/warmup (gpl=8) | 101 | 0.16 | 0.08 | 0.089 | Running. |

### Key finding

**abs_action=True is critical for BlockPush.** All abs runs outperform delta runs at equivalent epochs. abs_gpl2 hit the paper p1 target (0.56) and far exceeded p2 (0.30 vs 0.16). Smaller gen_per_label (2, 4) works best.

Reproduce best result:
```bash
python train.py --config-dir=. --config-name=drifting_blockpush_lowdim.yaml \
    task.dataset.zarr_path=data/block_pushing/multimodal_push_seed_abs.zarr \
    task.env_runner.abs_action=true \
    policy.gen_per_label=2 \
    training.rollout_every=1 training.seed=42 training.device=cuda:0
```
