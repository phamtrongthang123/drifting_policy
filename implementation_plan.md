# Implementation Plan: Evaluate All Tasks with Faithful Drifting Loss

## Goal Status
**Phase 2: Mass evaluation.** Loss port is complete and frozen. 4/8 tasks scored, 4 still running.

### Results (faithful port)
| Task | Best Score | Paper | Status |
|------|-----------|-------|--------|
| can_image | **0.98** | 0.99 | ✅ Done (stable ep50-150) |
| pusht_image | **0.862** | 0.86 | ✅ Done — matches paper |
| can_lowdim | **0.980** | 0.98 | ✅ Done (job 195505, ep50) — matches paper |
| lift_lowdim | **1.000** | 1.00 | ✅ Done (job 195501, ep50) — matches paper |
| lift_image | **1.000** | 1.00 | ✅ Done (job 195504, ep50) — matches paper |
| pusht_lowdim | **0.871** | 0.86 | ✅ Done (job 195506, ep50) — exceeds paper |
| tool_hang_image | **0.000** | 0.67 | Running (job 195514, ep0 only so far) |
| tool_hang_lowdim | **0.840** | 0.38 | ✅ Done (job 195515, ep50) — far exceeds paper |

## Next Steps
1. ~~BLOCKER: ToolHang env missing~~ **RESOLVED** — installed robosuite from cheng-chi/robosuite@offline_study (has ToolHang + mujoco_py). Failed attempts: robosuite 1.4.1 (new mujoco, EGL rendering incompatible).
2. Monitor remaining running jobs (lift_image, pusht_lowdim, tool_hang_image, tool_hang_lowdim) for ep50+ checkpoints.
3. Record scores in SCORES.md as checkpoints arrive.

---

## Active Jobs (2026-03-22 ~20:40)

| Job ID | Task | Node | Latest Checkpoint | Status |
|--------|------|------|-------------------|--------|
| 195501 | lift_lowdim | c1706 | ep50 score=1.000 | ✅ Done |
| 195504 | lift_image | c1708 | ep0 score=0.900 | Running, waiting for ep50 |
| 195505 | can_lowdim | c1709 | ep50 score=0.980 | ✅ Done |
| 195506 | pusht_lowdim | c1707 | ep0 score=0.216 | Running, waiting for ep50 |
| 195514 | tool_hang_image | c2008 | ep0 score=0.000 | Running, waiting for ep50 |
| 195515 | tool_hang_lowdim | c2108 | ep0 score=0.000 | Running, waiting for ep50 |

Completed: 195490 (can_image, 0.98), 195496 (pusht_image, 0.862).
Cancelled: 195499, 195503, 195507-195513 (tool_hang — various robosuite issues).

---

## Completed Work (collapsed)

Loss port, image policy, lowdim policy, numerical tests, config cleanup — all done. See git history.
- Key configs: gen_per_label=8, batch_size=64, per_timestep=True, bc_coeff removed.
- Scores in checkpoint filenames (e.g. `epoch=0050-test_mean_score=0.980.ckpt`).
- Loss ~8-9 is normal. Monitor MSE, not loss.
- Set TMPDIR in slurm scripts to avoid /tmp fills.

## Monitoring
```bash
squeue -u $USER
find data/outputs/ -name "*.ckpt" -newer SCORES.md | sort
tail -c 2000 slurm_logs/train_drifting_*_<JOBID>.out
```
