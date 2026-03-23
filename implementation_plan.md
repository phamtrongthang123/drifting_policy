# Implementation Plan: Evaluate All Tasks with Faithful Drifting Loss

## Goal Status
**Phase 2: Mass evaluation.** Loss port is complete and verified. Now running all remaining task/setting combos.

### Verified results (faithful port)
| Task | Score | Paper | Status |
|------|-------|-------|--------|
| can_image | **0.98** | 0.99 | ✅ Done |
| pusht_image | **0.86** | 0.86 | ✅ Done — matches paper |

## Next Step
1. Submit SLURM jobs for all 6 untested configs (see table below).
2. Monitor checkpoints for scores at ep50, ep100, etc.
3. Record results in SCORES.md as they come in.

---

## Jobs to Submit

Submit in priority order. Check `sinfo -p agpu,vgpu -N --format="%N %P %G %f %T" | grep -v "0gpu"` for available nodes first.

| # | Config | Script | Paper target | Epochs | Notes |
|---|--------|--------|-------------|--------|-------|
| 1 | `drifting_lift_image.yaml` | `scripts/slurm_train_drifting_lift_image.sh` | 1.00 | 300 | Previous best 0.92 (old code, batch=256). Now batch=64, gen_per_label=8. |
| 2 | `drifting_tool_hang_image.yaml` | `scripts/slurm_train_drifting_tool_hang_image.sh` | 0.67 | 300 | Previously diverged. Now batch=64, gen_per_label=8. |
| 3 | `drifting_can_lowdim.yaml` | `scripts/slurm_train_drifting_can_lowdim.sh` | 0.98 | 300 | |
| 4 | `drifting_lift_lowdim.yaml` | `scripts/slurm_train_drifting_lift_lowdim.sh` | 1.00 | 500 | |
| 5 | `drifting_pusht_lowdim.yaml` | `scripts/slurm_train_drifting_pusht_lowdim.sh` | 0.86 | 300 | Previous best 0.819 (old code). |
| 6 | `drifting_tool_hang_lowdim.yaml` | `scripts/slurm_train_drifting_tool_hang_lowdim.sh` | 0.38 | 300 | Previously diverged. |

### How to submit
```bash
cd /scrfs/storage/tp030/home/drifting_policy
sbatch scripts/slurm_train_drifting_lift_image.sh
sbatch scripts/slurm_train_drifting_tool_hang_image.sh
sbatch scripts/slurm_train_drifting_can_lowdim.sh
sbatch scripts/slurm_train_drifting_lift_lowdim.sh
sbatch scripts/slurm_train_drifting_pusht_lowdim.sh
sbatch scripts/slurm_train_drifting_tool_hang_lowdim.sh
```

### How to monitor
```bash
# Job status
squeue -u $USER

# Find scores in checkpoint filenames
find data/outputs/ -name "*.ckpt" -newer SCORES.md | sort

# Check a specific job's output
tail -50 slurm_logs/train_drifting_*_<JOBID>.out
```

---

## Active Jobs

| Job ID | Task | Config | Status |
|--------|------|--------|--------|
| — | — | — | No active jobs yet |

---

## Completed Work

| Phase | What was built | Constraints / gotchas |
|-------|---------------|----------------------|
| 1. Port loss | `drifting_util.py`: `drift_loss()` + `_cdist()`, 1:1 from JAX. | Dot-product cdist (not `torch.cdist`). Returns `loss: [B]`; call site `.mean()`s it. |
| 2. Image policy | `drifting_unet_hybrid_image_policy.py`: gen_per_label=8, batch_size=64, per_timestep=True | gen_per_label>1 required (self-mask makes C_g=1 degenerate). |
| 3. Lowdim policy | `drifting_unet_lowdim_policy.py`: same config as image. All 4 lowdim configs ready. | CPU-tested only. |
| 4. Numerical test | `tests/test_drift_loss_port.py`: 7 tests, rtol=2e-4 vs JAX | Always run before GPU jobs. |
| 5. Training (can) | **0.98@ep50** | Loss ~8-9 is normal. Monitor MSE, not loss. |
| 6. Training (pusht) | **0.86** — matches paper | |
| 7. Config cleanup | All configs: gen_per_label=8, batch_size=64, epochs capped at 200-500. bc_coeff removed. | |

---

## Lessons Learned

1. **gen_per_label > 1 is essential.** Self-mask makes C_g=1 degenerate (zero force). Official uses 8.
2. **batch_size=64 × gen_per_label=8 = 512 total UNet calls.** Same compute as old batch_size=512.
3. **Port cdist exactly.** Dot-product formula with eps=1e-8, not `torch.cdist`.
4. **Loss ~8-9 is normal.** Force-magnitude-normalized ≈ len(R_list)². Monitor MSE.
5. **Scores in checkpoint filenames** (e.g. `epoch=0050-test_mean_score=0.980.ckpt`), not logs.
6. **Set TMPDIR in slurm scripts.** Login node /tmp fills up. Use `export TMPDIR="$HOME/.tmp"`.
7. **CPU numerical validation works.** JAX+PyTorch comparison catches bugs before expensive GPU runs.

## Debugging Rule
**Never guess — print liberally.** Add print statements when debugging. Remove after fix is confirmed.
