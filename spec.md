# Spec: Evaluate Faithful Drifting Loss Across All Tasks

## FINAL GOAL
Run the **faithful drifting loss port** (already verified on can_image=0.98, pusht_image=0.86) on **all remaining task/setting combinations** and record scores. The loss code is frozen — do not modify `drifting_util.py` or the policies.

## Reference
- Official drifting codebase (JAX): `/home/tp030/drifting_policy/drifting/drift_loss.py`
- Our drifting util (faithful port): `diffusion_policy/model/drifting/drifting_util.py`
- Image policy: `diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`
- Lowdim policy: `diffusion_policy/policy/drifting_unet_lowdim_policy.py`

## What to run
Submit SLURM jobs for every config that hasn't been tested with the faithful port yet:

| Priority | Config | Script | Paper target |
|----------|--------|--------|-------------|
| 1 | `drifting_lift_image.yaml` | `scripts/slurm_train_drifting_lift_image.sh` | 1.00 |
| 2 | `drifting_tool_hang_image.yaml` | `scripts/slurm_train_drifting_tool_hang_image.sh` | 0.67 |
| 3 | `drifting_can_lowdim.yaml` | `scripts/slurm_train_drifting_can_lowdim.sh` | 0.98 |
| 4 | `drifting_lift_lowdim.yaml` | `scripts/slurm_train_drifting_lift_lowdim.sh` | 1.00 |
| 5 | `drifting_pusht_lowdim.yaml` | `scripts/slurm_train_drifting_pusht_lowdim.sh` | 0.86 |
| 6 | `drifting_tool_hang_lowdim.yaml` | `scripts/slurm_train_drifting_tool_hang_lowdim.sh` | 0.38 |

## Constraints
- **SLURM partitions:** Only use `vgpu` and `agpu` partitions.
- **Do NOT modify the loss or policies.** The code is frozen. Only submit jobs and record scores.
- **Num epochs:** 200-300 max. All configs already capped.
- **batch_size=64, gen_per_label=8** for all tasks (matching verified can/pusht configs).
- **Login node /tmp can fill up** — Set `export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR"` in all scripts and shells.
- **Agent on login node, GPU work via `sbatch`**
- **GPU partitions** available:
  - `agpu` — nodes c1912, c2008, c2108, c2110 (1×A100, 1024GB RAM, public)
  - `vgpu` — nodes c1612, c1706-c1714 (1×V100, 192GB RAM, public)
  - Before submitting: `sinfo -p agpu,vgpu -N --format="%N %P %G %f %T" | grep -v "0gpu"`
  - **Never wait on pending jobs.** If stuck, cancel and resubmit to a partition with idle nodes.
- **Max 72 hours per job**
- **Commit after every meaningful change.** Concise messages.

## Success Criteria
1. All 6 configs above have at least one score recorded in SCORES.md
2. Scores are from the faithful port (gen_per_label=8, batch_size=64)
3. README.md updated with final results

## Monitoring
- Scores are in checkpoint filenames: `epoch=0050-test_mean_score=0.980.ckpt`
- Check with: `ls data/outputs/*/checkpoints/ | grep -oP 'test_mean_score=[\d.]+' | sort`
- `squeue -u $USER` to check job status
- If a job fails, check `slurm_logs/` for the error
