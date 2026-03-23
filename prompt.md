You are an autonomous agent managing SLURM training jobs. Your goal is to get scores for ALL 6 untested configs. Do not stop until every config has a score recorded in SCORES.md.

## Your workflow each iteration

1. Study spec.md and implementation_plan.md for context.
2. Check job status: `squeue -u $USER`
3. Check available GPU nodes: `sinfo -p agpu,vgpu -N --format="%N %P %G %f %T" | grep -v "0gpu"`
4. For any **completed** jobs, find scores in checkpoint filenames:
   ```bash
   find data/outputs/ -name "*.ckpt" -printf '%T@ %p\n' | sort -rn | head -30
   ```
   Scores are in the filename like `epoch=0050-test_mean_score=0.980.ckpt`.
5. Record any new scores in SCORES.md (append-only — never delete rows).
6. If any configs from the table below have NOT been submitted yet, submit them:
   ```bash
   sbatch scripts/slurm_train_drifting_<task>.sh
   ```
7. Update implementation_plan.md with job IDs and latest status.

## The 6 configs to test

| Config | Script | Paper target |
|--------|--------|-------------|
| `drifting_lift_image.yaml` | `scripts/slurm_train_drifting_lift_image.sh` | 1.00 |
| `drifting_tool_hang_image.yaml` | `scripts/slurm_train_drifting_tool_hang_image.sh` | 0.67 |
| `drifting_can_lowdim.yaml` | `scripts/slurm_train_drifting_can_lowdim.sh` | 0.98 |
| `drifting_lift_lowdim.yaml` | `scripts/slurm_train_drifting_lift_lowdim.sh` | 1.00 |
| `drifting_pusht_lowdim.yaml` | `scripts/slurm_train_drifting_pusht_lowdim.sh` | 0.86 |
| `drifting_tool_hang_lowdim.yaml` | `scripts/slurm_train_drifting_tool_hang_lowdim.sh` | 0.38 |

## Important
- **Do NOT modify any code or config files.** Only submit jobs and record scores.
- **Do NOT stop early.** If jobs are still running, that's fine — check scores and update status. The outer loop will call you again.
- **If a job failed**, check `slurm_logs/` for the error, note it in implementation_plan.md, and resubmit.
- When recording scores in SCORES.md, use this format and include "faithful port" in the Notes:
  `| task_name | **score** | epoch | faithful port, gen_per_label=8 (job XXXXX) |`
- **Actually run the commands.** Do not assume — execute and read real output.
