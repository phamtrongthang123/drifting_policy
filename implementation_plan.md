# Implementation Plan: Port Official Drifting Loss

## Goal Status
**~95% complete.** Port is faithful (7/7 numerical tests pass). can_image score=0.98@ep50, bc_coeff=0 — exceeds 0.90 target.
**Remaining:** (1) Job 195490 (can_image) in progress — need ep100/150/200 scores. (2) Job 195496 (pusht_image) in progress — need ep50 score >= 0.78.

## Next Step
Monitor both running jobs. Record results in SCORES.md when rollouts complete.

---

## Active Jobs

| Job ID | Task | Node | Config | Status |
|--------|------|------|--------|--------|
| 195490 | can_image | c2008 (A100) | official port, bc_coeff=0, gen_per_label=8, 200ep | Running, 0.98@ep50. Currently ~epoch 68. |
| 195496 | pusht_image | c2108 (A100) | official port, bc_coeff=0, gen_per_label=8, 300ep | Running, started epoch 0. |

---

## Completed Work

| Phase | What was built | Constraints / gotchas |
|-------|---------------|----------------------|
| 1. Port loss | `drifting_util.py`: `drift_loss()` + `_cdist()`, 1:1 from JAX. Old `compute_V`/`compute_drifting_loss` deleted. | Uses dot-product cdist (not `torch.cdist`). Returns `loss: [B]`; call site `.mean()`s it. |
| 2. Call site | `drifting_unet_hybrid_image_policy.py`: gen_per_label=8, batch_size=64, per_timestep=True, bc_coeff=0 | gen_per_label>1 required (self-mask makes C_g=1 degenerate). Never mix observations across B dim. |
| 3. Numerical test | `tests/test_drift_loss_port.py`: 7 tests, rtol=2e-4, atol=1e-4 vs JAX | JAX installed in robodiff env. Tests run on CPU ~2s. |
| 4. Training (can) | Job 195490 (agpu/A100): **0.98@ep50**, MSE=0.0025, bc_coeff=0 | Loss ~8-9 is normal (force-normalized). Monitor MSE, not loss value. |
| 5. Cleanup | Deleted 9 stale test files that imported removed functions. | Valid tests: `test_drift_loss_port.py` (7 tests), `test_config_epochs.py`. |
| 6. pusht_image config | Updated `drifting_pusht_image.yaml`: bc_coeff=0, batch_size=64, gen_per_label=8 | Same settings as can_image. Job 195496 submitted. |

---

## Remaining Tasks

### Task 1: Record job 195490 (can_image) final results
When job completes, add epoch 100/150/200 scores to SCORES.md.

### Task 2: Record job 195496 (pusht_image) results
Spec criterion 3: "No regression on pusht_image (should still get >= 0.78)".
First rollout at epoch 50. Record in SCORES.md.

---

## Lessons Learned

1. **gen_per_label > 1 is essential.** Self-mask makes C_g=1 degenerate (zero force). Official uses 8.
2. **Don't mix observations in B=1.** `[1, batch_size, D]` causes cross-obs interference → MSE plateau at 0.246.
3. **Port cdist exactly.** Dot-product formula with eps=1e-8, not `torch.cdist`.
4. **Loss ~8-9 is normal.** Force-magnitude-normalized ≈ len(R_list)². Monitor MSE.
5. **batch_size=64 × gen_per_label=8 = 512 total UNet calls.** Same compute as old batch_size=512.
6. **CPU numerical validation works.** JAX+PyTorch comparison in ~2s catches bugs before expensive GPU runs.
7. **Set TMPDIR in slurm scripts.** Login node /tmp can fill up. Use `export TMPDIR="$HOME/.tmp"`.
8. **Slurm log output is dominated by tqdm progress bars.** Scores appear in checkpoint filenames (e.g. `epoch=0050-test_mean_score=0.980.ckpt`), not easily greppable from logs. Check `checkpoints/` directory for results.

## Debugging Rule
**Never guess — print liberally.** Add print statements when debugging. Remove after fix is confirmed.
