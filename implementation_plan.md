# Implementation Plan: Port Official Drifting Loss

## Goal Status
**CORE SPEC COMPLETE — all 3 success criteria met:**
1. `tests/test_drift_loss_port.py` passes (7/7) — numerics match JAX ✅
2. can_image `test/mean_score = 0.98` at ep50 and ep100, bc_coeff=0 — exceeds 0.90 target ✅
3. pusht_image `test/mean_score = 0.803` at ep50, bc_coeff=0 — no regression (>= 0.78) ✅

Port verified line-by-line (2026-03-22): every line of `drifting/drift_loss.py` has a corresponding line in `drifting_util.py`. No omissions, no additions, no "improvements."

## Next Step
1. **Commit lowdim changes** — policy + 4 configs + SCORES.md are modified but uncommitted.
2. Record final scores from running jobs when they complete.

---

## Active Jobs (verified running 2026-03-22)

| Job ID | Task | Config | Status |
|--------|------|--------|--------|
| 195490 | can_image | official port, bc_coeff=0, gen_per_label=8, 200ep | 0.98@ep50, 0.98@ep100. Running. |
| 195496 | pusht_image | official port, bc_coeff=0, gen_per_label=8, 300ep | 0.803@ep50. Running. |

---

## Completed Work

| Phase | What was built | Constraints / gotchas |
|-------|---------------|----------------------|
| 1. Port loss | `drifting_util.py`: `drift_loss()` + `_cdist()`, 1:1 from JAX. Old functions deleted. | Dot-product cdist (not `torch.cdist`). Returns `loss: [B]`; call site `.mean()`s it. |
| 2. Call site (image) | `drifting_unet_hybrid_image_policy.py`: gen_per_label=8, batch_size=64, per_timestep=True, bc_coeff=0 | gen_per_label>1 required (self-mask makes C_g=1 degenerate). |
| 3. Numerical test | `tests/test_drift_loss_port.py`: 7 tests, rtol=2e-4, atol=1e-4 vs JAX | Tests run on CPU ~2s. Always run before GPU jobs. |
| 4. Training (can) | Job 195490: **0.98@ep50**, **0.98@ep100**, MSE=0.0025 | Loss ~8-9 is normal (force-normalized). Monitor MSE, not loss. |
| 5. Cleanup | Deleted 9 stale test files. | Valid tests: `test_drift_loss_port.py` (7), `test_config_epochs.py` (4). |
| 6. Training (pusht) | Job 195496: **0.803@ep50** | Configs: bc_coeff=0, batch_size=64, gen_per_label=8. |
| 7. Lowdim policy | `drifting_unet_lowdim_policy.py`: updated to use `drift_loss()` with gen_per_label=8. All 4 lowdim configs updated. | **UNCOMMITTED.** CPU-tested only — no persistent test file. |

---

## Remaining Tasks

### Task 1: Commit lowdim changes
Stage and commit: `drifting_unet_lowdim_policy.py`, 4 lowdim yaml configs, `SCORES.md`.

### Task 2: Record final job results
When jobs complete, add remaining epoch scores to SCORES.md:
- Job 195490 (can): ep150, ep200
- Job 195496 (pusht): ep100, ep150, ep200, ep250, ep300

### Task 3: Cleanup (not blocking)
- **Remove dead `bc_coeff` param** from `drifting_unet_hybrid_image_policy.py` (line 33, 146). Stored but never used in `compute_loss`.
- **Delete `drifting_model_debug.md`** — references deleted functions.
- **Cap lowdim epochs.** can/pusht/tool_hang have `num_epochs: 3050`, lift has 500. Spec says 200-300 max. Reduce and optionally add lowdim configs to `test_config_epochs.py`.

---

## Lessons Learned

1. **gen_per_label > 1 is essential.** Self-mask makes C_g=1 degenerate (zero force). Official uses 8.
2. **Don't mix observations in B=1.** `[1, batch_size, D]` causes cross-obs interference → MSE plateau.
3. **Port cdist exactly.** Dot-product formula with eps=1e-8, not `torch.cdist`.
4. **Loss ~8-9 is normal.** Force-magnitude-normalized ≈ len(R_list)². Monitor MSE.
5. **batch_size=64 × gen_per_label=8 = 512 total UNet calls.** Same compute as old batch_size=512.
6. **CPU numerical validation works.** JAX+PyTorch comparison catches bugs before expensive GPU runs.
7. **Set TMPDIR in slurm scripts.** Login node /tmp fills up. Use `export TMPDIR="$HOME/.tmp"`.
8. **Scores in checkpoint filenames** (e.g. `epoch=0050-test_mean_score=0.980.ckpt`), not logs.
9. **Lowdim policy can be tested on CPU** — catches import/shape bugs without GPU.

## Debugging Rule
**Never guess — print liberally.** Add print statements when debugging. Remove after fix is confirmed.
