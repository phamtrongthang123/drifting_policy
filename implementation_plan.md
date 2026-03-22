# Implementation Plan: Port Official Drifting Loss

## Goal Status
~0% of final goal. Nothing ported yet. Next step: Phase 1.

## Next step
Phase 1 → Phase 3 → Phase 2 → Phase 4. Do Phase 1 and 3 together (port + test), then wire it up (Phase 2), then train (Phase 4).

---

## Phase 1: Port the loss function
**Source**: `drifting/drift_loss.py` (JAX, 135 lines)
**Target**: `diffusion_policy/model/drifting/drifting_util.py` (PyTorch)

Line-by-line port of `drift_loss()`. The PyTorch function must have the **exact same signature, logic, and math**. JAX→PyTorch translations:
- `jnp` → `torch`, `jnp.clip` → `torch.clamp`, `jnp.concatenate` → `torch.cat`
- `jax.nn.softmax` → `torch.softmax`
- `jnp.einsum` → `torch.einsum`
- `jax.lax.stop_gradient(...)` → wrap in `with torch.no_grad():` and `.detach()` the outputs
- `jnp.eye` → `torch.eye`, `jnp.pad` → `torch.nn.functional.pad`
- Official `cdist` (lines 6-12) uses manual dot-product formula. Use `torch.cdist` (L2 distance) which is equivalent.
- Official returns `loss` as `[B]` (per-batch). Keep that — the call site will `.mean()` it.

**Do not**:
- Add extra normalization, helper functions, or alternative code paths
- Keep the old `compute_V` / `compute_drifting_loss` — delete them after Phase 3 passes

## Phase 2: Adapt the call site
**File**: `diffusion_policy/policy/drifting_unet_hybrid_image_policy.py` (`compute_loss` method, ~line 181)

The official `drift_loss` expects `[B, C, S]` (batch, num_samples, feature_dim). Our policy produces `[B, T, D]` (batch, timesteps, action_dim).

- **`per_timestep_loss=True`**: for each timestep t:
  - `gen = pred_actions[:, t, :].unsqueeze(1)` → `[B, 1, D]`
  - `fixed_pos = nactions[:, t, :].unsqueeze(1)` → `[B, 1, D]`
  - `loss_t, info_t = drift_loss(gen, fixed_pos, R_list=tuple(self.temperatures))`
  - `loss_t` is `[B]` → `.mean()` then accumulate
- **`per_timestep_loss=False`**:
  - `gen = pred_actions.reshape(B, 1, -1)` → `[B, 1, T*D]`
  - `fixed_pos = nactions.reshape(B, 1, -1)` → `[B, 1, T*D]`
- `fixed_neg`: pass `None`. The official code handles gen-as-negatives internally.
- `bc_coeff`: keep config key. Code path already no-ops at 0.
- Set `bc_coeff: 0.0` in `drifting_can_image.yaml`.

## Phase 3: Numerical test (CRITICAL — do this right after Phase 1)
**File**: `tests/test_drift_loss_port.py`

1. Check if JAX is available (`try: import jax`). If yes:
   - Create random tensors with fixed seed, run both official and ported, `assert allclose(rtol=1e-4)`.
2. If JAX not available:
   - Pre-compute expected outputs from JAX on a machine that has it, hardcode them in the test.
3. Test cases:
   - `fixed_neg=None`, `C_g=1, C_p=1, S=7` (single sample, matches per_timestep_loss=True)
   - `C_g=4, C_p=4, C_n=0, S=16`
   - `C_g=4, C_p=4, C_n=2, S=16` (with explicit negatives)
   - Different `R_list` values

## Phase 4: Train and validate
- `sbatch scripts/slurm_train_drifting_can_image.sh` (bc_coeff=0, 200 epochs)
- Monitor `train_action_mse_error` (should decrease) and `test/mean_score` at epoch 50 (should be > 0)
- Record final score in `SCORES.md`
- **The score is what it is.** If the port is faithful (Phase 3 passes) and the score is low, report it honestly.
