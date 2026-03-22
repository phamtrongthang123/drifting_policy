# Implementation Plan: Port Official Drifting Loss

## Goal Status
~0% of final goal. Nothing ported yet. Critical next milestone: Phase 1 (port loss function) + Phase 3 (numerical test).

## Key Differences: Official (`drifting/drift_loss.py`) vs. Ours (`drifting_util.py`)

### 1. Concatenated target pool
- **Official**: `targets = concat([old_gen, fixed_neg, fixed_pos])` — gen samples are in the target set. Eye mask (`jnp.eye(C_g)` padded to full width) blocks gen-to-self attention. Single unified cdist.
- **Ours**: `y_neg = x` passed separately. Two `cdist` calls. Separate softmax over pos and neg.

### 2. Scaling / normalization
- **Official**: `dist = cdist(old_gen, targets)` → `scale = mean(weighted_dist) / mean(weights)`. Coords normalized by `scale / sqrt(S)`. Distances normalized by `scale`.
- **Ours**: `S_j = mean(cdist(x, y_pos)) / sqrt(D)`. Coords normalized by `S_j`.

### 3. Force computation
- **Official**: Affinity over full target matrix. `R_coeff = concat([-aff_neg * sum_pos, aff_pos * sum_neg])`. Force = `einsum("biy,byx->bix", R_coeff, targets_scaled) - total_coeffs * old_gen_scaled` (explicit zero-centering).
- **Ours**: `V = W_pos @ y_pos - W_neg @ y_neg` (no zero-centering term).

### 4. Force normalization
- **Official**: `force_scale = sqrt(mean(force^2))` — scalar over entire tensor.
- **Ours**: `lambda_j = sqrt(mean(sum(V^2, dim=-1)) / D)` — different reduction.

### 5. Loss computation
- **Official**: Entire goal computation in `stop_gradient`. `loss = mean((gen_scaled - goal_scaled)^2, axis=(-1,-2))` returns per-batch loss.
- **Ours**: `target = (x_norm + V_total).detach()`. `F.mse_loss(x_norm, target)` returns scalar.

### 6. Weights
- **Official**: Per-sample weights modulate distances and affinities. Default to ones.
- **Ours**: No weight support. Use uniform weights (ones) — functionally equivalent for diffusion policy.

---

## Phase 1: Port the loss function
**File**: `diffusion_policy/model/drifting/drifting_util.py`

Rewrite to a single function mirroring `drifting/drift_loss.py`:
```
drift_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None, weight_pos=None, weight_neg=None, R_list=(0.02, 0.05, 0.2))
```
- Input shape: `[B, C, S]` (B=batch, C=num samples, S=feature dim)
- Steps: concat targets → cdist → scale normalization → eye mask → for each R: softmax affinity (row+col geometric mean) → split neg/pos → compute R_coeff → einsum force → zero-center → normalize by `sqrt(mean(force^2))` → accumulate → goal = old_gen_scaled + force → MSE loss
- Use `torch.no_grad()` for goal computation (= `jax.lax.stop_gradient`)
- Return `(loss, info_dict)` with `scale` and per-R `loss_{R}` keys

**Keep old functions** (`compute_V`, `compute_drifting_loss`) until Phase 3 passes, then delete.

## Phase 2: Adapt the call site
**File**: `diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`

- `per_timestep_loss=True`: reshape `pred_actions[:, t, :]` from `[B, D]` → `[B, 1, D]`, same for expert actions. Pass as `gen` and `fixed_pos`. `fixed_neg=None` (official handles gen-as-neg internally).
- `per_timestep_loss=False`: reshape flat `[B, T*D]` → `[B, 1, T*D]`.
- `bc_coeff`: keep config key, but set to 0. Code path is a no-op at 0.

## Phase 3: Numerical test (CRITICAL)
**File**: `tests/test_drift_loss_port.py`

- Random tensors → convert to JAX arrays → run official `drift_loss` and ported version → `allclose(rtol=1e-4)` on loss and info values.
- Cases: `fixed_neg=None`, C_g=1, C_g=4, varying R_list.
- If JAX is not available in test env, pre-compute JAX outputs and hardcode expected values.

## Phase 4: Train and validate
- Set `bc_coeff: 0.0` in `drifting_can_image.yaml`
- `sbatch scripts/slurm_train_drifting_can_image.sh`
- Check `train_action_mse_error` decreasing, `test/mean_score > 0` at epoch 50, target >= 0.90 by epoch 200
- If epoch 50 score = 0: dump intermediate values (scale, force magnitudes) and compare against official on same batch
