# Implementation Plan: Port Official Drifting Loss

## Goal Status
**DONE.** can_image score=0.98 at epoch 50 with bc_coeff=0. Exceeds target (>=0.90).

Job 195490 still running (200 epochs total). Monitoring for later checkpoints.

---

## Phase 1: Port the loss function — DONE
**Source**: `drifting/drift_loss.py` (JAX, 135 lines)
**Target**: `diffusion_policy/model/drifting/drifting_util.py` (PyTorch)

Line-by-line port of `drift_loss()`. The PyTorch function has the **exact same signature, logic, and math**. JAX→PyTorch translations:
- `jnp` → `torch`, `jnp.clip` → `torch.clamp`, `jnp.concatenate` → `torch.cat`
- `jax.nn.softmax` → `torch.softmax`
- `jnp.einsum` → `torch.einsum`
- `jax.lax.stop_gradient(...)` → `with torch.no_grad():` + `.detach()`
- `jnp.eye` → `torch.eye`, `jnp.pad` → `torch.nn.functional.pad`
- Ported official `cdist` dot-product formula exactly (NOT `torch.cdist`) for numerical match
- Official returns `loss` as `[B]` (per-batch). Call site `.mean()`s it.

Old `compute_V` / `compute_drifting_loss` deleted.

## Phase 2: Adapt the call site — DONE
**File**: `diffusion_policy/policy/drifting_unet_hybrid_image_policy.py` (`compute_loss` method)

### Critical lesson: gen_per_label > 1 is REQUIRED

The official `drift_loss` expects `[B, C_g, S]` where C_g > 1. With C_g=1, the self-mask eliminates
the only negative sample, making force exactly zero — no learning signal.

The official `train.py` uses `gen_per_label=8`: it generates 8 noise samples per context, each
getting an independent UNet forward pass. The drift loss then operates per-context (B=batch_size)
with C_g=8 generated samples and C_p=1 real sample per context.

**What DOESN'T work** (and why):
- `[B, 1, D]` per observation (C_g=1): zero force, no learning
- `[1, B, D]` mixing all observations (C_g=B): cross-observation interference,
  MSE plateaus at 0.246 (model learns marginal mean, not conditional mapping).
  The drift force pushes gen_i toward the nearest pos_j from ANY observation,
  creating incoherent gradients for the observation-conditional UNet.

**What works**:
- `[B, gen_per_label, D]` per observation: each context gets its own set of
  generated samples. The kernel pushes gen samples toward the correct positive
  (same observation) and apart from each other (diversity within context).
- Requires batch_size * gen_per_label UNet forward passes per training step.
  Use batch_size=64, gen_per_label=8 for same total compute as batch_size=512.

### Config changes
- `bc_coeff: 0.0` (no BC crutch)
- `gen_per_label: 8`
- `batch_size: 64` (64 × 8 = 512 total UNet calls)
- `num_epochs: 200`

## Phase 3: Numerical test — DONE (7/7 pass)
**File**: `tests/test_drift_loss_port.py`

JAX installed in robodiff env. All 7 tests pass with rtol=2e-4, atol=1e-4 (float32 platform noise):
1. `C_g=1, C_p=1, S=7` — single sample, degenerate case
2. `C_g=4, C_p=4, S=16` — multi-sample, no explicit neg
3. `C_g=4, C_p=4, C_n=2, S=16` — with explicit negatives
4. Different R_list values
5. Training shape (`[1, 32, 7]`)
6. Gradient flow verification
7. Large batch NaN/Inf smoke test

## Phase 4: Train and validate — DONE
**Job 195490** on agpu (A100), `drifting_can_image.yaml`, bc_coeff=0, 200 epochs.

### Results
| Epoch | MSE        | Score  |
|-------|------------|--------|
| 0     | 0.0418     | 0.0200 |
| 5     | 0.0262     | —      |
| 10    | 0.0194     | —      |
| 15    | 0.0121     | —      |
| 20    | 0.0085     | —      |
| 25    | 0.0056     | —      |
| 30    | 0.0044     | —      |
| 35    | 0.0036     | —      |
| 40    | 0.0030     | —      |
| 45    | 0.0027     | —      |
| 50    | **0.0025** | **0.98** |
| 55    | 0.0020     | —      |

**Success criteria met**: score >= 0.90 within 200 epochs, bc_coeff=0.

---

## Lessons Learned

1. **gen_per_label > 1 is essential.** The drift loss self-mask makes C_g=1 degenerate (zero force).
   The official codebase uses gen_per_label=8 by default. Without this, the loss has no learning signal.

2. **Don't mix observations in the B=1 approach.** Using `[1, batch_size, D]` treats all observations
   as one context, causing the kernel to push generated actions toward nearest-neighbor positives from
   wrong observations. MSE plateaus because the UNet receives incoherent gradients.

3. **The official cdist uses dot-product formula with eps=1e-8 clamp.** Porting this exactly
   (not using `torch.cdist`) ensures numerical match. The max relative difference is ~0.013%
   in the scale info, well within float32 tolerance.

4. **Loss value ~8-9 is normal and does NOT indicate failure.** The drift loss is force-magnitude-normalized,
   so the value stays roughly constant (≈ len(R_list)²) regardless of convergence. Monitor MSE instead.

5. **Testing on login node (CPU) works for numerical validation.** JAX (CPU) + PyTorch (CPU) comparison
   is fast (~2s for 7 tests) and catches bugs before expensive GPU training.

6. **batch_size=64 with gen_per_label=8 gives same total compute** as batch_size=512 with gen_per_label=1.
   The per-observation structure is worth the same wall-clock cost.
