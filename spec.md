# Spec: Port Official Drifting Loss to Diffusion Policy

## FINAL GOAL
Replace our hand-written drifting loss with a **faithful PyTorch port of the official `drifting/drift_loss.py`**. After the swap, achieve **test/mean_score >= 0.90 on can_image** within 200 epochs, **with bc_coeff=0** (no BC crutch).

## Reference
- Official drifting codebase (JAX): `/home/tp030/drifting_policy/drifting/drift_loss.py`
- Our current drifting util: `diffusion_policy/model/drifting/drifting_util.py`
- Policy that calls it: `diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`
- Original diffusion policy: `/home/tp030/drifting_policy/original_diff/diffusion_policy`

## Constraints
- **SLURM partitions:** Only use `vgpu` and `agpu` partitions.
- **No BC loss as a crutch:** The official code has no bc_loss. Once the port is correct, bc_coeff must be 0.
- **Num epochs:** 200-300 max. DDPM converges in ~150; drifting should be comparable.
- **Do not touch anything outside the loss.** The U-Net, obs encoder, dataset, workspace, and config structure stay as-is. Only `drifting_util.py` and its call site in the policy change.
- **Login node /tmp can fill up** — Set `export TMPDIR="$HOME/.tmp" && mkdir -p "$TMPDIR"` in all scripts and shells.
- **Agent on login node, GPU work via `sbatch`**
- **GPU partitions** available:
  - `agpu` — nodes c1912, c2008, c2108, c2110 (1×A100, 1024GB RAM, public)
  - `vgpu` — nodes c1612, c1706-c1714 (1×V100, 192GB RAM, public)
  - Before submitting: `sinfo -p agpu,vgpu -N --format="%N %P %G %f %T" | grep -v "0gpu"`
  - **Never wait on pending jobs.** If stuck, cancel and resubmit to a partition with idle nodes.
- **Max 72 hours per job**
- **Commit after every meaningful change.** Concise messages. Don't commit checkpoints or large generated files.

## Success Criteria
1. `tests/test_drift_loss_port.py` passes — numerics match official JAX implementation
2. can_image `test/mean_score >= 0.90` within 200 epochs with `bc_coeff=0`
3. No regression on pusht_image (should still get >= 0.78)
