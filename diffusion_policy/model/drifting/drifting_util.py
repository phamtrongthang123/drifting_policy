import torch
import torch.nn.functional as F


def _cdist(x, y, eps=1e-8):
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M].

    Exact port of official JAX cdist (dot-product formula + eps clamp).
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None, weight_pos=None,
               weight_neg=None, R_list=(0.02, 0.05, 0.2)):
    """Faithful port of official drifting loss (drifting/drift_loss.py, JAX -> PyTorch).

    Args:
        gen: [B, C_g, S] generated samples
        fixed_pos: [B, C_p, S] positive (real) samples
        fixed_neg: [B, C_n, S] negative samples (optional, None = no explicit negatives)
        weight_gen: [B, C_g] (optional, default 1)
        weight_pos: [B, C_p] (optional, default 1)
        weight_neg: [B, C_n] (optional, default 1)
        R_list: tuple of temperature values
    Returns:
        loss: [B]
        info: dict with 'scale' and 'loss_{R}' entries
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # Goal computation (no gradients)
    with torch.no_grad():
        info = {}
        dist = _cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = dist / torch.clamp(scale, min=1e-3)

        # Mask self-connections for gen block
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        # Force loop over temperatures
        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R

            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))

            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val

            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    # Loss with gradients through gen
    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled.detach()
    loss = (diff ** 2).mean(dim=(-1, -2))

    info = {k: v.mean() for k, v in info.items()}

    return loss, info
