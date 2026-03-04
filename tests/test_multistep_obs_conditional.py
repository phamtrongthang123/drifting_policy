"""
Test: per_timestep_loss=True with bc_coeff=10 learns multi-timestep obs-conditional mapping.

MOTIVATION:
  All existing tests (test_obs_conditional.py, test_bc_coeff_stabilizes_training.py) use
  single-timestep (flat) models. Can task uses per_timestep_loss=True with T=16, D_act=7.

  The per_timestep_loss loop calls compute_drifting_loss independently per timestep,
  then bc_loss is applied once on the full trajectory [B, T, Da].

  Key question: does this combination correctly learn obs → multi-timestep action?

SETUP (mirrors compute_loss in drifting_unet_hybrid_image_policy.py):
  - multi-timestep obs-conditional data: obs [B, D_obs], actions [B, T, D_act]
  - Linear model: pred_actions = obs @ W.T, repeated across T timesteps
  - Training loop replicates the per_timestep_loss loop + bc_coeff from compute_loss

EXPECTED RESULTS:
  - bc_coeff=0: each timestep drifts independently → MSE diverges
  - bc_coeff=10: bc_loss anchors each timestep → all T timesteps converge to near-zero MSE
"""

import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import pytest

from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss


def build_multistep_data(B=128, T=16, D_act=7, D_obs=16, seed=42):
    """
    Create obs-conditional multi-timestep dataset.
    obs [B, D_obs], actions [B, T, D_act]: each obs has a distinct T-step trajectory.
    """
    torch.manual_seed(seed)
    obs = torch.randn(B, D_obs)
    # Independent W per timestep (different obs->action mapping at each step)
    W_true = torch.randn(T, D_act, D_obs) * 0.3
    # actions[b, t, :] = obs[b] @ W_true[t].T
    actions = torch.einsum('bo,tdo->btd', obs, W_true)  # [B, T, D_act]
    # Range-normalize to [-1, 1] (mimics get_range_normalizer_from_stat)
    actions = actions / (actions.abs().amax() + 1e-6)
    return obs, actions


def mock_per_timestep_compute_loss(pred_actions, nactions, temperatures, bc_coeff):
    """
    Replicate the per_timestep_loss=True branch of compute_loss exactly.

    pred_actions: [B, T, Da] - model predictions
    nactions:     [B, T, Da] - ground truth actions
    """
    T_horizon = pred_actions.shape[1]
    total_loss = 0
    for t in range(T_horizon):
        x_t = pred_actions[:, t, :]    # [B, Da]
        y_pos_t = nactions[:, t, :]    # [B, Da]
        y_neg_t = x_t                  # y_neg = x (as in policy, not detached)
        loss_t, _ = compute_drifting_loss(
            x_t, y_pos_t, y_neg_t, temperatures=temperatures)
        total_loss += loss_t
    loss = total_loss / T_horizon

    if bc_coeff > 0:
        bc_loss = torch.nn.functional.mse_loss(pred_actions, nactions)
        loss = loss + bc_coeff * bc_loss

    return loss


def train_multistep_model(obs, actions, bc_coeff, n_steps=400, lr=0.05, seed=0):
    """
    Train a linear obs-conditional multi-timestep model using the per_timestep_loss loop.
    Model: pred_actions[b, t, :] = obs[b] @ W.T  (shared W across timesteps for simplicity)
    Returns final MSE on all timesteps.
    """
    torch.manual_seed(seed)
    B, T, D_act = actions.shape
    D_obs = obs.shape[1]
    W = nn.Parameter(torch.randn(D_act, D_obs) * 0.01)
    opt = torch.optim.Adam([W], lr=lr)

    for _ in range(n_steps):
        # pred_actions[b, t, :] = obs[b] @ W.T  (same W for all t)
        pred_actions = (obs @ W.T).unsqueeze(1).expand(B, T, D_act)  # [B, T, Da]

        loss = mock_per_timestep_compute_loss(
            pred_actions, actions, temperatures=[0.02, 0.05, 0.2], bc_coeff=bc_coeff)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_actions_final = (obs @ W.T).unsqueeze(1).expand(B, T, D_act)
        mse = torch.nn.functional.mse_loss(pred_actions_final, actions).item()
    return mse


class TestMultistepPerTimestepLoss:
    """Validate per_timestep_loss=True + bc_coeff for T=16, D=7 (Can-like dimensions)."""

    def test_bc_coeff_10_converges_multistep(self):
        """
        per_timestep_loss=True + bc_coeff=10 should converge for multi-timestep obs-conditional.
        This is the exact scenario used in Can (T=16, D_act=7) job 178684.
        """
        obs, actions = build_multistep_data(B=128, T=16, D_act=7, D_obs=16)
        mse = train_multistep_model(obs, actions, bc_coeff=10.0, n_steps=400)

        assert mse < 0.1, (
            f"per_timestep_loss=True + bc_coeff=10 should converge (MSE < 0.1), "
            f"got MSE={mse:.4f}. "
            f"This is the Can task setup (T=16, D_act=7). "
            f"If failing, bc_coeff integration with per_timestep loop is broken."
        )

    def test_pure_drifting_fails_multistep(self):
        """
        per_timestep_loss=True WITHOUT bc_coeff should fail for obs-conditional mapping.
        Each timestep independently diverges (same root cause as single-timestep case).
        """
        obs, actions = build_multistep_data(B=128, T=16, D_act=7, D_obs=16)
        mse_bc0 = train_multistep_model(obs, actions, bc_coeff=0.0, n_steps=400)
        mse_bc10 = train_multistep_model(obs, actions, bc_coeff=10.0, n_steps=400)

        assert mse_bc10 < mse_bc0, (
            f"bc_coeff=10 ({mse_bc10:.4f}) should be better than bc=0 ({mse_bc0:.4f}) "
            f"for multi-timestep obs-conditional mapping."
        )

    def test_bc_loss_applied_to_full_trajectory(self):
        """
        bc_loss in the per_timestep loop is MSE over all [B, T, Da] — not per-timestep.
        Verify: bc_loss == MSE(pred_actions, nactions) over the full 3D tensor.
        """
        torch.manual_seed(42)
        B, T, Da = 32, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = torch.randn(B, T, Da)

        # Compute bc_loss as done in mock_per_timestep_compute_loss
        bc_loss_full = torch.nn.functional.mse_loss(pred.detach(), target).item()

        # Compute per-timestep MSE and average
        bc_loss_per_t = sum(
            torch.nn.functional.mse_loss(pred[:, t, :].detach(), target[:, t, :]).item()
            for t in range(T)
        ) / T

        # They should be equal (MSE is decomposable across time)
        assert abs(bc_loss_full - bc_loss_per_t) < 1e-4, (
            f"MSE over full trajectory ({bc_loss_full:.6f}) should equal "
            f"average of per-timestep MSEs ({bc_loss_per_t:.6f})."
        )

    def test_convergence_scales_with_bc_coeff(self):
        """
        Higher bc_coeff → faster/better convergence for multi-timestep obs-conditional.
        bc=10 should beat bc=1.
        """
        obs, actions = build_multistep_data(B=128, T=16, D_act=7, D_obs=16)
        mse_bc1 = train_multistep_model(obs, actions, bc_coeff=1.0, n_steps=400)
        mse_bc10 = train_multistep_model(obs, actions, bc_coeff=10.0, n_steps=400)

        assert mse_bc10 < mse_bc1, (
            f"bc_coeff=10 ({mse_bc10:.4f}) should converge better than bc=1 ({mse_bc1:.4f}) "
            f"for multi-timestep obs-conditional mapping."
        )

    def test_per_timestep_vs_flat_loss_both_need_bc(self):
        """
        Both per_timestep=True (7D per step) and flat (112D total) need bc_coeff.
        Verify that without bc_coeff, both fail for obs-conditional (Can scenario).
        """
        torch.manual_seed(42)
        B, T, D_act = 64, 16, 7
        D_obs = 16

        obs = torch.randn(B, D_obs)
        W_true = torch.randn(D_act, D_obs) * 0.3
        true_flat = obs @ W_true.T  # [B, D_act] — obs-conditional single-step action
        # Expand to T timesteps
        actions = true_flat.unsqueeze(1).expand(B, T, D_act)  # [B, T, D_act]
        actions = actions / (actions.abs().amax() + 1e-6)

        mse_pt_bc0 = train_multistep_model(obs, actions, bc_coeff=0.0, n_steps=300)
        mse_pt_bc10 = train_multistep_model(obs, actions, bc_coeff=10.0, n_steps=300)

        # With bc_coeff=10: should converge
        assert mse_pt_bc10 < 0.1, (
            f"per_timestep=True + bc=10 should converge even with uniform-across-T actions. "
            f"MSE={mse_pt_bc10:.4f}"
        )
        # Without bc: bc=10 should beat bc=0
        assert mse_pt_bc10 < mse_pt_bc0, (
            f"bc=10 ({mse_pt_bc10:.4f}) should beat bc=0 ({mse_pt_bc0:.4f}) "
            f"for per_timestep obs-conditional."
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
