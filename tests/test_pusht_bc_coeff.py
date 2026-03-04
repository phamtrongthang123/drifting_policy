"""
Test: bc_coeff=10 + per_timestep_loss=True for PushT-like dimensions (D_act=2, T=16).

MOTIVATION:
  PushT action space is 2D (XY velocity), T=16 horizon → flat loss is 32D.
  Previously PushT used per_timestep_loss=False (32D flat) and got 0.78 score.
  New config adds bc_coeff=10.0 AND per_timestep_loss=True (2D per timestep).

  Key questions:
  1. Does bc_coeff=10 enable obs-conditional convergence for D_act=2?
  2. Does per_timestep=True converge similarly to per_timestep=False when bc dominates?
  3. Does bc_coeff=0 (pure drifting) fail for D_act=2 T=16 (same root cause as Can)?

  This validates the PushT yaml changes before running expensive GPU jobs.

SETUP:
  - obs-conditional data: obs [B, D_obs], actions [B, T=16, D_act=2]
  - Linear model: pred = obs @ W.T  → single W shared across T
  - Both per_timestep=True (2D × 16 steps) and per_timestep=False (32D flat) tested

EXPECTED RESULTS:
  - bc_coeff=10 + either mode: converges (MSE < 0.1)
  - bc_coeff=0 + per_timestep=True: fails (MSE >> bc_coeff=10 case)
  - per_timestep=True ≈ per_timestep=False when bc_coeff=10 (bc dominates)
"""

import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import pytest

from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss


def build_pusht_data(B=512, T=16, D_act=2, D_obs=32, seed=42):
    """PushT-like obs-conditional dataset: obs->actions, each timestep independent."""
    torch.manual_seed(seed)
    obs = torch.randn(B, D_obs)
    W_true = torch.randn(T, D_act, D_obs) * 0.3
    actions = torch.einsum('bo,tdo->btd', obs, W_true)  # [B, T, D_act]
    # Normalize to [-1, 1] like range normalizer
    actions = actions / (actions.abs().amax() + 1e-6)
    return obs, actions


def per_timestep_loss(pred_actions, nactions, temperatures, bc_coeff):
    """per_timestep_loss=True branch of compute_loss (mirrors policy implementation)."""
    T_horizon = pred_actions.shape[1]
    total_loss = 0
    for t in range(T_horizon):
        x_t = pred_actions[:, t, :]
        y_pos_t = nactions[:, t, :]
        loss_t, _ = compute_drifting_loss(x_t, y_pos_t, x_t, temperatures=temperatures)
        total_loss += loss_t
    loss = total_loss / T_horizon
    if bc_coeff > 0:
        bc_loss = torch.nn.functional.mse_loss(pred_actions, nactions)
        loss = loss + bc_coeff * bc_loss
    return loss


def flat_loss(pred_actions, nactions, temperatures, bc_coeff):
    """per_timestep_loss=False branch: flatten to [B, T*D_act] then single drifting loss."""
    B = pred_actions.shape[0]
    x = pred_actions.reshape(B, -1)
    y_pos = nactions.reshape(B, -1)
    loss, _ = compute_drifting_loss(x, y_pos, x, temperatures=temperatures)
    if bc_coeff > 0:
        bc_loss = torch.nn.functional.mse_loss(pred_actions, nactions)
        loss = loss + bc_coeff * bc_loss
    return loss


def train(obs, actions, loss_fn, n_steps=300, lr=0.05, seed=0):
    """Train linear model with given loss function. Returns final MSE."""
    torch.manual_seed(seed)
    B, T, D_act = actions.shape
    D_obs = obs.shape[1]
    W = nn.Parameter(torch.randn(D_act, D_obs) * 0.01)
    opt = torch.optim.Adam([W], lr=lr)

    for _ in range(n_steps):
        pred = (obs @ W.T).unsqueeze(1).expand(B, T, D_act)  # [B, T, D_act]
        loss_val = loss_fn(pred, actions)
        opt.zero_grad()
        loss_val.backward()
        opt.step()

    with torch.no_grad():
        pred_final = (obs @ W.T).unsqueeze(1).expand(B, T, D_act)
        return torch.nn.functional.mse_loss(pred_final, actions).item()


TEMPS = [0.02, 0.05, 0.2]


class TestPushTBcCoeff:
    """Validate bc_coeff=10 + per_timestep_loss=True for PushT (D_act=2, T=16)."""

    def test_per_timestep_bc10_converges(self):
        """
        per_timestep=True + bc_coeff=10 should converge for PushT dims (D=2, T=16).
        This is the exact new PushT config (drifting_pusht_image.yaml).
        """
        obs, actions = build_pusht_data(B=256, T=16, D_act=2)
        mse = train(obs, actions,
                    loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=10.0))
        assert mse < 0.1, (
            f"per_timestep=True + bc_coeff=10 should converge for D_act=2 (PushT), "
            f"got MSE={mse:.4f}. bc_coeff integration broken."
        )

    def test_flat_bc10_also_converges(self):
        """
        per_timestep=False + bc_coeff=10 also converges for D_act=2.
        bc_coeff drives obs-conditional regardless of drifting mode (32D vs 2D).
        """
        obs, actions = build_pusht_data(B=256, T=16, D_act=2)
        mse = train(obs, actions,
                    loss_fn=lambda p, a: flat_loss(p, a, TEMPS, bc_coeff=10.0))
        assert mse < 0.1, (
            f"per_timestep=False + bc_coeff=10 should also converge for D_act=2, "
            f"got MSE={mse:.4f}. bc_coeff should dominate regardless of drifting mode."
        )

    def test_pure_drifting_fails_pusht(self):
        """
        bc_coeff=0 (pure drifting) should fail for PushT obs-conditional.
        Same root cause as Can: eye-mask asymmetry at fixed point.
        """
        obs, actions = build_pusht_data(B=256, T=16, D_act=2)
        mse_bc0 = train(obs, actions,
                        loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=0.0),
                        n_steps=500)
        mse_bc10 = train(obs, actions,
                         loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=10.0),
                         n_steps=300)
        assert mse_bc10 < mse_bc0, (
            f"bc=10 ({mse_bc10:.4f}) should significantly outperform bc=0 ({mse_bc0:.4f}) "
            f"for PushT obs-conditional. Pure drifting fails without bc_coeff."
        )

    def test_per_timestep_vs_flat_both_valid_with_bc(self):
        """
        Both per_timestep=True and per_timestep=False with bc_coeff=10 converge.
        bc_coeff dominates the gradient for obs-conditional learning in both modes.
        The per_timestep change in PushT yaml is a regularization improvement, not critical.
        """
        obs, actions = build_pusht_data(B=256, T=16, D_act=2)
        mse_pt = train(obs, actions,
                       loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=10.0))
        mse_flat = train(obs, actions,
                         loss_fn=lambda p, a: flat_loss(p, a, TEMPS, bc_coeff=10.0))

        # Both should converge to low MSE
        assert mse_pt < 0.1, f"per_timestep=True + bc=10 failed: MSE={mse_pt:.4f}"
        assert mse_flat < 0.1, f"per_timestep=False + bc=10 failed: MSE={mse_flat:.4f}"

    def test_bc_coeff_dominates_for_small_action_dim(self):
        """
        For D_act=2 (small), bc_coeff=10 strongly dominates bc_loss over drifting loss.
        The bc gradient signal is clean: MSE(pred, target) is simple in 2D.
        bc=10 >> bc=1 for obs-conditional convergence.
        """
        obs, actions = build_pusht_data(B=256, T=16, D_act=2)
        mse_bc1 = train(obs, actions,
                        loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=1.0))
        mse_bc10 = train(obs, actions,
                         loss_fn=lambda p, a: per_timestep_loss(p, a, TEMPS, bc_coeff=10.0))
        assert mse_bc10 < mse_bc1, (
            f"bc=10 ({mse_bc10:.4f}) should outperform bc=1 ({mse_bc1:.4f}) "
            f"for PushT-like data (D_act=2)."
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
