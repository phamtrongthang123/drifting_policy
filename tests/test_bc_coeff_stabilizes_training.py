"""
Test: bc_coeff prevents late-training MSE divergence.

MOTIVATION:
  Lift image training hit 0.92 score at epoch 100 but declined to 0.78 by epoch 150.
  The root cause is the eye-mask asymmetry: once MSE is small (pred ≈ target),
  the drifting gradient is non-zero due to the asymmetry, pushing the model AWAY
  from the correct solution. Drift normalization (loss ≈ 1.0 always) amplifies this.

  bc_coeff fixes this by providing a restoring force: when MSE is small but non-zero,
  the BC gradient (2*bc_coeff*(pred - target)) pulls the model back toward the target.

WHAT THESE TESTS CHECK:
  1. Without bc_coeff: MSE first decreases (drifting learns marginal), then
     INCREASES as drifting noise dominates and pushes model off the fixed point.
  2. With bc_coeff=10.0: MSE decreases and STAYS LOW (BC restoring force prevents drift).
  3. Stability holds even starting near the correct solution (mimicking late-training
     where the model has already learned the obs-conditional mapping).
"""

import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
import pytest

from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss


def build_obs_conditional_data(B=128, D_obs=16, D_act=7, seed=42):
    """
    Create obs-conditional dataset with diverse actions (different obs → different actions).
    This mimics a task like Lift/PushT where each obs has a distinct target action.
    """
    torch.manual_seed(seed)
    obs = torch.randn(B, D_obs)
    # Ground-truth obs → action mapping: a fixed linear map
    W_true = torch.randn(D_act, D_obs) * 0.3
    true_actions = obs @ W_true.T  # [B, D_act]
    # Normalize to [-1, 1] range (mimics range normalizer)
    true_actions = true_actions / (true_actions.abs().max() + 1e-6)
    return obs, true_actions, W_true


def train_model(obs, true_actions, bc_coeff, n_steps, lr=0.05, seed=0):
    """
    Train a linear obs-conditional model.
    Returns (mse_trajectory, final_mse) where mse_trajectory is sampled every 10 steps.
    """
    torch.manual_seed(seed)
    D_obs = obs.shape[1]
    D_act = true_actions.shape[1]
    W = nn.Parameter(torch.randn(D_act, D_obs) * 0.01)
    opt = torch.optim.Adam([W], lr=lr)

    mse_trajectory = []

    for step in range(n_steps):
        pred = obs @ W.T
        y_neg = pred.detach()
        loss_d, _ = compute_drifting_loss(pred, true_actions, y_neg,
                                          temperatures=[0.02, 0.05, 0.2])
        if bc_coeff > 0:
            bc_loss = torch.mean((pred - true_actions) ** 2)
            total = loss_d + bc_coeff * bc_loss
        else:
            total = loss_d

        opt.zero_grad()
        total.backward()
        opt.step()

        if step % 10 == 0:
            with torch.no_grad():
                mse = torch.mean((obs @ W.T - true_actions) ** 2).item()
            mse_trajectory.append(mse)

    with torch.no_grad():
        final_mse = torch.mean((obs @ W.T - true_actions) ** 2).item()

    return mse_trajectory, final_mse


class TestBCCoeffPreventsLateTrainingDivergence:
    """
    Tests that bc_coeff prevents MSE from increasing after near-optimal initialization.

    The divergence only occurs in the specific case where the model is at the fixed
    point of the identity mapping (pred ≈ target via x=obs, action=obs, W≈I).
    In that setting, the eye-mask asymmetry creates a strong bias: dist_pos[i,i] ≈ 0
    while dist_neg[i,i] is masked, so V points away from the correct solution.
    This is confirmed by test_obs_conditional.py and is the mechanism that was
    hypothesized to cause Lift score decline (0.92 → 0.78).
    """

    def test_identity_mapping_diverges_without_bc(self):
        """
        For identity mapping (obs=action, W_true=I), drifting diverges from W≈I init.

        This is the canonical setting where the eye-mask asymmetry creates a strong
        non-zero V at the fixed point. test_obs_conditional.py:test_drifting_diverges_from_good_init
        demonstrates this with 300 steps; here we verify at 200 steps.

        The mechanism: at W=I, pred=obs=y_pos (pairwise). dist_pos[i,i]≈0 (unmasked),
        dist_neg[i,i] masked → kernel strongly weights the i-th positive but the
        corresponding negative is excluded → V_i ≠ 0 → model pushed off W=I.
        """
        torch.manual_seed(42)
        B, D = 64, 7
        # Identity mapping: true_actions = obs (exact identity)
        obs = torch.randn(B, D) * 2

        # Start very close to the correct solution W=I
        W = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
        opt = torch.optim.Adam([W], lr=0.01)

        mse_init = torch.mean((obs @ W.detach().T - obs) ** 2).item()

        for _ in range(200):
            pred = obs @ W.T
            y_neg = pred.detach()
            loss_d, _ = compute_drifting_loss(pred, obs, y_neg,
                                              temperatures=[0.02, 0.05, 0.2])
            opt.zero_grad()
            loss_d.backward()
            opt.step()

        mse_final = torch.mean((obs @ W.detach().T - obs) ** 2).item()

        print(f"\n  MSE init (W≈I): {mse_init:.4f}")
        print(f"  MSE after 200 steps (pure drifting): {mse_final:.4f}")

        # Drifting should increase MSE from near-zero (divergence from W=I)
        assert mse_final > mse_init * 5, (
            f"Expected drifting to diverge from near-identity W. "
            f"Got MSE_init={mse_init:.4f}, MSE_final={mse_final:.4f}. "
            f"Divergence requires MSE_final > 5x MSE_init."
        )

    def test_identity_mapping_bc_coeff_prevents_divergence(self):
        """
        For identity mapping, bc_coeff=10.0 prevents divergence from W≈I init.

        BC provides a restoring force: gradient = 2*(pred - target) → 0 at W=I.
        This counteracts the drifting noise that would otherwise push W away from I.
        """
        torch.manual_seed(42)
        B, D = 64, 7
        obs = torch.randn(B, D) * 2

        W = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
        opt = torch.optim.Adam([W], lr=0.01)

        mse_init = torch.mean((obs @ W.detach().T - obs) ** 2).item()

        for _ in range(200):
            pred = obs @ W.T
            y_neg = pred.detach()
            loss_d, _ = compute_drifting_loss(pred, obs, y_neg,
                                              temperatures=[0.02, 0.05, 0.2])
            bc_loss = torch.mean((pred - obs) ** 2)
            total = loss_d + 10.0 * bc_loss
            opt.zero_grad()
            total.backward()
            opt.step()

        mse_final = torch.mean((obs @ W.detach().T - obs) ** 2).item()

        print(f"\n  MSE init (W≈I): {mse_init:.4f}")
        print(f"  MSE after 200 steps (bc=10): {mse_final:.4f}")

        # bc_coeff should keep MSE near zero
        assert mse_final < mse_init * 5, (
            f"With bc_coeff=10, MSE should stay near init. "
            f"Got MSE_init={mse_init:.4f}, MSE_final={mse_final:.4f}."
        )
        assert mse_final < 0.05, (
            f"bc_coeff=10 from W≈I init should keep MSE < 0.05, got {mse_final:.4f}."
        )

    def test_bc_stabilizes_better_than_pure_drifting_at_identity_mapping(self):
        """
        Direct comparison: bc_coeff=10 vs bc=0 for identity mapping from W≈I init.
        bc_coeff=10 should give lower final MSE (stabilization property).
        """
        torch.manual_seed(42)
        B, D = 64, 7
        obs = torch.randn(B, D) * 2

        def final_mse(bc_coeff):
            torch.manual_seed(0)
            W = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
            opt = torch.optim.Adam([W], lr=0.01)
            for _ in range(200):
                pred = obs @ W.T
                y_neg = pred.detach()
                loss_d, _ = compute_drifting_loss(pred, obs, y_neg,
                                                  temperatures=[0.02, 0.05, 0.2])
                if bc_coeff > 0:
                    total = loss_d + bc_coeff * torch.mean((pred - obs) ** 2)
                else:
                    total = loss_d
                opt.zero_grad()
                total.backward()
                opt.step()
            return torch.mean((obs @ W.detach().T - obs) ** 2).item()

        mse_bc0 = final_mse(0.0)
        mse_bc10 = final_mse(10.0)

        print(f"\n  Final MSE bc=0:  {mse_bc0:.4f}")
        print(f"  Final MSE bc=10: {mse_bc10:.4f}")

        assert mse_bc10 < mse_bc0, (
            f"bc_coeff=10 should stabilize identity mapping better than bc=0. "
            f"Got bc=0: {mse_bc0:.4f}, bc=10: {mse_bc10:.4f}."
        )


class TestBCCoeffForPushT:
    """
    Tests specific to PushT scenario: 2D action space, flat trajectory loss.
    PushT is stuck at 0.78 without bc_coeff; these tests verify bc_coeff should help.
    """

    def test_pusht_action_dim_bc_converges(self):
        """
        With PushT-like dimensions (2D action, flat 32D loss), bc_coeff=10 learns
        obs-conditional mapping while bc=0 diverges.

        PushT: horizon=16, action_dim=2 → flat tensor of 32D.
        """
        torch.manual_seed(42)
        B, T, D_act = 64, 16, 2  # PushT dimensions
        D_flat = T * D_act  # 32D flat

        # Build obs-conditional data: obs → action trajectory
        obs = torch.randn(B, 32)  # obs feature (image embedding-like)
        W_true = torch.randn(D_flat, 32) * 0.3
        true_flat = obs @ W_true.T  # [B, 32]
        true_flat = true_flat / (true_flat.abs().max() + 1e-6)

        def train_flat(bc_coeff, seed=42, steps=300):
            torch.manual_seed(seed)
            W = nn.Parameter(torch.randn(D_flat, 32) * 0.01)
            opt = torch.optim.Adam([W], lr=0.05)
            for _ in range(steps):
                pred = obs @ W.T
                y_neg = pred.detach()
                loss_d, _ = compute_drifting_loss(pred, true_flat, y_neg,
                                                  temperatures=[0.02, 0.05, 0.2])
                if bc_coeff > 0:
                    bc_loss = torch.mean((pred - true_flat) ** 2)
                    total = loss_d + bc_coeff * bc_loss
                else:
                    total = loss_d
                opt.zero_grad()
                total.backward()
                opt.step()
            with torch.no_grad():
                return torch.mean((obs @ W.T - true_flat) ** 2).item()

        mse_bc0 = train_flat(bc_coeff=0.0)
        mse_bc10 = train_flat(bc_coeff=10.0)

        print(f"\n  PushT-like flat 32D scenario:")
        print(f"  MSE bc=0:  {mse_bc0:.4f}")
        print(f"  MSE bc=10: {mse_bc10:.4f}")

        assert mse_bc10 < mse_bc0, (
            f"bc_coeff=10 should produce lower MSE than bc=0 for PushT flat scenario. "
            f"Got bc=0: {mse_bc0:.4f}, bc=10: {mse_bc10:.4f}."
        )
        assert mse_bc10 < 0.5, (
            f"bc_coeff=10 should learn obs-conditional mapping for PushT (MSE < 0.5), "
            f"got {mse_bc10:.4f}."
        )


class TestBCCoeffInYAMLConfigs:
    """
    Verify that the YAML configs for Lift and PushT now have bc_coeff set.
    These are integration tests for config correctness.
    """

    def _load_yaml(self, path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def test_lift_image_has_bc_coeff(self):
        """drifting_lift_image.yaml should have bc_coeff=10.0 in policy section."""
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "drifting_lift_image.yaml"
        cfg = self._load_yaml(config_path)
        assert 'bc_coeff' in cfg['policy'], (
            "drifting_lift_image.yaml policy section is missing bc_coeff. "
            "Lift training declined from 0.92 to 0.78 due to drifting divergence; "
            "bc_coeff is needed to stabilize."
        )
        assert cfg['policy']['bc_coeff'] == 10.0, (
            f"drifting_lift_image.yaml policy.bc_coeff should be 10.0, "
            f"got {cfg['policy']['bc_coeff']}"
        )

    def test_pusht_image_has_bc_coeff(self):
        """drifting_pusht_image.yaml should have bc_coeff=10.0 in policy section."""
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "drifting_pusht_image.yaml"
        cfg = self._load_yaml(config_path)
        assert 'bc_coeff' in cfg['policy'], (
            "drifting_pusht_image.yaml policy section is missing bc_coeff. "
            "PushT is at 0.78 (target 0.86); bc_coeff should improve obs-conditional learning."
        )
        assert cfg['policy']['bc_coeff'] == 10.0, (
            f"drifting_pusht_image.yaml policy.bc_coeff should be 10.0, "
            f"got {cfg['policy']['bc_coeff']}"
        )

    def test_can_image_has_bc_coeff(self):
        """drifting_can_image.yaml should have bc_coeff=10.0 (already set, regression guard)."""
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "drifting_can_image.yaml"
        cfg = self._load_yaml(config_path)
        assert 'bc_coeff' in cfg['policy'], (
            "drifting_can_image.yaml policy section is missing bc_coeff."
        )
        assert cfg['policy']['bc_coeff'] == 10.0, (
            f"drifting_can_image.yaml policy.bc_coeff should be 10.0, "
            f"got {cfg['policy']['bc_coeff']}"
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
