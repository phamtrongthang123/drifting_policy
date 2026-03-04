"""
Tests for obs-conditional convergence of the drifting loss.

ROOT CAUSE FINDING (2026-03-04):
The drifting loss alone CANNOT learn obs-conditional generation for small batches.
It actively DIVERGES from the correct solution.

Why:
  The sample-level V has a systematic non-zero bias even at p=q, caused by the eye mask
  asymmetry: dist(x, y_neg=x) masks self-distances but dist(x, y_pos) does not.
  At the correct solution (pred ≈ target), V ≠ 0 due to this asymmetry.
  The drift normalization (forcing loss ≈ 1.0) amplifies this noise, pushing the model
  AWAY from the fixed point on every gradient step.

Evidence:
  - Linear model test (D=7, B=64): drifting alone → MSE increases from 3.6 → 8.9
  - Good initialization (W≈I): MSE increases from 0.007 → 0.7 (diverges!)
  - 860 epochs of real Can training: lambda not decreasing, score=0.0

Why PushT works at 0.78:
  Unknown — possibly due to 2D action space + flat 32D loss providing
  more averaging, or the task being simpler. More investigation needed.

FIX:
  Add BC (behavior cloning) auxiliary loss: L = L_drifting + bc_coeff * MSE(pred, target)
  This provides the correct paired obs-conditional gradient.

  Test results for bc_coeff → MSE after 500 steps:
    0.0 → 10.1 (pure drifting: diverges)
    0.1 → 0.56
    1.0 → 0.12
    10.0 → 0.002 (≈ DDPM level)
    100.0 → 0.000 (pure BC)

  bc_coeff=10.0 recommended: BC dominates for obs-conditioning while drifting
  provides marginal distribution matching as a regularizer.
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


def build_train_data(B=64, D=7, seed=42):
    """Create obs-conditional data: obs ~ N(0,4), action = obs (identity mapping)."""
    torch.manual_seed(seed)
    obs = torch.randn(B, D) * 2
    true_actions = obs.clone()
    return obs, true_actions


def train_with_combined_loss(obs, true_actions, bc_coeff=0.0, n_steps=500, lr=0.05, seed=42):
    """
    Train a simple linear obs-conditional model with drifting + BC loss.

    Model: x = W * obs  (linear mapping, no noise)
    Loss: L_drifting + bc_coeff * MSE(x, true_actions)
    """
    torch.manual_seed(seed)
    D = obs.shape[1]
    W = nn.Parameter(torch.randn(D, D) * 0.01)
    opt = torch.optim.Adam([W], lr=lr)

    for _ in range(n_steps):
        x = obs @ W.T
        y_neg = x.detach()
        loss_d, _ = compute_drifting_loss(x, true_actions, y_neg,
                                           temperatures=[0.02, 0.05, 0.2])
        bc_loss = torch.mean((x - true_actions) ** 2)
        total = loss_d + bc_coeff * bc_loss
        opt.zero_grad()
        total.backward()
        opt.step()

    with torch.no_grad():
        pred = obs @ W.T
    return torch.mean((pred - true_actions) ** 2).item()


class TestDriftingFailsOBsConditional:
    """
    Tests confirming that drifting loss alone cannot learn obs-conditional mapping.
    """

    def test_drifting_alone_diverges(self):
        """
        Drifting loss without BC should diverge (MSE increases, not decreases).

        This is the root cause of Can training failure:
        860 epochs, score=0.0, lambda not decreasing.
        """
        obs, true_actions = build_train_data()

        # Random init MSE
        D = obs.shape[1]
        torch.manual_seed(42)
        W_init = nn.Parameter(torch.randn(D, D) * 0.01)
        mse_init = torch.mean((obs @ W_init.T.detach() - true_actions) ** 2).item()

        # After 500 steps of pure drifting
        mse_after = train_with_combined_loss(obs, true_actions, bc_coeff=0.0, n_steps=500)

        print(f"\n  MSE before drifting: {mse_init:.4f}")
        print(f"  MSE after 500 drifting steps: {mse_after:.4f}")
        print(f"  Drifting {'DIVERGED' if mse_after > mse_init else 'converged'}")

        # Drifting alone should NOT reduce MSE (it actually increases it)
        assert mse_after > 2.0, (
            f"Expected drifting alone to fail (MSE > 2.0), got MSE={mse_after:.4f}. "
            f"If this passes, something changed in the loss that fixed obs-conditioning."
        )

    def test_drifting_diverges_from_good_init(self):
        """
        Even starting near the correct solution (W≈I), drifting diverges.

        This confirms the eye mask asymmetry causes systematic overshoot:
        V_i ≠ 0 at the fixed point (p=q sample-level) → model oscillates away.
        """
        obs, true_actions = build_train_data()
        D = obs.shape[1]

        torch.manual_seed(42)
        W_good = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
        opt = torch.optim.Adam([W_good], lr=0.01)

        mse_init = torch.mean((obs @ W_good.detach().T - true_actions) ** 2).item()

        for _ in range(300):
            x = obs @ W_good.T
            y_neg = x.detach()
            loss_d, _ = compute_drifting_loss(x, true_actions, y_neg,
                                               temperatures=[0.02, 0.05, 0.2])
            opt.zero_grad()
            loss_d.backward()
            opt.step()

        mse_final = torch.mean((obs @ W_good.detach().T - true_actions) ** 2).item()

        print(f"\n  MSE at good init (W≈I): {mse_init:.4f}")
        print(f"  MSE after 300 drifting steps: {mse_final:.4f}")

        # Drifting should move AWAY from the correct solution
        assert mse_final > mse_init * 10, (
            f"Expected drifting to diverge from good init "
            f"(MSE_final > 10*MSE_init={10*mse_init:.4f}), "
            f"got MSE_final={mse_final:.4f}."
        )


class TestBCCoefficientFix:
    """
    Tests confirming that bc_coeff (BC auxiliary loss) fixes obs-conditional learning.
    """

    def test_bc_coeff_10_learns_obs_conditional(self):
        """
        With bc_coeff=10.0, the combined drifting+BC loss should converge
        to a near-perfect obs-conditional mapping.

        bc_coeff=10.0 is recommended for Can training:
        - BC dominates → correct obs-conditional gradients
        - Drifting acts as regularizer (marginal distribution matching)
        """
        obs, true_actions = build_train_data()
        mse = train_with_combined_loss(obs, true_actions, bc_coeff=10.0, n_steps=500)

        print(f"\n  MSE with bc_coeff=10.0: {mse:.4f}")

        assert mse < 0.05, (
            f"bc_coeff=10.0 should produce near-zero MSE (< 0.05), got {mse:.4f}. "
            f"The BC auxiliary loss fix is not working as expected."
        )

    def test_bc_coeff_required_for_convergence(self):
        """
        Higher bc_coeff → better obs-conditional convergence.
        bc_coeff=0 (pure drifting) → high MSE.
        bc_coeff=10 → near-zero MSE.
        """
        obs, true_actions = build_train_data()

        mse_0 = train_with_combined_loss(obs, true_actions, bc_coeff=0.0, n_steps=300)
        mse_1 = train_with_combined_loss(obs, true_actions, bc_coeff=1.0, n_steps=300)
        mse_10 = train_with_combined_loss(obs, true_actions, bc_coeff=10.0, n_steps=300)

        print(f"\n  MSE bc_coeff=0.0:  {mse_0:.4f}")
        print(f"  MSE bc_coeff=1.0:  {mse_1:.4f}")
        print(f"  MSE bc_coeff=10.0: {mse_10:.4f}")

        # Monotonically: higher bc_coeff → lower MSE
        assert mse_10 < mse_1 < mse_0, (
            f"Expected MSE(bc=10) < MSE(bc=1) < MSE(bc=0), got "
            f"{mse_10:.4f}, {mse_1:.4f}, {mse_0:.4f}"
        )
        # bc=10 should get close to the ground truth
        assert mse_10 < 0.1, (
            f"bc_coeff=10 should get MSE < 0.1, got {mse_10:.4f}"
        )

    def test_bc_coeff_10_generalizes_to_new_obs(self):
        """
        A model trained with bc_coeff=10.0 should generalize to held-out obs
        (not just the training obs).
        """
        obs_train, actions_train = build_train_data(B=64, seed=42)
        obs_test, actions_test = build_train_data(B=64, seed=99)  # different data

        D = obs_train.shape[1]
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(D, D) * 0.01)
        opt = torch.optim.Adam([W], lr=0.05)

        for _ in range(500):
            x = obs_train @ W.T
            y_neg = x.detach()
            loss_d, _ = compute_drifting_loss(x, actions_train, y_neg,
                                               temperatures=[0.02, 0.05, 0.2])
            bc_loss = torch.mean((x - actions_train) ** 2)
            total = loss_d + 10.0 * bc_loss
            opt.zero_grad()
            total.backward()
            opt.step()

        with torch.no_grad():
            pred_test = obs_test @ W.T
        mse_test = torch.mean((pred_test - actions_test) ** 2).item()

        print(f"\n  MSE on held-out obs: {mse_test:.4f}")
        assert mse_test < 0.1, (
            f"Model with bc_coeff=10 should generalize. MSE={mse_test:.4f}"
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
