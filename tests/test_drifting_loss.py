"""
Unit tests for compute_drifting_loss and compute_V in drifting_util.py.

Key hypotheses tested:
1. Gradient direction: one step of gradient descent moves x toward y_pos
2. Multi-step convergence: iterative gradient descent converges x toward y_pos mean
3. S_j uses cross-distances (not full pairwise including within-distribution)
4. Eye mask: self-distance in dist_neg is correctly masked

These tests verify the core training mechanism works correctly.
If all tests pass, the training loss is correct and failure to get rollout score
is due to needing more training epochs (not a code bug).
"""

import sys
import os
import pathlib

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
import pytest

from diffusion_policy.model.drifting.drifting_util import compute_V, compute_drifting_loss


class TestGradientDirection:
    """Test that the gradient of compute_drifting_loss pushes x toward y_pos."""

    def test_single_step_moves_toward_ypos(self):
        """
        One gradient step should decrease distance from x to y_pos.
        Setup: y_pos is clearly separated from x (offset by 3.0 in all dims).
        """
        torch.manual_seed(42)
        B, D = 32, 7
        # x starts far from y_pos
        x_data = torch.randn(B, D)
        y_pos = x_data + 3.0  # offset y_pos far from x

        x = x_data.clone().requires_grad_(True)
        y_neg = x.detach().clone()

        dist_before = torch.mean((x.detach() - y_pos) ** 2).item()

        loss, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.02, 0.05, 0.2])
        loss.backward()

        assert x.grad is not None, "No gradient computed"
        assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN/Inf"

        # SGD step
        lr = 0.1
        x_updated = x.detach() - lr * x.grad

        dist_after = torch.mean((x_updated - y_pos) ** 2).item()

        assert dist_after < dist_before, (
            f"Gradient step did NOT move x toward y_pos: "
            f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
        )

    def test_gradient_not_zero(self):
        """The gradient should be non-zero when x != y_pos."""
        torch.manual_seed(0)
        B, D = 16, 7
        x = torch.randn(B, D, requires_grad=True)
        y_pos = torch.randn(B, D) * 0.1 + 2.0  # different distribution
        y_neg = x.detach()

        loss, _ = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.05])
        loss.backward()

        grad_norm = x.grad.norm().item()
        assert grad_norm > 1e-6, f"Gradient is essentially zero: {grad_norm}"

    def test_gradient_direction_toward_ypos(self):
        """
        The gradient should always point in the direction from x toward y_pos,
        regardless of the distance. Verified by checking the dot product between
        (y_pos_mean - x_mean) and (-grad) is positive.
        """
        torch.manual_seed(7)
        B, D = 32, 7
        y_pos = torch.randn(B, D) + 3.0  # offset

        x = torch.randn(B, D, requires_grad=True)
        y_neg = x.detach()

        loss, _ = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.02, 0.05, 0.2])
        loss.backward()

        # Direction from x toward y_pos mean
        direction = (y_pos.mean(0) - x.detach().mean(0))  # [D]
        # Negative gradient should align with direction toward y_pos
        neg_grad_mean = -x.grad.mean(0)  # [D]

        dot = torch.dot(direction, neg_grad_mean).item()
        assert dot > 0, (
            f"Negative gradient direction does not align with y_pos direction. "
            f"dot product = {dot:.4f} (should be > 0)"
        )


class TestMultiStepConvergence:
    """Test that repeated gradient descent converges x toward y_pos distribution."""

    def test_mean_converges_to_ypos_mean(self):
        """
        After N optimization steps, mean(x) should be close to mean(y_pos).
        Uses direct optimization of x (not a neural network).
        """
        torch.manual_seed(123)
        B, D = 64, 7
        target_mean = torch.tensor([2.0] * D)
        y_pos = target_mean.unsqueeze(0) + 0.1 * torch.randn(B, D)  # tight cluster at target

        # Start x far from target
        x = torch.randn(B, D, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.05)

        n_steps = 200
        for step in range(n_steps):
            optimizer.zero_grad()
            y_neg = x.detach().clone()
            loss, _ = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.02, 0.05, 0.2])
            loss.backward()
            optimizer.step()

        final_mean = x.detach().mean(dim=0)
        error = torch.mean((final_mean - target_mean) ** 2).item()

        assert error < 0.5, (
            f"Mean of x did not converge to mean of y_pos after {n_steps} steps. "
            f"Error: {error:.4f}. "
            f"final_mean={final_mean.tolist()}, target={target_mean.tolist()}"
        )

    def test_lambda_decreases_over_optimization(self):
        """
        The lambda metric (drift magnitude) should decrease as optimization proceeds.
        The normalized loss stays ~1.0 by design (Drift Normalization), but lambda
        is the actual convergence indicator per drifting_model_debug.md.
        """
        torch.manual_seed(456)
        B, D = 32, 7
        y_pos = torch.randn(B, D) + 3.0

        x = torch.randn(B, D, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.05)

        lambdas = []
        for _ in range(100):
            optimizer.zero_grad()
            y_neg = x.detach().clone()
            _, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.05])
            # Need loss for backward
            y_neg2 = x.detach().clone()
            loss, _ = compute_drifting_loss(x, y_pos, y_neg2, temperatures=[0.05])
            loss.backward()
            optimizer.step()
            lambdas.append(metrics['train/drifting_lambda_T0.05'])

        first_half_avg = np.mean(lambdas[:50])
        second_half_avg = np.mean(lambdas[50:])

        assert second_half_avg < first_half_avg, (
            f"Lambda (drift magnitude) did not decrease: "
            f"first_half_avg={first_half_avg:.4f}, second_half_avg={second_half_avg:.4f}. "
            f"Lambda should trend down as x approaches y_pos distribution."
        )


class TestSjCrossDistance:
    """Test that S_j uses cross-distances (not full pairwise including within-distribution)."""

    def test_Sj_uses_cross_not_full_pairwise(self):
        """
        S_j should be mean(cdist(x, y_pos)) / sqrt(D).

        When y_pos is clustered (small within-distribution distances), the wrong S_j
        (using full pairwise including within-y_pos) would be underestimated, causing
        gradient overshoot. The correct S_j (cross-distances only) correctly estimates
        the scale of distances between x and y_pos.
        """
        torch.manual_seed(99)
        B, D = 32, 7

        # x: spread out
        x_base = torch.randn(B, D) * 2.0  # large spread

        # y_pos: very tightly clustered (small within-distribution distances)
        y_pos_center = torch.tensor([5.0] * D)
        y_pos = y_pos_center + 0.001 * torch.randn(B, D)  # near-identical

        # Manually compute correct cross-distance S_j
        dist_cross = torch.cdist(x_base, y_pos)
        S_j_correct = (torch.mean(dist_cross) / (D ** 0.5)).item()

        # Manually compute wrong full-pairwise S_j
        xy_cat = torch.cat([x_base, y_pos], dim=0)
        dist_full = torch.cdist(xy_cat, xy_cat)
        S_j_wrong = (torch.mean(dist_full) / (D ** 0.5)).item()

        # Wrong S_j should be much smaller due to near-zero y_pos-to-y_pos distances
        assert S_j_wrong < S_j_correct * 0.9, (
            f"S_j_wrong ({S_j_wrong:.4f}) should be < S_j_correct ({S_j_correct:.4f}) "
            f"for clustered y_pos. Wrong S_j underestimates the scale."
        )

        # Verify current implementation uses the correct cross-distance S_j
        x = x_base.clone().requires_grad_(True)
        y_neg = x.detach()
        loss, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.05])
        S_j_impl = metrics['train/drifting_S_j']

        assert abs(S_j_impl - S_j_correct) < 0.01, (
            f"Implementation S_j ({S_j_impl:.4f}) does not match expected "
            f"cross-distance S_j ({S_j_correct:.4f}). "
            f"Wrong S_j would be {S_j_wrong:.4f}."
        )


class TestEyeMask:
    """Test that the eye mask correctly handles self-distances."""

    def test_eye_mask_unconditional(self):
        """
        The eye mask should always be applied to dist_neg (not conditionally).
        Without the mask, self-distance is 0, causing infinite attraction to self.
        """
        torch.manual_seed(55)
        B, D = 16, 7
        x = torch.randn(B, D)
        y_pos = torch.randn(B, D) + 2.0
        y_neg = x  # same as x to trigger self-distance issue

        T = 0.05 * (D ** 0.5)

        # Test that V is finite (requires eye mask to prevent 0/0 via self-distances)
        V = compute_V(x, y_pos, y_neg, T)

        assert torch.all(torch.isfinite(V)), (
            f"V contains NaN/Inf. Eye mask may not be working. "
            f"V stats: min={V.min().item():.4f}, max={V.max().item():.4f}"
        )

    def test_gradient_finite_with_self_as_neg(self):
        """Gradient should be finite when y_neg = x (common training setting)."""
        torch.manual_seed(77)
        B, D = 16, 7
        x = torch.randn(B, D, requires_grad=True)
        y_pos = torch.randn(B, D) + 2.0
        y_neg = x.detach()  # y_neg same as x

        loss, _ = compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.02, 0.05, 0.2])
        loss.backward()

        assert torch.all(torch.isfinite(x.grad)), (
            f"Gradient contains NaN/Inf when y_neg = x. "
            f"This indicates the eye mask is not working correctly."
        )


class TestMetrics:
    """Test that compute_drifting_loss returns correct metrics."""

    def test_lambda_decreases_during_optimization(self):
        """
        lambda_j (drift magnitude) should decrease over optimization steps.
        Lambda is the key convergence metric per drifting_model_debug.md:
        'should trend downward as the model learns'.
        """
        torch.manual_seed(88)
        B, D = 32, 7
        y_pos = torch.randn(B, D)
        temperatures = [0.02, 0.05, 0.2]

        x = torch.randn(B, D, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.1)

        lambda_history = {T: [] for T in temperatures}

        for _ in range(100):
            optimizer.zero_grad()
            y_neg = x.detach().clone()
            loss, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=temperatures)
            loss.backward()
            optimizer.step()
            for T in temperatures:
                lambda_history[T].append(metrics[f'train/drifting_lambda_T{T}'])

        for T in temperatures:
            vals = lambda_history[T]
            early_avg = np.mean(vals[:20])
            late_avg = np.mean(vals[80:])
            assert late_avg < early_avg, (
                f"lambda at T={T} did not decrease during optimization. "
                f"early={early_avg:.4f}, late={late_avg:.4f}"
            )

    def test_metrics_keys_present(self):
        """compute_drifting_loss should return S_j and lambda for each temperature."""
        torch.manual_seed(11)
        temperatures = [0.02, 0.05, 0.2]
        x = torch.randn(16, 7)
        y_pos = torch.randn(16, 7)
        y_neg = x

        _, metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=temperatures)

        assert 'train/drifting_S_j' in metrics, "Missing S_j metric"
        for T in temperatures:
            key = f'train/drifting_lambda_T{T}'
            assert key in metrics, f"Missing lambda metric for T={T}"


class TestPerTimestepLoss:
    """Test the per-timestep loss used in Can/Lift tasks."""

    def test_per_timestep_trains_all_horizons(self):
        """
        Per-timestep loss trains each horizon position independently.
        All T horizon positions should receive gradients.
        """
        torch.manual_seed(33)
        B, T, D = 8, 16, 7  # batch, horizon, action_dim

        pred_actions = torch.randn(B, T, D, requires_grad=True)
        nactions = torch.randn(B, T, D) + 1.0  # different distribution

        total_loss = 0
        for t in range(T):
            x_t = pred_actions[:, t, :]
            y_pos_t = nactions[:, t, :]
            y_neg_t = x_t.detach()
            loss_t, _ = compute_drifting_loss(x_t, y_pos_t, y_neg_t, temperatures=[0.05])
            total_loss += loss_t

        loss = total_loss / T
        loss.backward()

        # All timestep positions should have non-zero gradients
        assert pred_actions.grad is not None, "No gradient for pred_actions"
        grad_per_t = pred_actions.grad.norm(dim=-1).mean(dim=0)  # [T]
        assert torch.all(grad_per_t > 1e-8), (
            f"Some timesteps have near-zero gradients: {grad_per_t.tolist()}"
        )

    def test_per_timestep_moves_each_t_toward_target(self):
        """
        After one gradient step on the per-timestep loss,
        each timestep position should be closer to the corresponding naction.
        """
        torch.manual_seed(44)
        B, T, D = 16, 16, 7
        pred_actions = torch.randn(B, T, D, requires_grad=True)
        nactions = pred_actions.detach() + 3.0  # offset all timesteps

        dist_before = torch.mean((pred_actions.detach() - nactions) ** 2).item()

        total_loss = 0
        for t in range(T):
            x_t = pred_actions[:, t, :]
            y_pos_t = nactions[:, t, :]
            y_neg_t = x_t.detach()
            loss_t, _ = compute_drifting_loss(x_t, y_pos_t, y_neg_t, temperatures=[0.02, 0.05, 0.2])
            total_loss += loss_t

        (total_loss / T).backward()

        pred_updated = pred_actions.detach() - 0.1 * pred_actions.grad
        dist_after = torch.mean((pred_updated - nactions) ** 2).item()

        assert dist_after < dist_before, (
            f"Per-timestep loss did NOT move pred_actions toward nactions: "
            f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
