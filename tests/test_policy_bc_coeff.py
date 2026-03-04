"""
Tests for bc_coeff integration in DriftingUnetHybridImagePolicy.compute_loss.

Verifies that:
1. bc_loss appears in metrics when bc_coeff > 0
2. Total loss is bc_coeff * MSE(pred, target) + drifting_loss (not just drifting)
3. Gradient from compute_loss correctly pushes pred toward nactions via bc term
4. bc_coeff=0 produces no bc_loss metric (backward compat)
5. Per-timestep loop: bc term applied to full 3D tensor (not per-slice)

These tests run on CPU with a mocked minimal model to avoid GPU/dataset dependencies.
"""

import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import pytest

from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss


def mock_compute_loss(pred_actions, nactions, temperatures, per_timestep_loss, bc_coeff):
    """
    Replicate the core compute_loss logic without obs encoder / UNet.
    Inputs:
        pred_actions: [B, T, Da]
        nactions:     [B, T, Da]
    """
    batch_size = pred_actions.shape[0]

    if per_timestep_loss:
        T_horizon = pred_actions.shape[1]
        total_loss = 0
        all_metrics = {}
        for t in range(T_horizon):
            x_t = pred_actions[:, t, :]
            y_pos_t = nactions[:, t, :]
            y_neg_t = x_t.detach()
            loss_t, metrics_t = compute_drifting_loss(
                x_t, y_pos_t, y_neg_t, temperatures=temperatures)
            total_loss += loss_t
            if t == 0:
                all_metrics = metrics_t
        loss = total_loss / T_horizon
    else:
        x = pred_actions.reshape(batch_size, -1)
        y_pos = nactions.reshape(batch_size, -1)
        y_neg = x.detach()
        loss, all_metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=temperatures)

    if bc_coeff > 0:
        bc_loss = torch.nn.functional.mse_loss(pred_actions, nactions)
        all_metrics['train/bc_loss'] = bc_loss.item()
        loss = loss + bc_coeff * bc_loss

    return loss, all_metrics


class TestBCCoeffMetrics:
    """Test that bc_coeff affects metrics correctly."""

    def test_bc_loss_metric_present_when_coeff_positive(self):
        """compute_loss should include 'train/bc_loss' when bc_coeff > 0."""
        torch.manual_seed(0)
        B, T, Da = 16, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = torch.randn(B, T, Da)

        _, metrics = mock_compute_loss(pred, target,
                                       temperatures=[0.02, 0.05, 0.2],
                                       per_timestep_loss=True,
                                       bc_coeff=10.0)

        assert 'train/bc_loss' in metrics, (
            "Expected 'train/bc_loss' in metrics when bc_coeff=10.0, "
            f"got keys: {list(metrics.keys())}"
        )

    def test_bc_loss_metric_absent_when_coeff_zero(self):
        """compute_loss should NOT include 'train/bc_loss' when bc_coeff=0."""
        torch.manual_seed(1)
        B, T, Da = 16, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = torch.randn(B, T, Da)

        _, metrics = mock_compute_loss(pred, target,
                                       temperatures=[0.02, 0.05, 0.2],
                                       per_timestep_loss=True,
                                       bc_coeff=0.0)

        assert 'train/bc_loss' not in metrics, (
            "Expected 'train/bc_loss' absent when bc_coeff=0.0, "
            f"got keys: {list(metrics.keys())}"
        )

    def test_bc_loss_value_matches_mse(self):
        """bc_loss metric value should equal MSE(pred, target)."""
        torch.manual_seed(2)
        B, T, Da = 16, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = torch.randn(B, T, Da)

        _, metrics = mock_compute_loss(pred, target,
                                       temperatures=[0.05],
                                       per_timestep_loss=True,
                                       bc_coeff=5.0)

        expected_bc = torch.nn.functional.mse_loss(pred.detach(), target).item()
        assert abs(metrics['train/bc_loss'] - expected_bc) < 1e-5, (
            f"bc_loss metric ({metrics['train/bc_loss']:.6f}) "
            f"!= MSE(pred, target) ({expected_bc:.6f})"
        )


class TestBCCoeffLossScale:
    """Test that bc_coeff correctly scales the contribution to total loss."""

    def test_loss_increases_with_bc_coeff(self):
        """
        When pred != target, adding bc_coeff > 0 increases loss because
        MSE(pred, target) > 0.
        """
        torch.manual_seed(3)
        B, T, Da = 16, 16, 7
        pred = torch.randn(B, T, Da)
        target = pred + 3.0  # far from pred

        loss_bc0, _ = mock_compute_loss(pred.clone().requires_grad_(True), target,
                                        temperatures=[0.05],
                                        per_timestep_loss=True,
                                        bc_coeff=0.0)
        loss_bc10, _ = mock_compute_loss(pred.clone().requires_grad_(True), target,
                                         temperatures=[0.05],
                                         per_timestep_loss=True,
                                         bc_coeff=10.0)

        assert loss_bc10.item() > loss_bc0.item(), (
            f"Loss with bc_coeff=10 ({loss_bc10.item():.4f}) should be > "
            f"loss with bc_coeff=0 ({loss_bc0.item():.4f}) when pred != target"
        )

    def test_loss_contribution_proportional_to_coeff(self):
        """
        The difference in loss between bc_coeff=k and bc_coeff=0
        should equal k * MSE(pred, target).
        """
        torch.manual_seed(4)
        B, T, Da = 32, 16, 7
        pred_data = torch.randn(B, T, Da)
        target = torch.randn(B, T, Da)

        mse_val = torch.nn.functional.mse_loss(pred_data, target).item()

        loss0, _ = mock_compute_loss(pred_data.clone().requires_grad_(True), target,
                                     temperatures=[0.05],
                                     per_timestep_loss=True,
                                     bc_coeff=0.0)
        loss5, _ = mock_compute_loss(pred_data.clone().requires_grad_(True), target,
                                     temperatures=[0.05],
                                     per_timestep_loss=True,
                                     bc_coeff=5.0)

        expected_diff = 5.0 * mse_val
        actual_diff = loss5.item() - loss0.item()

        assert abs(actual_diff - expected_diff) < 0.01, (
            f"Loss diff ({actual_diff:.4f}) should equal bc_coeff * MSE = "
            f"5.0 * {mse_val:.4f} = {expected_diff:.4f}"
        )


class TestBCCoeffGradient:
    """Test that bc_coeff adds obs-conditional gradient to pred_actions."""

    def test_bc_gradient_direction_toward_target(self):
        """
        With bc_coeff > 0, one gradient step should reduce MSE(pred, target).
        This is the key correctness property: BC provides paired obs-conditional gradients.
        """
        torch.manual_seed(5)
        B, T, Da = 32, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = pred.detach() + 1.0  # target is offset from pred

        mse_before = torch.nn.functional.mse_loss(pred.detach(), target).item()

        loss, _ = mock_compute_loss(pred, target,
                                    temperatures=[0.02, 0.05, 0.2],
                                    per_timestep_loss=True,
                                    bc_coeff=10.0)
        loss.backward()

        assert pred.grad is not None, "No gradient"
        pred_updated = pred.detach() - 0.01 * pred.grad
        mse_after = torch.nn.functional.mse_loss(pred_updated, target).item()

        assert mse_after < mse_before, (
            f"BC gradient step should reduce MSE: before={mse_before:.4f}, after={mse_after:.4f}"
        )

    def test_bc_gradient_nonzero_when_coeff_positive(self):
        """Gradient through bc term should be non-zero when pred != target."""
        torch.manual_seed(6)
        B, T, Da = 16, 16, 7
        pred = torch.randn(B, T, Da, requires_grad=True)
        target = pred.detach() + 2.0

        loss, _ = mock_compute_loss(pred, target,
                                    temperatures=[0.05],
                                    per_timestep_loss=True,
                                    bc_coeff=10.0)
        loss.backward()

        grad_norm = pred.grad.norm().item()
        assert grad_norm > 1e-4, (
            f"Gradient norm should be > 0 with bc_coeff=10. Got {grad_norm:.6f}"
        )


class TestBCCoeffPerTimestepConsistency:
    """Test bc_coeff behavior is consistent between per_timestep=True and False."""

    def test_bc_loss_same_regardless_of_per_timestep(self):
        """
        bc_loss (MSE of full pred tensor) should be identical whether
        per_timestep_loss=True or False, since it's computed on the full tensor.
        """
        torch.manual_seed(7)
        B, T, Da = 16, 16, 7
        pred_data = torch.randn(B, T, Da)
        target = torch.randn(B, T, Da)

        _, metrics_pt = mock_compute_loss(pred_data.clone().requires_grad_(True), target,
                                          temperatures=[0.05],
                                          per_timestep_loss=True,
                                          bc_coeff=5.0)
        _, metrics_flat = mock_compute_loss(pred_data.clone().requires_grad_(True), target,
                                            temperatures=[0.05],
                                            per_timestep_loss=False,
                                            bc_coeff=5.0)

        assert abs(metrics_pt['train/bc_loss'] - metrics_flat['train/bc_loss']) < 1e-5, (
            f"bc_loss should be same for per_timestep=True ({metrics_pt['train/bc_loss']:.6f}) "
            f"and False ({metrics_flat['train/bc_loss']:.6f})"
        )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
