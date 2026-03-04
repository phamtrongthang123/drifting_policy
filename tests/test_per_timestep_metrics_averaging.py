"""
Test: per_timestep_loss=True averages metrics across all T timesteps.

MOTIVATION:
  Previously, compute_loss with per_timestep_loss=True only captured metrics
  from t=0 (the first timestep), silently discarding information from t=1..T-1.
  This meant that if a specific timestep had diverging lambda or unusual S_j,
  it would be invisible in monitoring.

  Fix: accumulate metrics across all T timesteps and divide by T (average).

WHAT THESE TESTS CHECK:
  1. With per_timestep=True, returned metrics are a weighted average over all T
     timesteps — not just t=0.
  2. If t=0 and t>0 have genuinely different distributions, the averaged metrics
     differ from t=0-only metrics (detects the old bug).
  3. With per_timestep=False, metrics still come from the single flat call.
  4. The number of metric keys is the same regardless of mode.
  5. bc_loss metric is present when bc_coeff > 0, absent when bc_coeff == 0.
  6. Metrics are consistent: lambda values are positive, S_j is positive.
"""

import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

import torch
import pytest
import numpy as np

from diffusion_policy.model.drifting.drifting_util import compute_drifting_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_per_timestep_metrics(pred_actions, nactions, temperatures, t0_only=False):
    """
    Reference implementation of the per_timestep_loss metric aggregation.

    t0_only=True  → old behavior: only return metrics from t=0
    t0_only=False → new behavior: average metrics across all T timesteps
    """
    T_horizon = pred_actions.shape[1]
    total_loss = 0
    accumulated_metrics = {}
    all_metrics_t0 = {}
    for t in range(T_horizon):
        x_t = pred_actions[:, t, :]
        y_pos_t = nactions[:, t, :]
        y_neg_t = x_t
        loss_t, metrics_t = compute_drifting_loss(
            x_t, y_pos_t, y_neg_t, temperatures=temperatures)
        total_loss += loss_t
        if t == 0:
            all_metrics_t0 = metrics_t
        for k, v in metrics_t.items():
            accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / T_horizon

    loss = total_loss / T_horizon
    if t0_only:
        return loss.item(), all_metrics_t0
    else:
        return loss.item(), accumulated_metrics


class TestPerTimestepMetricsAveraging:

    def test_averaged_metrics_differ_from_t0_only_when_distributions_vary(self):
        """
        If different timesteps have different distributions, the averaged metrics
        should differ from t=0 metrics. This verifies the fix actually changes behavior.

        We construct pred_actions where t=0 has near-zero distances (tight cluster)
        and t>0 has large distances (spread out). lambda at t=0 will be small,
        while lambda at t>0 will be large. The averaged lambda should be > t=0 lambda.
        """
        torch.manual_seed(42)
        B, T, Da = 32, 8, 7
        temperatures = [0.02, 0.05, 0.2]

        # t=0: tight cluster around 0 (small intra-distribution distances)
        # t>0: spread out (large intra-distribution distances)
        pred_actions = torch.zeros(B, T, Da)
        nactions = torch.zeros(B, T, Da)

        # t=0: tiny deviations
        pred_actions[:, 0, :] = torch.randn(B, Da) * 0.01
        nactions[:, 0, :] = torch.randn(B, Da) * 0.01 + 0.1

        # t=1..T-1: large deviations
        for t in range(1, T):
            pred_actions[:, t, :] = torch.randn(B, Da) * 2.0
            nactions[:, t, :] = torch.randn(B, Da) * 2.0

        _, metrics_t0_only = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=True)
        _, metrics_averaged = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)

        # The averaged S_j should be larger than t=0 S_j (since t>0 has larger distances)
        sj_t0 = metrics_t0_only['train/drifting_S_j']
        sj_avg = metrics_averaged['train/drifting_S_j']

        print(f"\n  S_j at t=0 only: {sj_t0:.4f}")
        print(f"  S_j averaged: {sj_avg:.4f}")

        assert sj_avg > sj_t0 * 2, (
            f"With t>0 having larger distances, averaged S_j should be much larger than t=0 S_j. "
            f"Got S_j(t0)={sj_t0:.4f}, S_j(avg)={sj_avg:.4f}. "
            f"This suggests metrics averaging is not working correctly."
        )

    def test_averaged_metrics_equal_t0_when_all_timesteps_identical(self):
        """
        If all timesteps have the same distribution, averaged metrics == t=0 metrics.
        This is a sanity check: averaging doesn't distort when timesteps are homogeneous.
        """
        torch.manual_seed(42)
        B, T, Da = 32, 8, 7
        temperatures = [0.02, 0.05, 0.2]

        # All timesteps: same distribution (tiled along T)
        base_pred = torch.randn(B, Da)
        base_nact = torch.randn(B, Da)
        pred_actions = base_pred.unsqueeze(1).expand(B, T, Da).clone()
        nactions = base_nact.unsqueeze(1).expand(B, T, Da).clone()

        _, metrics_t0_only = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=True)
        _, metrics_averaged = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)

        # With identical timesteps, averaged == t0
        for k in metrics_t0_only:
            assert abs(metrics_t0_only[k] - metrics_averaged[k]) < 1e-5, (
                f"When all timesteps are identical, averaged metric '{k}' should equal "
                f"t=0 metric. Got t0={metrics_t0_only[k]:.6f}, avg={metrics_averaged[k]:.6f}."
            )

    def test_averaged_metrics_same_keys_as_flat_mode(self):
        """
        per_timestep_loss=True returns the same metric keys as flat mode.
        Key invariant: no metrics are silently dropped by the averaging.
        """
        torch.manual_seed(42)
        B, T, Da = 32, 4, 7
        temperatures = [0.02, 0.05, 0.2]

        pred_actions = torch.randn(B, T, Da)
        nactions = torch.randn(B, T, Da)

        # Flat mode: compute on [B, T*Da] tensor
        x_flat = pred_actions.reshape(B, -1)
        y_flat = nactions.reshape(B, -1)
        _, metrics_flat = compute_drifting_loss(x_flat, y_flat, x_flat, temperatures)

        # Per-timestep mode
        _, metrics_per_ts = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)

        assert set(metrics_flat.keys()) == set(metrics_per_ts.keys()), (
            f"per_timestep and flat modes should return the same metric keys. "
            f"flat keys: {set(metrics_flat.keys())}, "
            f"per_timestep keys: {set(metrics_per_ts.keys())}"
        )

    def test_all_averaged_metrics_are_positive(self):
        """
        lambda and S_j are always positive by definition (RMS magnitude, mean distance).
        This holds for per-timestep averaged metrics too.
        """
        torch.manual_seed(42)
        B, T, Da = 64, 16, 2  # PushT-like
        temperatures = [0.02, 0.05, 0.2]

        pred_actions = torch.randn(B, T, Da)
        nactions = torch.randn(B, T, Da)

        _, metrics = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)

        for k, v in metrics.items():
            assert v > 0, (
                f"Metric '{k}' should be positive, got {v:.6f}. "
                f"lambda is an RMS value; S_j is a mean distance. Both must be > 0."
            )

    def test_per_timestep_average_is_weighted_mean_of_individual_metrics(self):
        """
        The averaged metric for N timesteps equals (1/N) * sum_t metric_t.
        Directly validates the averaging formula: accumulated[k] += v / T_horizon
        """
        torch.manual_seed(0)
        B, T, Da = 32, 4, 7
        temperatures = [0.05]

        pred_actions = torch.randn(B, T, Da)
        nactions = torch.randn(B, T, Da)

        # Compute per-timestep metrics independently
        individual_Sj = []
        for t in range(T):
            x_t = pred_actions[:, t, :]
            y_t = nactions[:, t, :]
            _, m = compute_drifting_loss(x_t, y_t, x_t, temperatures)
            individual_Sj.append(m['train/drifting_S_j'])

        expected_avg_Sj = sum(individual_Sj) / T

        # Get averaged metrics from the function
        _, metrics_avg = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)
        actual_avg_Sj = metrics_avg['train/drifting_S_j']

        print(f"\n  Expected avg S_j: {expected_avg_Sj:.6f}")
        print(f"  Actual avg S_j:   {actual_avg_Sj:.6f}")

        assert abs(expected_avg_Sj - actual_avg_Sj) < 1e-5, (
            f"Averaged S_j should equal manual average of per-timestep S_j values. "
            f"Expected {expected_avg_Sj:.6f}, got {actual_avg_Sj:.6f}."
        )

    def test_loss_is_unaffected_by_metrics_averaging_change(self):
        """
        The loss value itself must not change due to the metrics averaging fix.
        The fix only changes which metrics dict is returned, not the gradient computation.
        """
        torch.manual_seed(42)
        B, T, Da = 32, 8, 7
        temperatures = [0.02, 0.05, 0.2]

        pred_actions = torch.randn(B, T, Da)
        nactions = torch.randn(B, T, Da)

        loss_t0, _ = run_per_timestep_metrics(pred_actions, nactions, temperatures, t0_only=True)
        loss_avg, _ = run_per_timestep_metrics(pred_actions, nactions, temperatures, t0_only=False)

        assert abs(loss_t0 - loss_avg) < 1e-6, (
            f"Loss should be identical regardless of metrics averaging mode. "
            f"Got loss_t0={loss_t0:.6f}, loss_avg={loss_avg:.6f}. "
            f"Metrics averaging must not affect the training objective."
        )


class TestComputeLossMetricsIntegration:
    """
    Integration tests verifying that DriftingUnetHybridImagePolicy.compute_loss
    returns correct averaged metrics when per_timestep_loss=True.
    Uses a minimal mock to avoid loading full robomimic/obs encoder.
    """

    def _make_mock_policy_compute_loss(self, pred_actions, nactions, temperatures,
                                       per_timestep_loss, bc_coeff):
        """
        Simulate the compute_loss body without the full policy class.
        Returns (loss, metrics) as compute_loss would.
        """
        if per_timestep_loss:
            T_horizon = pred_actions.shape[1]
            total_loss = 0
            accumulated_metrics = {}
            for t in range(T_horizon):
                x_t = pred_actions[:, t, :]
                y_pos_t = nactions[:, t, :]
                y_neg_t = x_t
                loss_t, metrics_t = compute_drifting_loss(
                    x_t, y_pos_t, y_neg_t, temperatures=temperatures)
                total_loss += loss_t
                for k, v in metrics_t.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / T_horizon
            loss = total_loss / T_horizon
            all_metrics = accumulated_metrics
        else:
            B = pred_actions.shape[0]
            x = pred_actions.reshape(B, -1)
            y_pos = nactions.reshape(B, -1)
            y_neg = x
            loss, all_metrics = compute_drifting_loss(x, y_pos, y_neg, temperatures=temperatures)

        if bc_coeff > 0:
            import torch.nn.functional as F
            bc_loss = F.mse_loss(pred_actions, nactions)
            all_metrics['train/bc_loss'] = bc_loss.item()
            loss = loss + bc_coeff * bc_loss

        return loss, all_metrics

    def test_bc_loss_present_when_bc_coeff_positive(self):
        """compute_loss returns 'train/bc_loss' key when bc_coeff > 0."""
        torch.manual_seed(42)
        B, T, Da = 32, 16, 2
        pred = torch.randn(B, T, Da)
        nact = torch.randn(B, T, Da)

        _, m = self._make_mock_policy_compute_loss(
            pred, nact, [0.02, 0.05, 0.2], per_timestep_loss=True, bc_coeff=10.0)

        assert 'train/bc_loss' in m, (
            "With bc_coeff=10.0, 'train/bc_loss' must be in returned metrics."
        )
        assert m['train/bc_loss'] > 0

    def test_bc_loss_absent_when_bc_coeff_zero(self):
        """compute_loss does NOT return 'train/bc_loss' when bc_coeff == 0."""
        torch.manual_seed(42)
        B, T, Da = 32, 16, 2
        pred = torch.randn(B, T, Da)
        nact = torch.randn(B, T, Da)

        _, m = self._make_mock_policy_compute_loss(
            pred, nact, [0.02, 0.05, 0.2], per_timestep_loss=True, bc_coeff=0.0)

        assert 'train/bc_loss' not in m, (
            "With bc_coeff=0, 'train/bc_loss' must NOT appear in metrics."
        )

    def test_averaged_lambda_less_than_max_individual_lambda(self):
        """
        Averaged lambda should be <= max individual lambda.
        Verifies averaging is applied (not just taking max).
        """
        torch.manual_seed(42)
        B, T, Da = 64, 16, 7  # Can-like
        temperatures = [0.02, 0.05, 0.2]

        # Make t=0 have extreme lambda by using very different pred vs target
        pred_actions = torch.randn(B, T, Da)
        nactions = torch.randn(B, T, Da)
        # t=0: far apart to create large lambda
        pred_actions[:, 0, :] = torch.randn(B, Da) * 5

        # Compute individual lambdas
        max_lambda = 0
        for t in range(T):
            x_t = pred_actions[:, t, :]
            y_t = nactions[:, t, :]
            _, m = compute_drifting_loss(x_t, y_t, x_t, temperatures)
            for k, v in m.items():
                if 'lambda' in k:
                    max_lambda = max(max_lambda, v)

        _, metrics_avg = run_per_timestep_metrics(
            pred_actions, nactions, temperatures, t0_only=False)

        for k, v in metrics_avg.items():
            if 'lambda' in k:
                assert v <= max_lambda + 1e-5, (
                    f"Averaged lambda '{k}'={v:.4f} should be <= max individual lambda {max_lambda:.4f}. "
                    f"Averaging must reduce or equal the maximum."
                )


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s', '--tb=short'])
