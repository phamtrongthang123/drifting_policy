"""Numerical test: verify PyTorch drift_loss matches official JAX implementation."""
import sys
import os
import numpy as np
import torch
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.model.drifting.drifting_util import drift_loss

# Try to import official JAX version
try:
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    from drifting.drift_loss import drift_loss as drift_loss_jax
    HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    HAS_JAX = False


def _compare(gen_np, pos_np, neg_np=None, R_list=(0.02, 0.05, 0.2),
             rtol=2e-4, atol=1e-4):
    """Run both JAX and PyTorch, assert outputs match."""
    # PyTorch
    gen_pt = torch.from_numpy(gen_np).float()
    pos_pt = torch.from_numpy(pos_np).float()
    neg_pt = torch.from_numpy(neg_np).float() if neg_np is not None else None
    loss_pt, info_pt = drift_loss(gen_pt, pos_pt, fixed_neg=neg_pt, R_list=R_list)

    if not HAS_JAX:
        pytest.skip("JAX not installed — skipping cross-check")

    # JAX
    gen_jx = jnp.array(gen_np, dtype=jnp.float32)
    pos_jx = jnp.array(pos_np, dtype=jnp.float32)
    neg_jx = jnp.array(neg_np, dtype=jnp.float32) if neg_np is not None else None
    loss_jx, info_jx = drift_loss_jax(gen_jx, pos_jx, fixed_neg=neg_jx, R_list=R_list)

    loss_pt_np = loss_pt.detach().numpy()
    loss_jx_np = np.array(loss_jx)

    np.testing.assert_allclose(
        loss_pt_np, loss_jx_np, rtol=rtol, atol=atol,
        err_msg=f"Loss mismatch: PT={loss_pt_np}, JAX={loss_jx_np}")

    for key in info_jx:
        pt_val = info_pt[key].item() if isinstance(info_pt[key], torch.Tensor) else info_pt[key]
        jx_val = float(info_jx[key])
        np.testing.assert_allclose(
            pt_val, jx_val, rtol=rtol, atol=atol,
            err_msg=f"Info[{key}] mismatch: PT={pt_val}, JAX={jx_val}")

    return loss_pt, info_pt


class TestDriftLossPort:

    def test_single_sample_no_neg(self):
        """C_g=1, C_p=1, fixed_neg=None, S=7."""
        rng = np.random.RandomState(42)
        gen = rng.randn(2, 1, 7).astype(np.float32)
        pos = rng.randn(2, 1, 7).astype(np.float32)
        loss, info = _compare(gen, pos, R_list=(0.02, 0.05, 0.2))
        assert loss.shape == (2,)
        assert "scale" in info

    def test_multi_sample_no_neg(self):
        """C_g=4, C_p=4, fixed_neg=None, S=16."""
        rng = np.random.RandomState(123)
        gen = rng.randn(2, 4, 16).astype(np.float32)
        pos = rng.randn(2, 4, 16).astype(np.float32)
        loss, info = _compare(gen, pos, R_list=(0.02, 0.05, 0.2))
        assert loss.shape == (2,)

    def test_with_explicit_neg(self):
        """C_g=4, C_p=4, C_n=2, S=16."""
        rng = np.random.RandomState(456)
        gen = rng.randn(2, 4, 16).astype(np.float32)
        pos = rng.randn(2, 4, 16).astype(np.float32)
        neg = rng.randn(2, 2, 16).astype(np.float32)
        loss, info = _compare(gen, pos, neg, R_list=(0.02, 0.05, 0.2))
        assert loss.shape == (2,)

    def test_different_R_list(self):
        """Different R_list values produce different losses."""
        rng = np.random.RandomState(789)
        gen_np = rng.randn(2, 4, 16).astype(np.float32)
        pos_np = rng.randn(2, 4, 16).astype(np.float32)

        gen1 = torch.from_numpy(gen_np).float()
        pos1 = torch.from_numpy(pos_np).float()
        loss1, _ = drift_loss(gen1, pos1, R_list=(0.1,))

        gen2 = torch.from_numpy(gen_np).float()
        pos2 = torch.from_numpy(pos_np).float()
        loss2, _ = drift_loss(gen2, pos2, R_list=(0.01, 0.1, 1.0))

        assert not torch.allclose(loss1, loss2)

    def test_training_shape(self):
        """Shape matching training: B_train samples as C_g, per-timestep D=7."""
        rng = np.random.RandomState(999)
        B_train = 32
        D = 7
        gen = rng.randn(1, B_train, D).astype(np.float32)
        pos = rng.randn(1, B_train, D).astype(np.float32)
        loss, info = _compare(gen, pos, R_list=(0.02, 0.05, 0.2))
        assert loss.shape == (1,)
        assert loss.item() > 0  # non-trivial force with 32 samples

    def test_gradient_flows(self):
        """Verify gradient flows through gen to model parameters."""
        rng = np.random.RandomState(111)
        gen = torch.from_numpy(rng.randn(1, 8, 7).astype(np.float32))
        gen.requires_grad_(True)
        pos = torch.from_numpy(rng.randn(1, 8, 7).astype(np.float32))
        loss, _ = drift_loss(gen, pos, R_list=(0.05,))
        loss.sum().backward()
        assert gen.grad is not None
        assert gen.grad.abs().sum() > 0

    def test_large_batch_no_nan(self):
        """Smoke test with training-sized batch — no NaN/Inf."""
        rng = np.random.RandomState(222)
        gen = torch.from_numpy(rng.randn(1, 256, 7).astype(np.float32))
        pos = torch.from_numpy(rng.randn(1, 256, 7).astype(np.float32))
        loss, info = drift_loss(gen, pos, R_list=(0.02, 0.05, 0.2))
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        for v in info.values():
            assert not torch.isnan(v).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
