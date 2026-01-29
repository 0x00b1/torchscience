"""Tests for inverse Hankel transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck

import torchscience.transform as T


class TestInverseHankelTransformForward:
    """Test inverse Hankel transform forward pass correctness."""

    def test_round_trip_gaussian(self):
        """Round-trip with Gaussian should approximately recover original."""
        # Original radial function: Gaussian
        r_fwd = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
        f = torch.exp(-(r_fwd**2))

        # Forward Hankel transform
        k = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = T.hankel_transform(f, k, r_fwd, order=0.0)

        # Inverse Hankel transform
        r_inv = torch.linspace(0.1, 5, 40, dtype=torch.float64)
        f_reconstructed = T.inverse_hankel_transform(F, r_inv, k, order=0.0)

        # Expected values
        f_expected = torch.exp(-(r_inv**2))

        # Numerical integration has errors; use relaxed tolerance
        assert torch.allclose(f_reconstructed, f_expected, rtol=0.3, atol=0.15)

    def test_self_reciprocal_order_0(self):
        """Order 0 Hankel transform should be self-reciprocal."""
        # For order 0, forward and inverse have same form
        k = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = torch.exp(-(k**2))  # Gaussian in k-space

        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)

        # Forward and inverse should give similar structure
        f_fwd = T.hankel_transform(F, r, k, order=0.0)
        f_inv = T.inverse_hankel_transform(F, r, k, order=0.0)

        # Self-reciprocal means they should be equal
        assert torch.allclose(f_fwd, f_inv, rtol=1e-10)

    def test_output_shape(self):
        """Output shape should match r shape."""
        k = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(100, dtype=torch.float64)

        # Scalar r
        r = torch.tensor(1.0, dtype=torch.float64)
        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f.shape == torch.Size([])

        # 1D r
        r = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f.shape == torch.Size([3])

    def test_batched_input(self):
        """Inverse Hankel transform should work with batched inputs."""
        k = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(5, 100, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, dim=-1, order=0.0)
        assert f.shape == torch.Size([5, 2])

    def test_1d_input(self):
        """Inverse Hankel transform should work with 1D input."""
        k = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(100, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f.shape == torch.Size([3])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        k = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = torch.exp(-(k**2))
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f_trap = T.inverse_hankel_transform(
            F, r, k, order=0.0, integration_method="trapezoidal"
        )
        f_simp = T.inverse_hankel_transform(
            F, r, k, order=0.0, integration_method="simpson"
        )

        # Should be close to each other
        assert torch.allclose(f_trap, f_simp, rtol=0.1)

    def test_different_orders(self):
        """Inverse Hankel transform should work with different orders."""
        k = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = torch.exp(-(k**2))
        r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        # Order 0
        f0 = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f0.shape == torch.Size([3])
        assert torch.isfinite(f0).all()

        # Order 1
        f1 = T.inverse_hankel_transform(F, r, k, order=1.0)
        assert f1.shape == torch.Size([3])
        assert torch.isfinite(f1).all()

        # Order 2
        f2 = T.inverse_hankel_transform(F, r, k, order=2.0)
        assert f2.shape == torch.Size([3])
        assert torch.isfinite(f2).all()


class TestInverseHankelTransformGradient:
    """Test inverse Hankel transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        k = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(30, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            return T.inverse_hankel_transform(inp, r, k, order=0.0)

        assert gradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        k = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(3, 30, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.inverse_hankel_transform(inp, r, k, dim=-1, order=0.0)

        assert gradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        k = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(50, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        loss = f.sum()
        loss.backward()

        assert F.grad is not None
        assert F.grad.shape == F.shape
        assert torch.isfinite(F.grad).all()
        # Verify gradient is non-zero
        assert F.grad.abs().max() > 1e-10


class TestInverseHankelTransformMeta:
    """Test inverse Hankel transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        k = torch.empty(100, device="meta", dtype=torch.float64)
        F = torch.empty(100, device="meta", dtype=torch.float64)
        r = torch.empty(5, device="meta", dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, order=0.0)

        assert f.shape == torch.Size([5])
        assert f.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        k = torch.empty(100, device="meta", dtype=torch.float64)
        F = torch.empty(3, 100, device="meta", dtype=torch.float64)
        r = torch.empty(5, device="meta", dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, dim=-1, order=0.0)

        assert f.shape == torch.Size([3, 5])


class TestInverseHankelTransformDtype:
    """Test inverse Hankel transform dtype handling."""

    def test_float32_input(self):
        """Inverse Hankel transform should work with float32 input."""
        k = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        F = torch.randn(50, dtype=torch.float32)
        r = torch.tensor([0.5, 1.0], dtype=torch.float32)

        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f.dtype == torch.float32

    def test_float64_input(self):
        """Inverse Hankel transform should work with float64 input."""
        k = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(50, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f = T.inverse_hankel_transform(F, r, k, order=0.0)
        assert f.dtype == torch.float64


class TestInverseHankelTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        k = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(100, dtype=torch.float64)
        r = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.inverse_hankel_transform(
                F, r, k, order=0.0, integration_method="invalid"
            )


class TestInverseHankelTransformDevice:
    """Test inverse Hankel transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        k = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        F = torch.randn(50, dtype=torch.float64, device="cuda")
        r = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        try:
            f = T.inverse_hankel_transform(F, r, k, order=0.0)
            assert f.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise
