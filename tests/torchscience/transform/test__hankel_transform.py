"""Tests for numerical Hankel transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck

import torchscience.transform as T


class TestHankelTransformForward:
    """Test Hankel transform forward pass correctness."""

    def test_gaussian_order_0(self):
        """H_0{exp(-a*r^2)} = (1/(2a)) * exp(-k^2/(4a))."""
        r = torch.linspace(0.001, 20, 2000, dtype=torch.float64)
        a = 1.0
        f = torch.exp(-a * r**2)
        k = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r, order=0.0)
        expected = (1.0 / (2 * a)) * torch.exp(-(k**2) / (4 * a))

        # Larger tolerance due to numerical integration
        assert torch.allclose(F, expected, rtol=0.1, atol=0.05)

    def test_exponential_decay_order_0(self):
        """H_0{exp(-a*r)} = a / (a^2 + k^2)^(3/2)."""
        r = torch.linspace(0.001, 30, 3000, dtype=torch.float64)
        a = 1.0
        f = torch.exp(-a * r)
        k = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r, order=0.0)
        expected = a / (a**2 + k**2) ** 1.5

        assert torch.allclose(F, expected, rtol=0.1, atol=0.05)

    def test_output_shape(self):
        """Output shape should match k shape."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)

        # Scalar k
        k = torch.tensor(1.0, dtype=torch.float64)
        F = T.hankel_transform(f, k, r)
        assert F.shape == torch.Size([])

        # 1D k
        k = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        F = T.hankel_transform(f, k, r)
        assert F.shape == torch.Size([3])

    def test_batched_input(self):
        """Hankel transform should work with batched inputs."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.float64)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r, dim=-1)
        assert F.shape == torch.Size([5, 2])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        r = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
        f = torch.exp(-r)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F_trap = T.hankel_transform(f, k, r, integration_method="trapezoidal")
        F_simp = T.hankel_transform(f, k, r, integration_method="simpson")

        # Should be close to each other
        assert torch.allclose(F_trap, F_simp, rtol=0.01)

    def test_order_1(self):
        """Test Hankel transform with order 1."""
        r = torch.linspace(0.001, 20, 2000, dtype=torch.float64)
        f = r * torch.exp(-(r**2))  # r * exp(-r^2)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        # Should compute without error
        F = T.hankel_transform(f, k, r, order=1.0)
        assert F.shape == torch.Size([2])
        assert torch.isfinite(F).all()


class TestHankelTransformGradient:
    """Test Hankel transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            return T.hankel_transform(inp, k, r)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(3, 50, dtype=torch.float64, requires_grad=True)
        k = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.hankel_transform(inp, k, r, dim=-1)

        assert gradcheck(func, (f,), raise_exception=True)


class TestHankelTransformMeta:
    """Test Hankel transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        r = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(100, device="meta", dtype=torch.float64)
        k = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.hankel_transform(f, k, r)

        assert F.shape == torch.Size([5])
        assert F.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        r = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(3, 100, device="meta", dtype=torch.float64)
        k = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.hankel_transform(f, k, r, dim=-1)

        assert F.shape == torch.Size([3, 5])


class TestHankelTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.hankel_transform(f, k, r, integration_method="invalid")
