"""Tests for numerical Mellin transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck

import torchscience.transform as T


class TestMellinTransformForward:
    """Test Mellin transform forward pass correctness."""

    def test_exponential_decay(self):
        """M{exp(-t)} = Gamma(s)."""
        # Use t starting slightly above 0 to avoid t^(s-1) singularity at t=0
        t = torch.linspace(0.001, 30, 3000, dtype=torch.float64)
        f = torch.exp(-t)
        s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        F = T.mellin_transform(f, s, t)
        # Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2
        expected = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)

        # Larger tolerance due to numerical integration and truncation
        assert torch.allclose(F, expected, rtol=0.05)

    def test_power_law_decay(self):
        """M{t^a * exp(-t)} = Gamma(s+a)."""
        t = torch.linspace(0.001, 30, 3000, dtype=torch.float64)
        a = 1.0  # So f(t) = t * exp(-t)
        f = (t**a) * torch.exp(-t)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.mellin_transform(f, s, t)
        # Gamma(s+1): Gamma(2)=1, Gamma(3)=2
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)

        assert torch.allclose(F, expected, rtol=0.1)

    def test_output_shape(self):
        """Output shape should match s shape."""
        t = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)

        # Scalar s
        s = torch.tensor(1.0, dtype=torch.float64)
        F = T.mellin_transform(f, s, t)
        assert F.shape == torch.Size([])

        # 1D s
        s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        F = T.mellin_transform(f, s, t)
        assert F.shape == torch.Size([3])

    def test_batched_input(self):
        """Mellin transform should work with batched inputs."""
        t = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.mellin_transform(f, s, t, dim=-1)
        assert F.shape == torch.Size([5, 2])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        t = torch.linspace(0.01, 20, 2000, dtype=torch.float64)
        f = torch.exp(-t)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F_trap = T.mellin_transform(f, s, t, integration_method="trapezoidal")
        F_simp = T.mellin_transform(f, s, t, integration_method="simpson")

        # Should be close to each other
        assert torch.allclose(F_trap, F_simp, rtol=0.01)


class TestMellinTransformGradient:
    """Test Mellin transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def func(inp):
            return T.mellin_transform(inp, s, t)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(3, 50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.mellin_transform(inp, s, t, dim=-1)

        assert gradcheck(func, (f,), raise_exception=True)


class TestMellinTransformMeta:
    """Test Mellin transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.mellin_transform(f, s, t)

        assert F.shape == torch.Size([5])
        assert F.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(3, 100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.mellin_transform(f, s, t, dim=-1)

        assert F.shape == torch.Size([3, 5])


class TestMellinTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        t = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.mellin_transform(f, s, t, integration_method="invalid")


class TestMellinTransformDtype:
    """Test Mellin transform dtype handling."""

    def test_float32_input(self):
        """Mellin transform should work with float32 input."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        f = torch.randn(50, dtype=torch.float32)
        s = torch.tensor([1.0, 2.0], dtype=torch.float32)

        F = T.mellin_transform(f, s, t)
        assert F.dtype == torch.float32

    def test_float64_input(self):
        """Mellin transform should work with float64 input."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.mellin_transform(f, s, t)
        assert F.dtype == torch.float64


class TestMellinTransformDevice:
    """Test Mellin transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(50, dtype=torch.float64, device="cuda")
        s = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")

        try:
            F = T.mellin_transform(f, s, t)
            assert F.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


class TestMellinTransformVmap:
    """Test Mellin transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        t = torch.linspace(0.1, 10, 100, dtype=torch.float64)
        f = torch.randn(8, 100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        # Manual batching
        y_batched = T.mellin_transform(f, s, t, dim=-1)

        # vmap
        def mellin_single(fi):
            return T.mellin_transform(fi, s, t)

        y_vmap = torch.vmap(mellin_single)(f)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(4, 4, 50, dtype=torch.float64)
        s = torch.tensor([1.0], dtype=torch.float64)

        def mellin_single(fi):
            return T.mellin_transform(fi, s, t)

        y_vmap = torch.vmap(torch.vmap(mellin_single))(f)

        assert y_vmap.shape == torch.Size([4, 4, 1])


class TestMellinTransformCompile:
    """Test Mellin transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_mellin(x):
            return T.mellin_transform(x, s, t)

        y_compiled = compiled_mellin(f)
        y_eager = T.mellin_transform(f, s, t)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        t = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_mellin(x):
            return T.mellin_transform(x, s, t)

        y = compiled_mellin(f)
        y.sum().backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
