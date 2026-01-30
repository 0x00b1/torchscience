"""Tests for numerical two-sided Laplace transform implementation."""

import math

import pytest
import torch
from torch.autograd import gradcheck

import torchscience.transform as T


class TestTwoSidedLaplaceTransformForward:
    """Test two-sided Laplace transform forward pass correctness."""

    def test_double_sided_exponential(self):
        """B{exp(-a|t|)} = 2a / (a^2 - s^2) for |s| < a."""
        t = torch.linspace(-20, 20, 4000, dtype=torch.float64)
        a = 2.0
        f = torch.exp(-a * torch.abs(t))
        # Use s values with |s| < a
        s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)
        expected = 2 * a / (a**2 - s**2)

        assert torch.allclose(F, expected, rtol=0.05)

    def test_gaussian(self):
        """B{exp(-t^2)} = sqrt(pi) * exp(s^2/4)."""
        t = torch.linspace(-10, 10, 2000, dtype=torch.float64)
        f = torch.exp(-(t**2))
        s = torch.tensor([0.0], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)
        # At s=0: sqrt(pi)
        expected = torch.tensor([math.sqrt(math.pi)], dtype=torch.float64)

        assert torch.allclose(F, expected, rtol=0.05)

    def test_symmetric_function(self):
        """Transform of even function at s=0 should be twice the one-sided integral."""
        t = torch.linspace(-10, 10, 2000, dtype=torch.float64)
        # Even function: exp(-t^2)
        f = torch.exp(-(t**2))
        s = torch.tensor([0.0], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)

        # At s=0, this equals the integral of exp(-t^2) over all reals = sqrt(pi)
        expected = math.sqrt(math.pi)
        assert abs(F.item() - expected) < 0.1

    def test_output_shape(self):
        """Output shape should match s shape."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)

        # Scalar s
        s = torch.tensor(0.0, dtype=torch.float64)
        F = T.two_sided_laplace_transform(f, s, t)
        assert F.shape == torch.Size([])

        # 1D s
        s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        F = T.two_sided_laplace_transform(f, s, t)
        assert F.shape == torch.Size([3])

    def test_batched_input(self):
        """Two-sided Laplace transform should work with batched inputs."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.float64)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t, dim=-1)
        assert F.shape == torch.Size([5, 2])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        t = torch.linspace(-10, 10, 2000, dtype=torch.float64)
        f = torch.exp(-torch.abs(t))
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        F_trap = T.two_sided_laplace_transform(
            f, s, t, integration_method="trapezoidal"
        )
        F_simp = T.two_sided_laplace_transform(
            f, s, t, integration_method="simpson"
        )

        # Should be close to each other
        assert torch.allclose(F_trap, F_simp, rtol=0.01)


class TestTwoSidedLaplaceTransformGradient:
    """Test two-sided Laplace transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        def func(inp):
            return T.two_sided_laplace_transform(inp, s, t)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(3, 50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([0.0], dtype=torch.float64)

        def func(inp):
            return T.two_sided_laplace_transform(inp, s, t, dim=-1)

        assert gradcheck(func, (f,), raise_exception=True)


class TestTwoSidedLaplaceTransformMeta:
    """Test two-sided Laplace transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)

        assert F.shape == torch.Size([5])
        assert F.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(3, 100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t, dim=-1)

        assert F.shape == torch.Size([3, 5])


class TestTwoSidedLaplaceTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([0.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.two_sided_laplace_transform(
                f, s, t, integration_method="invalid"
            )


class TestTwoSidedLaplaceTransformDtype:
    """Test two-sided Laplace transform dtype handling."""

    def test_float32_input(self):
        """Two-sided Laplace transform should work with float32 input."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float32)
        f = torch.randn(100, dtype=torch.float32)
        s = torch.tensor([0.0, 0.5], dtype=torch.float32)

        F = T.two_sided_laplace_transform(f, s, t)
        assert F.dtype == torch.float32

    def test_float64_input(self):
        """Two-sided Laplace transform should work with float64 input."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)
        assert F.dtype == torch.float64


class TestTwoSidedLaplaceTransformComplex:
    """Test two-sided Laplace transform with complex tensors."""

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex64_input(self):
        """Two-sided Laplace transform should work with complex64 input."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float32)
        f = torch.randn(100, dtype=torch.complex64)
        s = torch.tensor([0.0, 0.5], dtype=torch.float32)

        F = T.two_sided_laplace_transform(f, s, t)
        assert F.dtype == torch.complex64
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex128_input(self):
        """Two-sided Laplace transform should work with complex128 input."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.complex128)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_input_batched(self):
        """Two-sided Laplace transform should work with batched complex input."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.complex128)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        F = T.two_sided_laplace_transform(f, s, t, dim=-1)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([5, 2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_gradcheck(self):
        """Gradient for complex input should pass numerical check."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.complex128, requires_grad=True)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        def func(inp):
            result = T.two_sided_laplace_transform(inp, s, t)
            return result.real, result.imag

        assert gradcheck(func, (f,), raise_exception=True)


class TestTwoSidedLaplaceTransformDevice:
    """Test two-sided Laplace transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64, device="cuda")
        f = torch.randn(100, dtype=torch.float64, device="cuda")
        s = torch.tensor([0.0, 0.5], dtype=torch.float64, device="cuda")

        try:
            F = T.two_sided_laplace_transform(f, s, t)
            assert F.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


class TestTwoSidedLaplaceTransformVmap:
    """Test two-sided Laplace transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        t = torch.linspace(-10, 10, 100, dtype=torch.float64)
        f = torch.randn(8, 100, dtype=torch.float64)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        # Manual batching
        y_batched = T.two_sided_laplace_transform(f, s, t, dim=-1)

        # vmap
        def laplace_single(fi):
            return T.two_sided_laplace_transform(fi, s, t)

        y_vmap = torch.vmap(laplace_single)(f)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(4, 4, 50, dtype=torch.float64)
        s = torch.tensor([0.0], dtype=torch.float64)

        def laplace_single(fi):
            return T.two_sided_laplace_transform(fi, s, t)

        y_vmap = torch.vmap(torch.vmap(laplace_single))(f)

        assert y_vmap.shape == torch.Size([4, 4, 1])


class TestTwoSidedLaplaceTransformCompile:
    """Test two-sided Laplace transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        s = torch.tensor([0.0, 0.5], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_laplace(x):
            return T.two_sided_laplace_transform(x, s, t)

        y_compiled = compiled_laplace(f)
        y_eager = T.two_sided_laplace_transform(f, s, t)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        t = torch.linspace(-5, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([0.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_laplace(x):
            return T.two_sided_laplace_transform(x, s, t)

        y = compiled_laplace(f)
        y.sum().backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
