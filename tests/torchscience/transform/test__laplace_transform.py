"""Tests for numerical Laplace transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestLaplaceTransformForward:
    """Test Laplace transform forward pass correctness."""

    def test_exponential_decay(self):
        """L{exp(-a*t)} = 1/(s+a)."""
        t = torch.linspace(0, 20, 2000, dtype=torch.float64)
        a = 2.0
        f = torch.exp(-a * t)
        s = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t)
        expected = 1.0 / (s + a)

        assert torch.allclose(F, expected, rtol=1e-2)

    def test_constant_function(self):
        """L{1} = 1/s for Re(s) > 0."""
        t = torch.linspace(0, 50, 5000, dtype=torch.float64)
        f = torch.ones_like(t)
        s = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t)
        expected = 1.0 / s

        # Larger tolerance due to truncation at finite t
        assert torch.allclose(F, expected, rtol=0.05)

    def test_linear_function(self):
        """L{t} = 1/s^2."""
        t = torch.linspace(0, 30, 3000, dtype=torch.float64)
        f = t.clone()
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t)
        expected = 1.0 / (s**2)

        # Larger tolerance for polynomial
        assert torch.allclose(F, expected, rtol=0.1)

    def test_output_shape(self):
        """Output shape should match s shape."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)

        # Scalar s
        s = torch.tensor(1.0, dtype=torch.float64)
        F = T.laplace_transform(f, s, t)
        assert F.shape == torch.Size([])

        # 1D s
        s = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        F = T.laplace_transform(f, s, t)
        assert F.shape == torch.Size([3])

    def test_batched_input(self):
        """Laplace transform should work with batched inputs."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t, dim=-1)
        assert F.shape == torch.Size([5, 2])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        t = torch.linspace(0, 10, 1000, dtype=torch.float64)
        f = torch.exp(-2.0 * t)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F_trap = T.laplace_transform(f, s, t, integration_method="trapezoidal")
        F_simp = T.laplace_transform(f, s, t, integration_method="simpson")

        # Should be close to each other
        assert torch.allclose(F_trap, F_simp, rtol=0.01)


class TestLaplaceTransformGradient:
    """Test Laplace transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        t = torch.linspace(0, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def func(inp):
            return T.laplace_transform(inp, s, t)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        t = torch.linspace(0, 5, 30, dtype=torch.float64)
        f = torch.randn(30, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.laplace_transform(inp, s, t)

        assert gradgradcheck(func, (f,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        t = torch.linspace(0, 5, 50, dtype=torch.float64)
        f = torch.randn(3, 50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.laplace_transform(inp, s, t, dim=-1)

        assert gradcheck(func, (f,), raise_exception=True)


class TestLaplaceTransformMeta:
    """Test Laplace transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.laplace_transform(f, s, t)

        assert F.shape == torch.Size([5])
        assert F.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        t = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(3, 100, device="meta", dtype=torch.float64)
        s = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.laplace_transform(f, s, t, dim=-1)

        assert F.shape == torch.Size([3, 5])


class TestLaplaceTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.laplace_transform(f, s, t, integration_method="invalid")

    @pytest.mark.skip(
        reason="Complex s requires complex input - not yet implemented"
    )
    def test_complex_s(self):
        """Should work with complex s values."""
        t = torch.linspace(0, 10, 1000, dtype=torch.float64)
        f = torch.exp(-t)
        s = torch.tensor([1.0 + 0.5j, 2.0 + 1.0j], dtype=torch.complex128)

        # Should not raise
        F = T.laplace_transform(f, s, t)
        assert F.shape == torch.Size([2])


class TestLaplaceTransformDtype:
    """Test Laplace transform with different dtypes."""

    def test_float32_input(self):
        """Should work with float32."""
        t = torch.linspace(0, 10, 100, dtype=torch.float32)
        f = torch.randn(100, dtype=torch.float32)
        s = torch.tensor([1.0, 2.0], dtype=torch.float32)

        F = T.laplace_transform(f, s, t)
        assert F.dtype == torch.float32

    def test_float64_input(self):
        """Should work with float64."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t)
        assert F.dtype == torch.float64


class TestLaplaceTransformVmap:
    """Test Laplace transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(8, 100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        # Manual batching
        y_batched = T.laplace_transform(f, s, t, dim=-1)

        # vmap
        def laplace_single(fi):
            return T.laplace_transform(fi, s, t)

        y_vmap = torch.vmap(laplace_single)(f)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        t = torch.linspace(0, 10, 50, dtype=torch.float64)
        f = torch.randn(4, 4, 50, dtype=torch.float64)
        s = torch.tensor([1.0], dtype=torch.float64)

        def laplace_single(fi):
            return T.laplace_transform(fi, s, t)

        y_vmap = torch.vmap(torch.vmap(laplace_single))(f)

        assert y_vmap.shape == torch.Size([4, 4, 1])


class TestLaplaceTransformCompile:
    """Test Laplace transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_laplace(x):
            return T.laplace_transform(x, s, t)

        y_compiled = compiled_laplace(f)
        y_eager = T.laplace_transform(f, s, t)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        t = torch.linspace(0, 10, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        s = torch.tensor([1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_laplace(x):
            return T.laplace_transform(x, s, t)

        y = compiled_laplace(f)
        y.sum().backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape


class TestLaplaceTransformComplex:
    """Test Laplace transform with complex tensors."""

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex64_input(self):
        """Laplace transform should work with complex64 input."""
        t = torch.linspace(0, 10, 100, dtype=torch.float32)
        f = torch.randn(100, dtype=torch.complex64)
        s = torch.tensor([1.0, 2.0], dtype=torch.float32)

        F = T.laplace_transform(f, s, t)
        assert F.dtype == torch.complex64
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex128_input(self):
        """Laplace transform should work with complex128 input."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.complex128)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_input_batched(self):
        """Laplace transform should work with batched complex input."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.complex128)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        F = T.laplace_transform(f, s, t, dim=-1)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([5, 2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_gradcheck(self):
        """Gradient for complex input should pass numerical check."""
        t = torch.linspace(0, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.complex128, requires_grad=True)
        s = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def func(inp):
            result = T.laplace_transform(inp, s, t)
            return result.real, result.imag

        assert gradcheck(func, (f,), raise_exception=True)


class TestLaplaceTransformDevice:
    """Test Laplace transform on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Should work on CUDA."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64, device="cuda")
        f = torch.randn(100, dtype=torch.float64, device="cuda")
        s = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")

        F = T.laplace_transform(f, s, t)
        assert F.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLaplaceTransformCUDA:
    """Test Laplace transform CUDA backend."""

    def test_cuda_forward_matches_cpu(self):
        """CUDA forward should match CPU output."""
        t_cpu = torch.linspace(0, 10, 100, dtype=torch.float64)
        f_cpu = torch.randn(100, dtype=torch.float64)
        s_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        t_cuda = t_cpu.cuda()
        f_cuda = f_cpu.cuda()
        s_cuda = s_cpu.cuda()

        F_cpu = T.laplace_transform(f_cpu, s_cpu, t_cpu)
        F_cuda = T.laplace_transform(f_cuda, s_cuda, t_cuda)

        assert torch.allclose(F_cpu, F_cuda.cpu(), rtol=1e-10, atol=1e-10)

    def test_cuda_gradient(self):
        """Gradient should work on CUDA."""
        t = torch.linspace(0, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(
            50, dtype=torch.float64, device="cuda", requires_grad=True
        )
        s = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")

        F = T.laplace_transform(f, s, t)
        loss = F.sum()
        loss.backward()

        assert f.grad is not None
        assert f.grad.device.type == "cuda"

    def test_cuda_gradcheck(self):
        """Gradient check on CUDA."""
        t = torch.linspace(0, 5, 30, dtype=torch.float64, device="cuda")
        f = torch.randn(
            30, dtype=torch.float64, device="cuda", requires_grad=True
        )
        s = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")

        def func(inp):
            return T.laplace_transform(inp, s, t)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_cuda_batched(self):
        """Batched Laplace transform on CUDA."""
        t = torch.linspace(0, 10, 100, dtype=torch.float64, device="cuda")
        f = torch.randn(5, 100, dtype=torch.float64, device="cuda")
        s = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")

        F = T.laplace_transform(f, s, t, dim=-1)

        assert F.shape == torch.Size([5, 2])
        assert F.device.type == "cuda"
