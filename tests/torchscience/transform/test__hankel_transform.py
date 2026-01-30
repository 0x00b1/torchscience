"""Tests for numerical Hankel transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

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

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(30, dtype=torch.float64, requires_grad=True)
        k = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.hankel_transform(inp, k, r)

        assert gradgradcheck(func, (f,), raise_exception=True)

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


class TestHankelTransformDtype:
    """Test Hankel transform dtype handling."""

    def test_float32_input(self):
        """Hankel transform should work with float32 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        f = torch.randn(50, dtype=torch.float32)
        k = torch.tensor([0.5, 1.0], dtype=torch.float32)

        F = T.hankel_transform(f, k, r)
        assert F.dtype == torch.float32

    def test_float64_input(self):
        """Hankel transform should work with float64 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r)
        assert F.dtype == torch.float64


class TestHankelTransformComplex:
    """Test Hankel transform with complex tensors."""

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex64_input(self):
        """Hankel transform should work with complex64 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        f = torch.randn(50, dtype=torch.complex64)
        k = torch.tensor([0.5, 1.0], dtype=torch.float32)

        F = T.hankel_transform(f, k, r)
        assert F.dtype == torch.complex64
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex128_input(self):
        """Hankel transform should work with complex128 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.complex128)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_input_batched(self):
        """Hankel transform should work with batched complex input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(5, 50, dtype=torch.complex128)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.hankel_transform(f, k, r, dim=-1)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([5, 2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_gradcheck(self):
        """Gradient for complex input should pass numerical check."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.complex128, requires_grad=True)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            result = T.hankel_transform(inp, k, r)
            return result.real, result.imag

        assert gradcheck(func, (f,), raise_exception=True)


class TestHankelTransformDevice:
    """Test Hankel transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(50, dtype=torch.float64, device="cuda")
        k = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        try:
            F = T.hankel_transform(f, k, r)
            assert F.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHankelTransformCUDA:
    """Test Hankel transform CUDA backend."""

    def test_cuda_forward_matches_cpu(self):
        """CUDA forward should match CPU output."""
        r_cpu = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f_cpu = torch.randn(50, dtype=torch.float64)
        k_cpu = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        r_cuda = r_cpu.cuda()
        f_cuda = f_cpu.cuda()
        k_cuda = k_cpu.cuda()

        F_cpu = T.hankel_transform(f_cpu, k_cpu, r_cpu, order=0.0)
        F_cuda = T.hankel_transform(f_cuda, k_cuda, r_cuda, order=0.0)

        assert torch.allclose(F_cpu, F_cuda.cpu(), rtol=1e-10, atol=1e-10)

    def test_cuda_gradient(self):
        """Gradient should work on CUDA."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(
            50, dtype=torch.float64, device="cuda", requires_grad=True
        )
        k = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        F = T.hankel_transform(f, k, r, order=0.0)
        loss = F.sum()
        loss.backward()

        assert f.grad is not None
        assert f.grad.device.type == "cuda"

    def test_cuda_gradcheck(self):
        """Gradient check on CUDA."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64, device="cuda")
        f = torch.randn(
            30, dtype=torch.float64, device="cuda", requires_grad=True
        )
        k = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        def func(inp):
            return T.hankel_transform(inp, k, r, order=0.0)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_cuda_batched(self):
        """Batched Hankel transform on CUDA."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(5, 50, dtype=torch.float64, device="cuda")
        k = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        F = T.hankel_transform(f, k, r, dim=-1, order=0.0)

        assert F.shape == torch.Size([5, 2])
        assert F.device.type == "cuda"


class TestHankelTransformVmap:
    """Test Hankel transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        r = torch.linspace(0.1, 10, 100, dtype=torch.float64)
        f = torch.randn(8, 100, dtype=torch.float64)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        # Manual batching
        y_batched = T.hankel_transform(f, k, r, dim=-1)

        # vmap
        def hankel_single(fi):
            return T.hankel_transform(fi, k, r)

        y_vmap = torch.vmap(hankel_single)(f)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(4, 4, 50, dtype=torch.float64)
        k = torch.tensor([1.0], dtype=torch.float64)

        def hankel_single(fi):
            return T.hankel_transform(fi, k, r)

        y_vmap = torch.vmap(torch.vmap(hankel_single))(f)

        assert y_vmap.shape == torch.Size([4, 4, 1])


class TestHankelTransformCompile:
    """Test Hankel transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        k = torch.tensor([0.5, 1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_hankel(x):
            return T.hankel_transform(x, k, r)

        y_compiled = compiled_hankel(f)
        y_eager = T.hankel_transform(f, k, r)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        k = torch.tensor([1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_hankel(x):
            return T.hankel_transform(x, k, r)

        y = compiled_hankel(f)
        y.sum().backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
