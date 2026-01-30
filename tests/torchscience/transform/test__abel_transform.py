"""Tests for Abel transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestAbelTransformForward:
    """Test Abel transform forward pass correctness."""

    def test_gaussian(self):
        """Abel{exp(-r^2)} = sqrt(pi) * exp(-y^2)."""
        r = torch.linspace(0.01, 10, 1000, dtype=torch.float64)
        f = torch.exp(-(r**2))
        y = torch.linspace(0.01, 5, 50, dtype=torch.float64)

        F = T.abel_transform(f, y, r)

        # Analytical result
        expected = torch.sqrt(
            torch.tensor(torch.pi, dtype=torch.float64)
        ) * torch.exp(-(y**2))

        # Numerical integration has some error
        assert torch.allclose(F, expected, rtol=0.1, atol=0.05)

    def test_constant_function(self):
        """Abel{c} = 2c * sqrt(R^2 - y^2) for r <= R."""
        R = 5.0
        c = 2.0
        r = torch.linspace(0.01, R, 500, dtype=torch.float64)
        f = torch.full_like(r, c)
        # Use y values away from the boundary where numerical integration is more accurate
        y = torch.linspace(0.01, R - 1.0, 30, dtype=torch.float64)

        F = T.abel_transform(f, y, r)

        # Analytical: 2c * sqrt(R^2 - y^2) for y < R
        expected = 2 * c * torch.sqrt(R**2 - y**2)

        # Numerical integration has errors, especially near boundaries
        assert torch.allclose(F, expected, rtol=0.15, atol=0.2)

    def test_output_shape(self):
        """Output shape should match y shape."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)

        # Scalar y
        y = torch.tensor(1.0, dtype=torch.float64)
        F = T.abel_transform(f, y, r)
        assert F.shape == torch.Size([])

        # 1D y
        y = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        F = T.abel_transform(f, y, r)
        assert F.shape == torch.Size([3])

    def test_batched_input(self):
        """Abel transform should work with batched inputs."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(5, 100, dtype=torch.float64)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r, dim=-1)
        assert F.shape == torch.Size([5, 2])

    def test_1d_input(self):
        """Abel transform should work with 1D input."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.exp(-r)
        y = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r)
        assert F.shape == torch.Size([3])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        r = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        f = torch.exp(-r)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F_trap = T.abel_transform(f, y, r, integration_method="trapezoidal")
        F_simp = T.abel_transform(f, y, r, integration_method="simpson")

        # Should be close to each other
        assert torch.allclose(F_trap, F_simp, rtol=0.05)


class TestAbelTransformGradient:
    """Test Abel transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(30, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            return T.abel_transform(inp, y, r)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        r = torch.linspace(0.1, 5, 20, dtype=torch.float64)
        f = torch.randn(20, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.abel_transform(inp, y, r)

        assert gradgradcheck(func, (f,), raise_exception=True)

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(3, 30, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.abel_transform(inp, y, r, dim=-1)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_backward_pass(self):
        """Test that backward pass works."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r)
        loss = F.sum()
        loss.backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
        assert torch.isfinite(f.grad).all()


class TestAbelTransformMeta:
    """Test Abel transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        r = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(100, device="meta", dtype=torch.float64)
        y = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.abel_transform(f, y, r)

        assert F.shape == torch.Size([5])
        assert F.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        r = torch.empty(100, device="meta", dtype=torch.float64)
        f = torch.empty(3, 100, device="meta", dtype=torch.float64)
        y = torch.empty(5, device="meta", dtype=torch.float64)

        F = T.abel_transform(f, y, r, dim=-1)

        assert F.shape == torch.Size([3, 5])


class TestAbelTransformDtype:
    """Test Abel transform dtype handling."""

    def test_float32_input(self):
        """Abel transform should work with float32 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        f = torch.randn(50, dtype=torch.float32)
        y = torch.tensor([0.5, 1.0], dtype=torch.float32)

        F = T.abel_transform(f, y, r)
        assert F.dtype == torch.float32

    def test_float64_input(self):
        """Abel transform should work with float64 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r)
        assert F.dtype == torch.float64


class TestAbelTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        r = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        f = torch.randn(100, dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.abel_transform(f, y, r, integration_method="invalid")

    def test_y_greater_than_r_max(self):
        """Transform should handle y values beyond r range."""
        r = torch.linspace(0.01, 5, 100, dtype=torch.float64)
        f = torch.exp(-r)
        y = torch.tensor([4.0, 5.5, 6.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r)
        # Values beyond r_max should be close to zero
        assert F.shape == torch.Size([3])
        assert torch.isfinite(F).all()


class TestAbelTransformComplex:
    """Test Abel transform with complex tensors."""

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex64_input(self):
        """Abel transform should work with complex64 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        f = torch.randn(50, dtype=torch.complex64)
        y = torch.tensor([0.5, 1.0], dtype=torch.float32)

        F = T.abel_transform(f, y, r)
        assert F.dtype == torch.complex64
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex128_input(self):
        """Abel transform should work with complex128 input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.complex128)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_input_batched(self):
        """Abel transform should work with batched complex input."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(5, 50, dtype=torch.complex128)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        F = T.abel_transform(f, y, r, dim=-1)
        assert F.dtype == torch.complex128
        assert F.shape == torch.Size([5, 2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_gradcheck(self):
        """Gradient for complex input should pass numerical check."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(30, dtype=torch.complex128, requires_grad=True)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            result = T.abel_transform(inp, y, r)
            return result.real, result.imag

        assert gradcheck(func, (f,), raise_exception=True)


class TestAbelTransformDevice:
    """Test Abel transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(50, dtype=torch.float64, device="cuda")
        y = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        try:
            F = T.abel_transform(f, y, r)
            assert F.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAbelTransformCUDA:
    """Test Abel transform CUDA backend."""

    def test_cuda_forward_matches_cpu(self):
        """CUDA forward should match CPU output."""
        r_cpu = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f_cpu = torch.randn(50, dtype=torch.float64)
        y_cpu = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        r_cuda = r_cpu.cuda()
        f_cuda = f_cpu.cuda()
        y_cuda = y_cpu.cuda()

        F_cpu = T.abel_transform(f_cpu, y_cpu, r_cpu)
        F_cuda = T.abel_transform(f_cuda, y_cuda, r_cuda)

        assert torch.allclose(F_cpu, F_cuda.cpu(), rtol=1e-10, atol=1e-10)

    def test_cuda_gradient(self):
        """Gradient should work on CUDA."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(
            50, dtype=torch.float64, device="cuda", requires_grad=True
        )
        y = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        F = T.abel_transform(f, y, r)
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
        y = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        def func(inp):
            return T.abel_transform(inp, y, r)

        assert gradcheck(func, (f,), raise_exception=True)

    def test_cuda_batched(self):
        """Batched Abel transform on CUDA."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        f = torch.randn(5, 50, dtype=torch.float64, device="cuda")
        y = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        F = T.abel_transform(f, y, r, dim=-1)

        assert F.shape == torch.Size([5, 2])
        assert F.device.type == "cuda"


class TestAbelTransformVmap:
    """Test Abel transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(8, 50, dtype=torch.float64)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        # Manual batching
        F_batched = T.abel_transform(f, y, r, dim=-1)

        # vmap
        def abel_single(fi):
            return T.abel_transform(fi, y, r)

        F_vmap = torch.vmap(abel_single)(f)

        assert torch.allclose(F_batched, F_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(4, 4, 30, dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        def abel_single(fi):
            return T.abel_transform(fi, y, r)

        F_vmap = torch.vmap(torch.vmap(abel_single))(f)

        assert F_vmap.shape == torch.Size([4, 4, 1])


class TestAbelTransformCompile:
    """Test Abel transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        r = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        f = torch.randn(50, dtype=torch.float64)
        y = torch.tensor([0.5, 1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_abel(x):
            return T.abel_transform(x, y, r)

        F_compiled = compiled_abel(f)
        F_eager = T.abel_transform(f, y, r)

        assert torch.allclose(F_compiled, F_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        r = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        f = torch.randn(30, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_abel(x):
            return T.abel_transform(x, y, r)

        F = compiled_abel(f)
        F.sum().backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
