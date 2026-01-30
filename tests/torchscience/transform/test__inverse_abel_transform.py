"""Tests for inverse Abel transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestInverseAbelTransformForward:
    """Test inverse Abel transform forward pass correctness."""

    def test_round_trip_gaussian(self):
        """Round-trip with Gaussian should approximately recover original."""
        # Original radial function
        r = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        f = torch.exp(-(r**2))

        # Forward Abel transform
        y = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = T.abel_transform(f, y, r)

        # Inverse Abel transform
        r_out = torch.linspace(0.01, 8, 400, dtype=torch.float64)
        f_reconstructed = T.inverse_abel_transform(F, r_out, y)

        # Compare at interior points (away from boundaries where errors are larger)
        # Find indices where r_out is in the middle range
        mask = (r_out > 0.5) & (r_out < 5)
        r_compare = r_out[mask]
        f_recon_compare = f_reconstructed[mask]

        # Compute expected values
        f_expected = torch.exp(-(r_compare**2))

        # Large tolerance due to numerical differentiation in inverse
        assert torch.allclose(f_recon_compare, f_expected, rtol=0.3, atol=0.1)

    def test_output_shape(self):
        """Output shape should match r shape."""
        y = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(100, dtype=torch.float64)

        # Scalar r
        r = torch.tensor(1.0, dtype=torch.float64)
        f = T.inverse_abel_transform(F, r, y)
        assert f.shape == torch.Size([])

        # 1D r
        r = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        f = T.inverse_abel_transform(F, r, y)
        assert f.shape == torch.Size([3])

    def test_batched_input(self):
        """Inverse Abel transform should work with batched inputs."""
        y = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(5, 100, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y, dim=-1)
        assert f.shape == torch.Size([5, 2])

    def test_1d_input(self):
        """Inverse Abel transform should work with 1D input."""
        y = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.exp(-y)
        r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y)
        assert f.shape == torch.Size([3])

    def test_different_integration_methods(self):
        """Different integration methods should give similar results."""
        y = torch.linspace(0.01, 10, 500, dtype=torch.float64)
        F = torch.exp(-y)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f_trap = T.inverse_abel_transform(
            F, r, y, integration_method="trapezoidal"
        )
        f_simp = T.inverse_abel_transform(
            F, r, y, integration_method="simpson"
        )

        # Should be close to each other
        assert torch.allclose(f_trap, f_simp, rtol=0.1)


class TestInverseAbelTransformGradient:
    """Test inverse Abel transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        y = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(30, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        def func(inp):
            return T.inverse_abel_transform(inp, r, y)

        assert gradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        y = torch.linspace(0.1, 5, 20, dtype=torch.float64)
        F = torch.randn(20, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.inverse_abel_transform(inp, r, y)

        assert gradgradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        y = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(3, 30, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([1.0], dtype=torch.float64)

        def func(inp):
            return T.inverse_abel_transform(inp, r, y, dim=-1)

        assert gradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(50, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y)
        loss = f.sum()
        loss.backward()

        assert F.grad is not None
        assert F.grad.shape == F.shape
        assert torch.isfinite(F.grad).all()
        # Verify gradient is non-zero (was the P0 bug)
        assert F.grad.abs().max() > 1e-10

    def test_gradient_1d_input(self):
        """Test gradient works with 1D input."""
        y = torch.linspace(0.1, 5, 40, dtype=torch.float64)
        F = torch.randn(40, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y)
        loss = f.sum()
        loss.backward()

        assert F.grad is not None
        assert F.grad.shape == F.shape
        assert F.grad.abs().max() > 1e-10


class TestInverseAbelTransformMeta:
    """Test inverse Abel transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        y = torch.empty(100, device="meta", dtype=torch.float64)
        F = torch.empty(100, device="meta", dtype=torch.float64)
        r = torch.empty(5, device="meta", dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y)

        assert f.shape == torch.Size([5])
        assert f.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        y = torch.empty(100, device="meta", dtype=torch.float64)
        F = torch.empty(3, 100, device="meta", dtype=torch.float64)
        r = torch.empty(5, device="meta", dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y, dim=-1)

        assert f.shape == torch.Size([3, 5])


class TestInverseAbelTransformDtype:
    """Test inverse Abel transform dtype handling."""

    def test_float32_input(self):
        """Inverse Abel transform should work with float32 input."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float32)
        F = torch.randn(50, dtype=torch.float32)
        r = torch.tensor([0.5, 1.0], dtype=torch.float32)

        f = T.inverse_abel_transform(F, r, y)
        assert f.dtype == torch.float32

    def test_float64_input(self):
        """Inverse Abel transform should work with float64 input."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(50, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        f = T.inverse_abel_transform(F, r, y)
        assert f.dtype == torch.float64


class TestInverseAbelTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_integration_method(self):
        """Should raise error for invalid integration method."""
        y = torch.linspace(0.01, 10, 100, dtype=torch.float64)
        F = torch.randn(100, dtype=torch.float64)
        r = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="integration_method"):
            T.inverse_abel_transform(F, r, y, integration_method="invalid")


class TestInverseAbelTransformDevice:
    """Test inverse Abel transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        F = torch.randn(50, dtype=torch.float64, device="cuda")
        r = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        try:
            f = T.inverse_abel_transform(F, r, y)
            assert f.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestInverseAbelTransformCUDA:
    """Test inverse Abel transform CUDA backend."""

    def test_cuda_forward_matches_cpu(self):
        """CUDA forward should match CPU output."""
        y_cpu = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F_cpu = torch.randn(50, dtype=torch.float64)
        r_cpu = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

        y_cuda = y_cpu.cuda()
        F_cuda = F_cpu.cuda()
        r_cuda = r_cpu.cuda()

        f_cpu = T.inverse_abel_transform(F_cpu, r_cpu, y_cpu)
        f_cuda = T.inverse_abel_transform(F_cuda, r_cuda, y_cuda)

        assert torch.allclose(f_cpu, f_cuda.cpu(), rtol=1e-10, atol=1e-10)

    def test_cuda_gradient(self):
        """Gradient should work on CUDA."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        F = torch.randn(
            50, dtype=torch.float64, device="cuda", requires_grad=True
        )
        r = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        f = T.inverse_abel_transform(F, r, y)
        loss = f.sum()
        loss.backward()

        assert F.grad is not None
        assert F.grad.device.type == "cuda"

    def test_cuda_gradcheck(self):
        """Gradient check on CUDA."""
        y = torch.linspace(0.1, 5, 30, dtype=torch.float64, device="cuda")
        F = torch.randn(
            30, dtype=torch.float64, device="cuda", requires_grad=True
        )
        r = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        def func(inp):
            return T.inverse_abel_transform(inp, r, y)

        assert gradcheck(
            func, (F,), raise_exception=True, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_cuda_batched(self):
        """Batched inverse Abel transform on CUDA."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64, device="cuda")
        F = torch.randn(5, 50, dtype=torch.float64, device="cuda")
        r = torch.tensor([0.5, 1.0], dtype=torch.float64, device="cuda")

        f = T.inverse_abel_transform(F, r, y, dim=-1)

        assert f.shape == torch.Size([5, 2])
        assert f.device.type == "cuda"


class TestInverseAbelTransformVmap:
    """Test inverse Abel transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(8, 50, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        # Manual batching
        f_batched = T.inverse_abel_transform(F, r, y, dim=-1)

        # vmap
        def inv_abel_single(Fi):
            return T.inverse_abel_transform(Fi, r, y)

        f_vmap = torch.vmap(inv_abel_single)(F)

        assert torch.allclose(f_batched, f_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        y = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(4, 4, 30, dtype=torch.float64)
        r = torch.tensor([1.0], dtype=torch.float64)

        def inv_abel_single(Fi):
            return T.inverse_abel_transform(Fi, r, y)

        f_vmap = torch.vmap(torch.vmap(inv_abel_single))(F)

        assert f_vmap.shape == torch.Size([4, 4, 1])


class TestInverseAbelTransformCompile:
    """Test inverse Abel transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        y = torch.linspace(0.1, 5, 50, dtype=torch.float64)
        F = torch.randn(50, dtype=torch.float64)
        r = torch.tensor([0.5, 1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_inv_abel(x):
            return T.inverse_abel_transform(x, r, y)

        f_compiled = compiled_inv_abel(F)
        f_eager = T.inverse_abel_transform(F, r, y)

        assert torch.allclose(f_compiled, f_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        y = torch.linspace(0.1, 5, 30, dtype=torch.float64)
        F = torch.randn(30, dtype=torch.float64, requires_grad=True)
        r = torch.tensor([1.0], dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_inv_abel(x):
            return T.inverse_abel_transform(x, r, y)

        f = compiled_inv_abel(F)
        f.sum().backward()

        assert F.grad is not None
        assert F.grad.shape == F.shape
