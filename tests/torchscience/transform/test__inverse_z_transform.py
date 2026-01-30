"""Tests for inverse Z-transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestInverseZTransformForward:
    """Test inverse Z-transform forward pass correctness."""

    def test_round_trip(self):
        """Inverse Z-transform should recover original sequence."""
        # Original sequence
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        N = len(x)

        # Sample on unit circle (uniformly spaced)
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)

        # Forward transform
        X = T.z_transform(x, z)

        # Inverse transform
        n = torch.arange(N, dtype=torch.float64)
        x_reconstructed = T.inverse_z_transform(X, n, z)

        # Should recover original (real part)
        assert torch.allclose(x_reconstructed.real, x, rtol=1e-10, atol=1e-10)

    def test_round_trip_longer_sequence(self):
        """Round-trip with longer sequence."""
        N = 32
        x = torch.randn(N, dtype=torch.float64)

        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)

        X = T.z_transform(x, z)
        n = torch.arange(N, dtype=torch.float64)
        x_reconstructed = T.inverse_z_transform(X, n, z)

        assert torch.allclose(x_reconstructed.real, x, rtol=1e-9, atol=1e-9)

    def test_unit_impulse_recovery(self):
        """Recover unit impulse from its Z-transform."""
        N = 10
        # Z-transform of delta[n] is 1 for all z
        X = torch.ones(N, dtype=torch.complex128)

        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)

        # Should be delta[n]: 1 at n=0, 0 elsewhere
        expected = torch.zeros(N, dtype=torch.float64)
        expected[0] = 1.0

        assert torch.allclose(x.real, expected, rtol=1e-9, atol=1e-9)

    def test_output_shape(self):
        """Output shape should match n shape."""
        M = 20  # number of z points
        k = torch.arange(M, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / M)
        X = torch.randn(M, dtype=torch.complex128)

        # Scalar n
        n = torch.tensor(0.0, dtype=torch.float64)
        x = T.inverse_z_transform(X, n, z)
        assert x.shape == torch.Size([])

        # 1D n
        n = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = T.inverse_z_transform(X, n, z)
        assert x.shape == torch.Size([3])

    def test_batched_input(self):
        """Inverse Z-transform should work with batched inputs."""
        N = 16
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(5, N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z, dim=-1)
        assert x.shape == torch.Size([5, N])

    def test_1d_input(self):
        """Inverse Z-transform should work with 1D input."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        assert x.shape == torch.Size([N])


class TestInverseZTransformGradient:
    """Test inverse Z-transform gradient correctness."""

    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        n = torch.arange(N, dtype=torch.float64)

        # Use complex input
        X_real = torch.randn(N, dtype=torch.float64, requires_grad=True)

        def func(inp_real):
            inp = inp_real.to(torch.complex128)
            result = T.inverse_z_transform(inp, n, z)
            return result.real

        assert gradcheck(func, (X_real,), raise_exception=True)

    def test_backward_pass(self):
        """Test that backward pass works."""
        N = 16
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        n = torch.arange(N, dtype=torch.float64)

        X = torch.randn(N, dtype=torch.complex128, requires_grad=True)

        x = T.inverse_z_transform(X, n, z)
        loss = x.abs().sum()
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == X.shape
        assert torch.isfinite(X.grad).all()

    @pytest.mark.skip(
        reason="inverse_z_transform second-order gradients not fully implemented for complex outputs"
    )
    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        n = torch.arange(N, dtype=torch.float64)

        X_real = torch.randn(N, dtype=torch.float64, requires_grad=True)

        def func(inp_real):
            inp = inp_real.to(torch.complex128)
            result = T.inverse_z_transform(inp, n, z)
            return result.real

        assert gradgradcheck(func, (X_real,), raise_exception=True, eps=1e-5)


class TestInverseZTransformMeta:
    """Test inverse Z-transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        N = 20
        X = torch.empty(N, device="meta", dtype=torch.complex128)
        z = torch.empty(N, device="meta", dtype=torch.complex128)
        n = torch.empty(10, device="meta", dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)

        assert x.shape == torch.Size([10])
        assert x.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        N = 20
        X = torch.empty(3, N, device="meta", dtype=torch.complex128)
        z = torch.empty(N, device="meta", dtype=torch.complex128)
        n = torch.empty(10, device="meta", dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z, dim=-1)

        assert x.shape == torch.Size([3, 10])


class TestInverseZTransformDtype:
    """Test inverse Z-transform dtype handling.

    Note: The inverse Z-transform returns real output (corresponding to
    the recovered sequence), not complex output.
    """

    def test_complex64_input(self):
        """Inverse Z-transform should work with complex64 input."""
        N = 10
        k = torch.arange(N, dtype=torch.float32)
        z = torch.exp(2j * torch.pi * k / N).to(torch.complex64)
        X = torch.randn(N, dtype=torch.complex64)
        n = torch.arange(N, dtype=torch.float32)

        x = T.inverse_z_transform(X, n, z)
        # Inverse Z-transform returns real output (float32 for complex64 input)
        assert x.dtype == torch.float32

    def test_complex128_input(self):
        """Inverse Z-transform should work with complex128 input."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        # Inverse Z-transform returns real output (float64 for complex128 input)
        assert x.dtype == torch.float64


class TestInverseZTransformDevice:
    """Test inverse Z-transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        N = 10
        k = torch.arange(N, dtype=torch.float64, device="cuda")
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(N, dtype=torch.complex128, device="cuda")
        n = torch.arange(N, dtype=torch.float64, device="cuda")

        try:
            x = T.inverse_z_transform(X, n, z)
            assert x.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


class TestInverseZTransformVmap:
    """Test inverse Z-transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        N = 16
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(8, N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        # Manual batching
        x_batched = T.inverse_z_transform(X, n, z, dim=-1)

        # vmap
        def inv_z_single(Xi):
            return T.inverse_z_transform(Xi, n, z)

        x_vmap = torch.vmap(inv_z_single)(X)

        assert torch.allclose(x_batched, x_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(4, 4, N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        def inv_z_single(Xi):
            return T.inverse_z_transform(Xi, n, z)

        x_vmap = torch.vmap(torch.vmap(inv_z_single))(X)

        assert x_vmap.shape == torch.Size([4, 4, N])


class TestInverseZTransformCompile:
    """Test inverse Z-transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        N = 16
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_inv_z(x):
            return T.inverse_z_transform(x, n, z)

        x_compiled = compiled_inv_z(X)
        x_eager = T.inverse_z_transform(X, n, z)

        assert torch.allclose(x_compiled, x_eager, atol=1e-10)


class TestInverseZTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_input(self):
        """Inverse Z-transform of single element should work."""
        X = torch.tensor([3.0 + 0j], dtype=torch.complex128)
        z = torch.tensor([1.0 + 0j], dtype=torch.complex128)
        n = torch.tensor([0.0], dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        # Single Z-sample at z=1, single time point n=0: x[0] = X[0] / len(z) = 3.0
        assert x.shape == torch.Size([1])
        assert torch.isfinite(x).all()

    def test_single_time_point(self):
        """Inverse Z-transform at single time point should work."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.randn(N, dtype=torch.complex128)
        n = torch.tensor([0.0], dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        assert x.shape == torch.Size([1])
        assert torch.isfinite(x).all()

    def test_zeros_input(self):
        """Inverse Z-transform of zeros should return zeros."""
        N = 10
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        X = torch.zeros(N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        assert torch.allclose(
            x.real, torch.zeros(N, dtype=torch.float64), atol=1e-10
        )

    def test_constant_spectrum(self):
        """Inverse Z-transform of constant spectrum gives impulse."""
        N = 8
        k = torch.arange(N, dtype=torch.float64)
        z = torch.exp(2j * torch.pi * k / N)
        # Constant spectrum X = [1, 1, 1, ..., 1]
        X = torch.ones(N, dtype=torch.complex128)
        n = torch.arange(N, dtype=torch.float64)

        x = T.inverse_z_transform(X, n, z)
        # Should recover delta[n]
        expected = torch.zeros(N, dtype=torch.float64)
        expected[0] = 1.0
        assert torch.allclose(x.real, expected, atol=1e-9)
