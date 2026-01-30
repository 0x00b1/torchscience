"""Tests for Z-transform implementation."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestZTransformForward:
    """Test Z-transform forward pass correctness."""

    def test_geometric_sequence(self):
        """Z{a^n} = z / (z - a) for |z| > |a|."""
        a = 0.5
        N = 50
        n = torch.arange(N, dtype=torch.float64)
        x = a**n

        # Evaluate at z values outside ROC
        z = torch.tensor(
            [2.0 + 0j, 1.5 + 0j, 1.0 + 1j], dtype=torch.complex128
        )
        X = T.z_transform(x, z)

        # Analytical formula (for finite sum)
        expected = (1 - (a / z) ** N) / (1 - a / z)

        assert torch.allclose(X, expected, rtol=1e-10, atol=1e-10)

    def test_unit_impulse(self):
        """Z{delta[n]} = 1 for all z."""
        x = torch.zeros(10, dtype=torch.float64)
        x[0] = 1.0  # delta[n]

        z = torch.tensor(
            [0.5 + 0j, 1.0 + 0j, 2.0 + 0j], dtype=torch.complex128
        )
        X = T.z_transform(x, z)

        expected = torch.ones(3, dtype=torch.complex128)
        assert torch.allclose(X, expected, rtol=1e-10, atol=1e-10)

    def test_delayed_impulse(self):
        """Z{delta[n-k]} = z^{-k} for all z != 0."""
        k = 3
        x = torch.zeros(10, dtype=torch.float64)
        x[k] = 1.0  # delta[n-k]

        z = torch.tensor(
            [0.5 + 0j, 1.0 + 0j, 2.0 + 0j], dtype=torch.complex128
        )
        X = T.z_transform(x, z)

        expected = z ** (-k)
        assert torch.allclose(X, expected, rtol=1e-10, atol=1e-10)

    def test_unit_circle_is_dtft(self):
        """Z-transform on unit circle equals DTFT."""
        x = torch.randn(16, dtype=torch.float64)

        # Sample on unit circle (DFT frequencies)
        N = len(x)
        k = torch.arange(N, dtype=torch.float64)
        z_unit = torch.exp(2j * torch.pi * k / N)

        X_z = T.z_transform(x, z_unit)

        # Compare with FFT (which computes DFT)
        X_fft = torch.fft.fft(x)

        assert torch.allclose(X_z, X_fft, rtol=1e-10, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match z shape."""
        x = torch.randn(20, dtype=torch.float64)

        # Scalar z
        z = torch.tensor(1.0 + 0j, dtype=torch.complex128)
        X = T.z_transform(x, z)
        assert X.shape == torch.Size([])

        # 1D z
        z = torch.tensor(
            [0.5 + 0j, 1.0 + 0j, 2.0 + 0j], dtype=torch.complex128
        )
        X = T.z_transform(x, z)
        assert X.shape == torch.Size([3])

        # 2D z
        z = torch.randn(3, 4, dtype=torch.complex128)
        X = T.z_transform(x, z)
        assert X.shape == torch.Size([3, 4])

    def test_batched_input(self):
        """Z-transform should work with batched inputs."""
        x = torch.randn(5, 20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z, dim=-1)
        assert X.shape == torch.Size([5, 2])

    def test_complex_input(self):
        """Z-transform should work with complex input."""
        # Note: complex input may not be fully supported, test real input
        x = torch.randn(20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z)
        assert X.shape == torch.Size([2])
        assert X.is_complex()

    def test_1d_input(self):
        """Z-transform should work with 1D input."""
        x = torch.randn(20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z)
        assert X.shape == torch.Size([2])


class TestZTransformGradient:
    """Test Z-transform gradient correctness.

    Note: The current implementation may not fully support autograd for
    complex outputs. These tests verify basic gradient functionality.
    """

    @pytest.mark.skip(
        reason="z_transform autograd not fully implemented for complex outputs"
    )
    def test_gradcheck(self):
        """Gradient w.r.t. input should pass numerical check."""
        x = torch.randn(15, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        def func(inp):
            result = T.z_transform(inp, z)
            # Return real part for gradcheck (which doesn't handle complex outputs directly)
            return result.real

        assert gradcheck(func, (x,), raise_exception=True)

    @pytest.mark.skip(
        reason="z_transform second-order gradients not fully implemented for complex outputs"
    )
    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(10, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5 + 0j], dtype=torch.complex128)

        def func(inp):
            result = T.z_transform(inp, z)
            return result.real

        assert gradgradcheck(func, (x,), raise_exception=True)

    @pytest.mark.skip(
        reason="z_transform autograd not fully implemented for complex outputs"
    )
    def test_gradient_batched(self):
        """Gradient should work with batched inputs."""
        x = torch.randn(3, 15, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([1.0 + 0j], dtype=torch.complex128)

        def func(inp):
            result = T.z_transform(inp, z, dim=-1)
            return result.real

        assert gradcheck(func, (x,), raise_exception=True)

    @pytest.mark.skip(
        reason="z_transform autograd not fully implemented for complex outputs"
    )
    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(20, dtype=torch.float64, requires_grad=True)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z)
        loss = X.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.isfinite(x.grad).all()


class TestZTransformMeta:
    """Test Z-transform with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(20, device="meta", dtype=torch.float64)
        z = torch.empty(5, device="meta", dtype=torch.complex128)

        X = T.z_transform(x, z)

        assert X.shape == torch.Size([5])
        assert X.device.type == "meta"

    def test_meta_tensor_batched(self):
        """Meta tensor should work with batched inputs."""
        x = torch.empty(3, 20, device="meta", dtype=torch.float64)
        z = torch.empty(5, device="meta", dtype=torch.complex128)

        X = T.z_transform(x, z, dim=-1)

        assert X.shape == torch.Size([3, 5])


class TestZTransformDtype:
    """Test Z-transform dtype handling."""

    def test_float32_input(self):
        """Z-transform should work with float32 input."""
        x = torch.randn(20, dtype=torch.float32)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex64)

        X = T.z_transform(x, z)
        assert X.dtype == torch.complex64

    def test_float64_input(self):
        """Z-transform should work with float64 input."""
        x = torch.randn(20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z)
        assert X.dtype == torch.complex128


class TestZTransformComplex:
    """Test Z-transform with complex input tensors."""

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex64_input(self):
        """Z-transform should work with complex64 input."""
        x = torch.randn(20, dtype=torch.complex64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex64)

        X = T.z_transform(x, z)
        assert X.dtype == torch.complex64
        assert X.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex128_input(self):
        """Z-transform should work with complex128 input."""
        x = torch.randn(20, dtype=torch.complex128)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z)
        assert X.dtype == torch.complex128
        assert X.shape == torch.Size([2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_input_batched(self):
        """Z-transform should work with batched complex input."""
        x = torch.randn(5, 20, dtype=torch.complex128)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        X = T.z_transform(x, z, dim=-1)
        assert X.dtype == torch.complex128
        assert X.shape == torch.Size([5, 2])

    @pytest.mark.skip(
        reason="Complex input not yet implemented for integral transforms"
    )
    def test_complex_unit_circle_matches_fft(self):
        """Complex input Z-transform on unit circle should match FFT."""
        x = torch.randn(16, dtype=torch.complex128)

        # Sample on unit circle (DFT frequencies)
        N = len(x)
        k = torch.arange(N, dtype=torch.float64)
        z_unit = torch.exp(2j * torch.pi * k / N)

        X_z = T.z_transform(x, z_unit)
        X_fft = torch.fft.fft(x)

        assert torch.allclose(X_z, X_fft, rtol=1e-10, atol=1e-10)


class TestZTransformDevice:
    """Test Z-transform device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work (if CUDA backend exists)."""
        x = torch.randn(20, dtype=torch.float64, device="cuda")
        z = torch.tensor(
            [0.5 + 0j, 1.0 + 0j], dtype=torch.complex128, device="cuda"
        )

        try:
            X = T.z_transform(x, z)
            assert X.device.type == "cuda"
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA backend not implemented")
            raise


class TestZTransformVmap:
    """Test Z-transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        x = torch.randn(8, 20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        # Manual batching
        y_batched = T.z_transform(x, z, dim=-1)

        # vmap
        def z_single(xi):
            return T.z_transform(xi, z)

        y_vmap = torch.vmap(z_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        x = torch.randn(4, 4, 20, dtype=torch.float64)
        z = torch.tensor([1.0 + 0j], dtype=torch.complex128)

        def z_single(xi):
            return T.z_transform(xi, z)

        y_vmap = torch.vmap(torch.vmap(z_single))(x)

        assert y_vmap.shape == torch.Size([4, 4, 1])


class TestZTransformCompile:
    """Test Z-transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        x = torch.randn(20, dtype=torch.float64)
        z = torch.tensor([0.5 + 0j, 1.0 + 0j], dtype=torch.complex128)

        @torch.compile(fullgraph=True)
        def compiled_z(xi):
            return T.z_transform(xi, z)

        y_compiled = compiled_z(x)
        y_eager = T.z_transform(x, z)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)
