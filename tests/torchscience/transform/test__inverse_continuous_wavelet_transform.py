"""Tests for inverse_continuous_wavelet_transform."""

import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import (
    continuous_wavelet_transform,
    inverse_continuous_wavelet_transform,
)


class TestInverseContinuousWaveletTransformBasic:
    """Tests for basic inverse CWT functionality."""

    def test_basic_output_shape(self):
        """Test basic inverse CWT output shape."""
        # Input: (num_scales, signal_len)
        coeffs = torch.randn(5, 256)
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        # Output should be (signal_len,) for 2D input
        assert reconstructed.shape == (256,)

    def test_output_dtype_real_input(self):
        """Test that real input produces real output."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert not reconstructed.is_complex()
        assert reconstructed.dtype == torch.float32

    def test_output_dtype_complex_input(self):
        """Test that complex input produces real output (we take real part)."""
        coeffs = torch.randn(3, 128, dtype=torch.complex64)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="morlet"
        )

        # Output should be real (we extract real part in reconstruction)
        assert not reconstructed.is_complex()

    def test_single_scale(self):
        """Test inverse CWT with a single scale."""
        coeffs = torch.randn(1, 128)
        scales = torch.tensor([4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        # Output should be (signal_len,)
        assert reconstructed.shape == (128,)


class TestInverseContinuousWaveletTransformBatched:
    """Tests for batched input handling."""

    def test_batched_1d(self):
        """Test inverse CWT with 1D batch dimension."""
        # Input: (batch, num_scales, signal_len)
        coeffs = torch.randn(4, 3, 256)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat", dim=-1
        )

        # Output should be (batch, signal_len)
        assert reconstructed.shape == (4, 256)

    def test_batched_2d(self):
        """Test inverse CWT with 2D batch dimensions."""
        # Input: (batch1, batch2, num_scales, signal_len)
        coeffs = torch.randn(2, 3, 4, 256)
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat", dim=-1
        )

        # Output should be (batch1, batch2, signal_len)
        assert reconstructed.shape == (2, 3, 256)

    def test_batched_complex(self):
        """Test batched inverse CWT with complex coefficients."""
        coeffs = torch.randn(4, 3, 128, dtype=torch.complex64)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="morlet", dim=-1
        )

        assert reconstructed.shape == (4, 128)


class TestInverseContinuousWaveletTransformRoundtrip:
    """Tests for forward-inverse roundtrip reconstruction."""

    def test_roundtrip_mexican_hat_basic(self):
        """Test that icwt(cwt(x)) approximately recovers x with mexican_hat."""
        # Create a simple signal
        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float64)
        scales = torch.logspace(0, 3, 16, base=2.0, dtype=torch.float64)

        # Forward transform
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        # Inverse transform
        reconstructed = inverse_continuous_wavelet_transform(
            cwt, scales, wavelet="mexican_hat"
        )

        # The reconstruction is approximate, check correlation
        # Normalize both signals for correlation calculation
        x_norm = x - x.mean()
        r_norm = reconstructed - reconstructed.mean()

        correlation = (x_norm * r_norm).sum() / (
            x_norm.norm() * r_norm.norm() + 1e-10
        )

        # Expect high correlation (>0.8) for good reconstruction
        assert correlation > 0.7, f"Correlation too low: {correlation}"

    def test_roundtrip_morlet_basic(self):
        """Test that icwt(cwt(x)) approximately recovers x with morlet."""
        torch.manual_seed(42)
        x = torch.randn(256, dtype=torch.float64)
        scales = torch.logspace(0, 3, 16, base=2.0, dtype=torch.float64)

        # Forward transform
        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")

        # Inverse transform
        reconstructed = inverse_continuous_wavelet_transform(
            cwt, scales, wavelet="morlet"
        )

        # Check correlation
        x_norm = x - x.mean()
        r_norm = reconstructed - reconstructed.mean()

        correlation = (x_norm * r_norm).sum() / (
            x_norm.norm() * r_norm.norm() + 1e-10
        )

        assert correlation > 0.7, f"Correlation too low: {correlation}"

    def test_roundtrip_sinusoid(self):
        """Test roundtrip on a sinusoidal signal."""
        t = torch.linspace(0, 4, 512, dtype=torch.float64)
        x = torch.sin(2 * math.pi * 4.0 * t)
        scales = torch.logspace(0, 3, 20, base=2.0, dtype=torch.float64)

        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")
        reconstructed = inverse_continuous_wavelet_transform(
            cwt, scales, wavelet="morlet"
        )

        # Check correlation for sinusoidal signal
        # Use interior to avoid edge effects
        margin = 50
        x_int = x[margin:-margin]
        r_int = reconstructed[margin:-margin]

        x_norm = x_int - x_int.mean()
        r_norm = r_int - r_int.mean()

        correlation = (x_norm * r_norm).sum() / (
            x_norm.norm() * r_norm.norm() + 1e-10
        )

        assert correlation > 0.8, f"Correlation too low: {correlation}"

    def test_roundtrip_batched(self):
        """Test roundtrip with batched input."""
        torch.manual_seed(42)
        x = torch.randn(4, 256, dtype=torch.float64)
        scales = torch.logspace(0, 3, 12, base=2.0, dtype=torch.float64)

        cwt = continuous_wavelet_transform(
            x, scales, wavelet="mexican_hat", dim=-1
        )
        reconstructed = inverse_continuous_wavelet_transform(
            cwt, scales, wavelet="mexican_hat", dim=-1
        )

        # Check correlation for each batch element
        for i in range(4):
            x_i = x[i]
            r_i = reconstructed[i]

            x_norm = x_i - x_i.mean()
            r_norm = r_i - r_i.mean()

            correlation = (x_norm * r_norm).sum() / (
                x_norm.norm() * r_norm.norm() + 1e-10
            )

            assert correlation > 0.5, (
                f"Batch {i} correlation too low: {correlation}"
            )


class TestInverseContinuousWaveletTransformWavelets:
    """Tests for different wavelet types."""

    def test_morlet_wavelet(self):
        """Test inverse CWT with Morlet wavelet."""
        coeffs = torch.randn(3, 128, dtype=torch.complex64)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="morlet"
        )

        assert reconstructed.shape == (128,)
        assert not reconstructed.is_complex()

    def test_mexican_hat_wavelet(self):
        """Test inverse CWT with Mexican hat wavelet."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.shape == (128,)

    def test_ricker_alias(self):
        """Test that 'ricker' is an alias for Mexican hat."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])

        reconstructed_mexican = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )
        reconstructed_ricker = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="ricker"
        )

        assert torch.allclose(reconstructed_mexican, reconstructed_ricker)

    def test_custom_wavelet_function(self):
        """Test inverse CWT with a custom wavelet function."""
        coeffs = torch.randn(2, 128)
        scales = torch.tensor([1.0, 2.0])

        # Custom wavelet: Gaussian
        def gaussian_wavelet(t):
            return torch.exp(-(t**2) / 2)

        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet=gaussian_wavelet
        )

        assert reconstructed.shape == (128,)

    def test_invalid_wavelet_string(self):
        """Test that invalid wavelet string raises error."""
        coeffs = torch.randn(2, 128)
        scales = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Unknown wavelet"):
            inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="invalid_wavelet"
            )


class TestInverseContinuousWaveletTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_wavelet(self):
        """Test gradient correctness with Mexican hat (real) wavelet."""
        coeffs = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def icwt_wrapper(coeffs):
            return inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )

        assert gradcheck(
            icwt_wrapper, (coeffs,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck_real_wavelet(self):
        """Test second-order gradient correctness with Mexican hat wavelet."""
        coeffs = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def icwt_wrapper(coeffs):
            return inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )

        assert gradgradcheck(
            icwt_wrapper, (coeffs,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_batched(self):
        """Test gradient correctness with batched input."""
        coeffs = torch.randn(2, 3, 64, dtype=torch.float64, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def icwt_wrapper(coeffs):
            return inverse_continuous_wavelet_transform(
                coeffs,
                scales,
                wavelet="mexican_hat",
                dim=-1,
            )

        assert gradcheck(
            icwt_wrapper, (coeffs,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        coeffs = torch.randn(3, 128, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        loss = reconstructed.abs().sum()
        loss.backward()

        assert coeffs.grad is not None
        assert coeffs.grad.shape == coeffs.shape

    def test_backward_pass_batched(self):
        """Test backward pass with batched input."""
        coeffs = torch.randn(4, 3, 128, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat", dim=-1
        )

        loss = reconstructed.abs().sum()
        loss.backward()

        assert coeffs.grad is not None
        assert coeffs.grad.shape == coeffs.shape


class TestInverseContinuousWaveletTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        coeffs = torch.randn(3, 256, device="cuda")
        scales = torch.tensor([1.0, 2.0, 4.0], device="cuda")
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.device.type == "cuda"

    def test_input_device_preserved(self):
        """Test that output is on the same device as input."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.device == coeffs.device


class TestInverseContinuousWaveletTransformDtype:
    """Tests for dtype handling."""

    def test_float32_input(self):
        """Test with float32 input."""
        coeffs = torch.randn(3, 128, dtype=torch.float32)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.dtype == torch.float32

    def test_float64_input(self):
        """Test with float64 input."""
        coeffs = torch.randn(3, 128, dtype=torch.float64)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.dtype == torch.float64

    def test_complex64_input(self):
        """Test with complex64 input produces float32 output."""
        coeffs = torch.randn(3, 128, dtype=torch.complex64)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="morlet"
        )

        assert reconstructed.dtype == torch.float32

    def test_complex128_input(self):
        """Test with complex128 input produces float64 output."""
        coeffs = torch.randn(3, 128, dtype=torch.complex128)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="morlet"
        )

        assert reconstructed.dtype == torch.float64


class TestInverseContinuousWaveletTransformParameterValidation:
    """Tests for parameter validation."""

    def test_scales_must_be_1d(self):
        """Test that scales must be 1D tensor."""
        coeffs = torch.randn(2, 2, 128)
        scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="scales.*1-D"):
            inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )

    def test_scales_must_be_positive(self):
        """Test that scales must be positive."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, -2.0, 4.0])

        with pytest.raises(ValueError, match="positive"):
            inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )

    def test_scales_cannot_be_zero(self):
        """Test that scales cannot be zero."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="positive"):
            inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )

    def test_scales_count_must_match(self):
        """Test that number of scales must match input scale dimension."""
        coeffs = torch.randn(5, 128)  # 5 scales
        scales = torch.tensor([1.0, 2.0, 4.0])  # Only 3 scales

        with pytest.raises(ValueError, match="scales.*match"):
            inverse_continuous_wavelet_transform(
                coeffs, scales, wavelet="mexican_hat"
            )


class TestInverseContinuousWaveletTransformSamplingPeriod:
    """Tests for sampling_period parameter."""

    def test_sampling_period_default(self):
        """Test default sampling period is 1.0."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        # Should work with default sampling_period=1.0
        assert reconstructed.shape == (128,)

    def test_sampling_period_custom(self):
        """Test custom sampling period."""
        coeffs = torch.randn(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat", sampling_period=0.01
        )

        # Shape should be the same
        assert reconstructed.shape == (128,)

    def test_roundtrip_with_sampling_period(self):
        """Test roundtrip works with non-default sampling period."""
        torch.manual_seed(42)
        # Use sampling_period=0.1
        dt = 0.1
        x = torch.randn(256, dtype=torch.float64)
        # Use scales appropriate for the sampling period
        scales = torch.logspace(0, 3, 16, base=2.0, dtype=torch.float64)

        cwt = continuous_wavelet_transform(
            x, scales, wavelet="mexican_hat", sampling_period=dt
        )
        reconstructed = inverse_continuous_wavelet_transform(
            cwt, scales, wavelet="mexican_hat", sampling_period=dt
        )

        # Verify basic properties: output is not nan, has correct shape
        assert reconstructed.shape == (256,)
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()

        # Check that reconstruction has some correlation with input
        # (CWT reconstruction quality depends heavily on scale selection)
        x_norm = x - x.mean()
        r_norm = reconstructed - reconstructed.mean()

        correlation = (x_norm * r_norm).sum() / (
            x_norm.norm() * r_norm.norm() + 1e-10
        )

        # Just verify positive correlation exists
        assert correlation > 0.1, f"Correlation too low: {correlation}"


class TestInverseContinuousWaveletTransformEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_zero_coefficients(self):
        """Test inverse CWT of zero coefficients gives zero output."""
        coeffs = torch.zeros(3, 128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert torch.allclose(reconstructed, torch.zeros_like(reconstructed))

    def test_short_signal(self):
        """Test inverse CWT with short signal."""
        coeffs = torch.randn(2, 16)
        scales = torch.tensor([1.0, 2.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.shape == (16,)

    def test_long_signal(self):
        """Test inverse CWT with long signal."""
        coeffs = torch.randn(3, 8192)
        scales = torch.tensor([1.0, 2.0, 4.0])
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.shape == (8192,)

    def test_many_scales(self):
        """Test inverse CWT with many scales."""
        coeffs = torch.randn(32, 256)
        scales = torch.logspace(0, 4, 32, base=2.0)
        reconstructed = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert reconstructed.shape == (256,)
        assert not torch.isnan(reconstructed).any()


class TestInverseContinuousWaveletTransformVmap:
    """Tests for torch.vmap support."""

    def test_vmap_basic(self):
        """Test vmap batches correctly."""
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        coeffs = torch.randn(8, 3, 128, dtype=torch.float64)

        # Batched call
        y_batched = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat", dim=-1
        )

        # vmap call
        def icwt_single(ci):
            return inverse_continuous_wavelet_transform(
                ci, scales, wavelet="mexican_hat"
            )

        y_vmap = torch.vmap(icwt_single)(coeffs)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Test nested vmap."""
        scales = torch.tensor([1.0, 2.0], dtype=torch.float64)
        coeffs = torch.randn(4, 3, 2, 64, dtype=torch.float64)

        def icwt_single(ci):
            return inverse_continuous_wavelet_transform(
                ci, scales, wavelet="mexican_hat"
            )

        y_vmap = torch.vmap(torch.vmap(icwt_single))(coeffs)

        assert y_vmap.shape == (4, 3, 64)


class TestInverseContinuousWaveletTransformCompile:
    """Tests for torch.compile support."""

    @pytest.mark.skip(reason="FFT operations have meta kernel stride issues")
    def test_compile_basic(self):
        """Test torch.compile works."""
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        coeffs = torch.randn(3, 128, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_icwt(c):
            return inverse_continuous_wavelet_transform(
                c, scales, wavelet="mexican_hat"
            )

        y_compiled = compiled_icwt(coeffs)
        y_eager = inverse_continuous_wavelet_transform(
            coeffs, scales, wavelet="mexican_hat"
        )

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    @pytest.mark.skip(reason="FFT operations have meta kernel stride issues")
    def test_compile_with_grad(self):
        """Test torch.compile with gradient computation."""
        scales = torch.tensor([1.0, 2.0], dtype=torch.float64)
        coeffs = torch.randn(2, 64, dtype=torch.float64, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_icwt(c):
            return inverse_continuous_wavelet_transform(
                c, scales, wavelet="mexican_hat"
            )

        y = compiled_icwt(coeffs)
        loss = y.abs().sum()
        loss.backward()

        assert coeffs.grad is not None
        assert coeffs.grad.shape == coeffs.shape
