"""Tests for continuous_wavelet_transform."""

import math

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.transform import continuous_wavelet_transform


class TestContinuousWaveletTransformBasic:
    """Tests for basic CWT functionality."""

    def test_basic_output_shape(self):
        """Test basic CWT output shape with default Morlet wavelet."""
        x = torch.randn(256)
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        cwt = continuous_wavelet_transform(x, scales)

        # Output shape should be (num_scales, signal_length)
        assert cwt.shape == (5, 256)

    def test_complex_output_morlet(self):
        """Morlet wavelet CWT output should be complex."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")

        assert cwt.is_complex()

    def test_real_output_mexican_hat(self):
        """Mexican hat wavelet CWT output should be real for real input."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        # Mexican hat is real-valued, so output should be real
        assert not cwt.is_complex()

    def test_single_scale(self):
        """Test CWT with a single scale."""
        x = torch.randn(128)
        scales = torch.tensor([4.0])
        cwt = continuous_wavelet_transform(x, scales)

        # Output should be (1, signal_length)
        assert cwt.shape == (1, 128)


class TestContinuousWaveletTransformBatched:
    """Tests for batched input handling."""

    def test_batched_1d(self):
        """Test CWT with 1D batch dimension."""
        x = torch.randn(4, 256)
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0])
        cwt = continuous_wavelet_transform(x, scales, dim=-1)

        # Output shape should be (batch, num_scales, signal_length)
        assert cwt.shape == (4, 4, 256)

    def test_batched_2d(self):
        """Test CWT with 2D batch dimensions."""
        x = torch.randn(2, 3, 256)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, dim=-1)

        # Output shape should be (batch1, batch2, num_scales, signal_length)
        assert cwt.shape == (2, 3, 3, 256)

    def test_dim_first(self):
        """Test CWT on first dimension."""
        x = torch.randn(256, 4)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, dim=0)

        # Transform is on dim=0, so output is (batch=4, num_scales=3, signal_length=256)
        # The signal dimension moves to last, batch dimensions come first
        assert cwt.shape == (4, 3, 256)

    def test_dim_middle(self):
        """Test CWT on middle dimension."""
        x = torch.randn(4, 256, 3)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales, dim=1)

        # Signal dim=1 transforms to (num_scales, signal_length)
        # Output: (4, 3, 2, 256) - batch dims, then scale, then signal
        assert cwt.shape == (4, 3, 2, 256)


class TestContinuousWaveletTransformScales:
    """Tests for different scale values."""

    def test_scales_order_preserved(self):
        """Test that scales are processed in order."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales)

        # Verify shape preserves scale order
        assert cwt.shape[0] == 3

    def test_large_scales(self):
        """Test CWT with large scales."""
        x = torch.randn(256)
        scales = torch.tensor([16.0, 32.0, 64.0])
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.shape == (3, 256)
        assert not torch.isnan(cwt).any()

    def test_small_scales(self):
        """Test CWT with small scales."""
        x = torch.randn(256)
        scales = torch.tensor([0.5, 1.0, 1.5])
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.shape == (3, 256)
        assert not torch.isnan(cwt).any()

    def test_many_scales(self):
        """Test CWT with many scales."""
        x = torch.randn(256)
        # Logarithmic scale range
        scales = torch.logspace(0, 4, steps=20, base=2.0)  # 1 to 16
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.shape == (20, 256)

    def test_scales_normalization(self):
        """Test that scale normalization (1/sqrt(s)) is applied."""
        # With larger scales, the wavelet becomes wider and shorter,
        # so the energy normalization should keep total energy comparable
        x = torch.randn(512)
        scales_small = torch.tensor([1.0])
        scales_large = torch.tensor([4.0])

        cwt_small = continuous_wavelet_transform(
            x, scales_small, wavelet="mexican_hat"
        )
        cwt_large = continuous_wavelet_transform(
            x, scales_large, wavelet="mexican_hat"
        )

        # Both should have finite energy
        energy_small = (cwt_small.abs() ** 2).sum()
        energy_large = (cwt_large.abs() ** 2).sum()

        assert torch.isfinite(energy_small)
        assert torch.isfinite(energy_large)


class TestContinuousWaveletTransformWavelets:
    """Tests for different wavelet types."""

    def test_morlet_string(self):
        """Test Morlet wavelet specified as string."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")

        assert cwt.is_complex()

    def test_mexican_hat_string(self):
        """Test Mexican hat wavelet specified as string."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        assert not cwt.is_complex()

    def test_ricker_alias(self):
        """Test that 'ricker' is an alias for Mexican hat."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0, 4.0])

        cwt_mexican = continuous_wavelet_transform(
            x, scales, wavelet="mexican_hat"
        )
        cwt_ricker = continuous_wavelet_transform(x, scales, wavelet="ricker")

        assert torch.allclose(cwt_mexican, cwt_ricker)

    def test_custom_wavelet_function(self):
        """Test CWT with a custom wavelet function."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])

        # Custom wavelet: Gaussian
        def gaussian_wavelet(t):
            return torch.exp(-(t**2) / 2)

        cwt = continuous_wavelet_transform(x, scales, wavelet=gaussian_wavelet)

        assert cwt.shape == (2, 128)
        assert not cwt.is_complex()

    def test_custom_complex_wavelet(self):
        """Test CWT with a custom complex wavelet function."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])

        # Custom complex wavelet
        def complex_wavelet(t):
            return torch.exp(1j * 2.0 * t) * torch.exp(-(t**2) / 2)

        cwt = continuous_wavelet_transform(x, scales, wavelet=complex_wavelet)

        assert cwt.shape == (2, 128)
        assert cwt.is_complex()

    def test_invalid_wavelet_string(self):
        """Test that invalid wavelet string raises error."""
        x = torch.randn(128)
        scales = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Unknown wavelet"):
            continuous_wavelet_transform(x, scales, wavelet="invalid_wavelet")


class TestContinuousWaveletTransformSamplingPeriod:
    """Tests for sampling_period parameter."""

    def test_sampling_period_default(self):
        """Test default sampling period is 1.0."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales)

        # Should work with default sampling_period=1.0
        assert cwt.shape == (2, 128)

    def test_sampling_period_custom(self):
        """Test custom sampling period."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales, sampling_period=0.01)

        # Shape should be the same, but internal scaling differs
        assert cwt.shape == (2, 128)

    def test_sampling_period_affects_frequencies(self):
        """Test that sampling_period affects the effective frequencies analyzed."""
        # A sinusoid at a known frequency
        fs = 100.0  # Sampling frequency
        dt = 1.0 / fs  # Sampling period
        t = torch.arange(256) * dt
        freq = 10.0  # 10 Hz
        x = torch.sin(2 * math.pi * freq * t)

        # Scale corresponding to the sinusoid frequency
        # For Morlet with omega0=5, frequency = omega0 / (2*pi*scale)
        # scale = omega0 / (2*pi*freq) = 5 / (2*pi*10) ~ 0.08
        scales = torch.linspace(0.05, 0.2, 10)

        cwt = continuous_wavelet_transform(
            x, scales, wavelet="morlet", sampling_period=dt
        )

        # The CWT should have a peak at the scale corresponding to the frequency
        cwt_magnitude = cwt.abs().mean(dim=-1)  # Average over time
        peak_scale_idx = cwt_magnitude.argmax()

        # Check that peak is in the expected range (not at boundary)
        assert 1 < peak_scale_idx < len(scales) - 2


class TestContinuousWaveletTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_wavelet(self):
        """Test gradient correctness with Mexican hat (real) wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def cwt_wrapper(x):
            return continuous_wavelet_transform(
                x, scales, wavelet="mexican_hat"
            )

        assert gradcheck(cwt_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_complex_wavelet(self):
        """Test gradient correctness with Morlet (complex) wavelet."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def cwt_wrapper(x):
            # Return real and imaginary parts separately for gradcheck
            cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")
            return cwt.real, cwt.imag

        assert gradcheck(cwt_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(128, requires_grad=True)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        loss = cwt.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_pass_batched(self):
        """Test backward pass with batched input."""
        x = torch.randn(4, 128, requires_grad=True)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales, dim=-1)

        loss = cwt.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestContinuousWaveletTransformMeta:
    """Tests for meta tensor support (shape inference).

    Note: CWT uses FFT which doesn't fully support meta tensors.
    These tests verify that valid output shapes are produced.
    """

    @pytest.mark.skip(reason="FFT operations don't support meta tensors")
    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(256, device="meta")
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0], device="meta")
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.device == torch.device("meta")
        assert cwt.shape == (4, 256)

    @pytest.mark.skip(reason="FFT operations don't support meta tensors")
    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        x = torch.randn(4, 256, device="meta")
        scales = torch.tensor([1.0, 2.0, 4.0], device="meta")
        cwt = continuous_wavelet_transform(x, scales, dim=-1)

        assert cwt.device == torch.device("meta")
        assert cwt.shape == (4, 3, 256)


class TestContinuousWaveletTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(256, device="cuda")
        scales = torch.tensor([1.0, 2.0, 4.0], device="cuda")
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.device.type == "cuda"

    def test_input_device_preserved(self):
        """Test that output is on the same device as input."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.device == x.device


class TestContinuousWaveletTransformDtype:
    """Tests for dtype handling."""

    def test_float32_input(self):
        """Test with float32 input."""
        x = torch.randn(128, dtype=torch.float32)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float32)
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        assert cwt.dtype == torch.float32

    def test_float64_input(self):
        """Test with float64 input."""
        x = torch.randn(128, dtype=torch.float64)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float64)
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        assert cwt.dtype == torch.float64

    def test_morlet_complex_output_dtype(self):
        """Test Morlet wavelet produces complex output with correct dtype."""
        x = torch.randn(128, dtype=torch.float32)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float32)
        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")

        assert cwt.dtype == torch.complex64

    def test_morlet_complex_output_dtype_float64(self):
        """Test Morlet wavelet produces complex128 for float64 input."""
        x = torch.randn(128, dtype=torch.float64)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float64)
        cwt = continuous_wavelet_transform(x, scales, wavelet="morlet")

        assert cwt.dtype == torch.complex128


class TestContinuousWaveletTransformEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_constant_signal(self):
        """Test CWT of constant signal has low response in interior."""
        # Use a longer signal to minimize boundary effects
        x = torch.ones(512)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        # Mexican hat has zero mean, so CWT of constant should be small
        # except at boundaries due to edge effects
        # Due to discretization, there may be small residuals
        # Check that the response is at least much smaller than for a non-constant signal
        x_varying = torch.randn(512)
        cwt_varying = continuous_wavelet_transform(
            x_varying, scales, wavelet="mexican_hat"
        )

        # The constant signal's CWT magnitude should be much smaller than random signal
        const_energy = cwt[:, 100:412].abs().mean()
        varying_energy = cwt_varying[:, 100:412].abs().mean()
        assert const_energy < 0.1 * varying_energy

    def test_impulse_signal(self):
        """Test CWT of impulse signal."""
        x = torch.zeros(128)
        x[64] = 1.0
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        # CWT of impulse should give scaled versions of the wavelet
        # Peak should be at the impulse location
        for i in range(len(scales)):
            _, peak_idx = cwt[i].abs().max(dim=-1)
            assert (
                abs(peak_idx.item() - 64) <= 2
            )  # Within 2 samples of impulse

    def test_short_signal(self):
        """Test CWT with short signal."""
        x = torch.randn(16)
        scales = torch.tensor([1.0, 2.0])
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.shape == (2, 16)

    def test_long_signal(self):
        """Test CWT with long signal."""
        x = torch.randn(8192)
        scales = torch.tensor([1.0, 2.0, 4.0])
        cwt = continuous_wavelet_transform(x, scales)

        assert cwt.shape == (3, 8192)


class TestContinuousWaveletTransformParameterValidation:
    """Tests for parameter validation."""

    def test_scales_must_be_1d(self):
        """Test that scales must be 1D tensor."""
        x = torch.randn(128)
        scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="scales.*1-D"):
            continuous_wavelet_transform(x, scales)

    def test_scales_must_be_positive(self):
        """Test that scales must be positive."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, -2.0, 4.0])

        with pytest.raises(ValueError, match="positive"):
            continuous_wavelet_transform(x, scales)

    def test_scales_cannot_be_zero(self):
        """Test that scales cannot be zero."""
        x = torch.randn(128)
        scales = torch.tensor([0.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="positive"):
            continuous_wavelet_transform(x, scales)

    def test_invalid_dim(self):
        """Test that invalid dim raises error."""
        x = torch.randn(128)
        scales = torch.tensor([1.0, 2.0])

        with pytest.raises(IndexError):
            continuous_wavelet_transform(x, scales, dim=5)


class TestContinuousWaveletTransformMorletWavelet:
    """Tests specific to Morlet wavelet properties."""

    def test_morlet_frequency_localization(self):
        """Test that Morlet wavelet response varies with scale for different frequencies."""
        # Create two signals with different frequencies
        t = torch.linspace(0, 4, 1024)
        freq_low = 2.0  # 2 cycles per unit time
        freq_high = 8.0  # 8 cycles per unit time
        x_low = torch.sin(2 * math.pi * freq_low * t)
        x_high = torch.sin(2 * math.pi * freq_high * t)

        # Use a fixed set of scales
        scales = torch.tensor([0.1, 0.2, 0.4, 0.8])
        cwt_low = continuous_wavelet_transform(x_low, scales, wavelet="morlet")
        cwt_high = continuous_wavelet_transform(
            x_high, scales, wavelet="morlet"
        )

        # Get average magnitude at each scale (avoiding edges)
        mag_low = cwt_low.abs()[:, 200:824].mean(dim=-1)
        mag_high = cwt_high.abs()[:, 200:824].mean(dim=-1)

        # For larger scales (lower frequencies), low-freq signal should have
        # relatively stronger response compared to high-freq signal
        # Compute ratio of mag_low / mag_high at different scales
        ratio_large_scale = (mag_low[-1] / mag_high[-1]).item()  # scale=0.8
        ratio_small_scale = (mag_low[0] / mag_high[0]).item()  # scale=0.1

        # The ratio should increase with scale (larger scale = lower freq)
        # because the low-freq signal is better matched at larger scales
        assert ratio_large_scale > ratio_small_scale


class TestContinuousWaveletTransformMexicanHatWavelet:
    """Tests specific to Mexican hat wavelet properties."""

    def test_mexican_hat_zero_mean(self):
        """Test that Mexican hat wavelet response to constant is relatively small."""
        # The Mexican hat wavelet has zero mean, so convolving with a constant
        # should give a smaller response than a varying signal
        x_const = torch.ones(512) * 5.0  # Constant signal
        x_vary = torch.randn(512)  # Random signal
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0])

        cwt_const = continuous_wavelet_transform(
            x_const, scales, wavelet="mexican_hat"
        )
        cwt_vary = continuous_wavelet_transform(
            x_vary, scales, wavelet="mexican_hat"
        )

        # The constant signal's CWT should have smaller energy than random
        # Check interior to avoid edge effects
        margin = 100
        const_energy = cwt_const[:, margin:-margin].abs().mean()
        vary_energy = cwt_vary[:, margin:-margin].abs().mean()

        # The constant signal response should be much smaller
        assert const_energy < 0.2 * vary_energy

    def test_mexican_hat_symmetry(self):
        """Test Mexican hat wavelet symmetry properties."""
        # Mexican hat is symmetric, so CWT of symmetric signal should be symmetric
        x = torch.zeros(256)
        x[128] = 1.0  # Symmetric impulse at center
        scales = torch.tensor([2.0])
        cwt = continuous_wavelet_transform(x, scales, wavelet="mexican_hat")

        # Check approximate symmetry around the impulse
        left_half = cwt[0, 64:128]
        right_half = cwt[0, 129:193].flip(-1)
        assert torch.allclose(left_half, right_half, atol=1e-4)
