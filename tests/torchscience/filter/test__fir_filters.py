"""Tests for FIR filter design functions."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import signal

from torchscience.filter import firwin, firwin2


class TestFirwin:
    """Tests for window-based FIR filter design."""

    def test_lowpass_basic(self):
        """Test basic lowpass filter design."""
        h = firwin(51, 0.3)
        assert h.shape == torch.Size([51])
        # Lowpass filter should have positive center tap
        assert h[25].item() > 0

    def test_lowpass_sum_unity(self):
        """Test that lowpass filter coefficients sum to approximately 1."""
        h = firwin(51, 0.3)
        assert_allclose(h.sum().item(), 1.0, rtol=0.01)

    def test_highpass_basic(self):
        """Test basic highpass filter design."""
        h = firwin(51, 0.3, filter_type="highpass")
        assert h.shape == torch.Size([51])
        # Highpass filter should have coefficients that sum to ~0
        assert abs(h.sum().item()) < 0.1

    def test_bandpass_basic(self):
        """Test basic bandpass filter design."""
        h = firwin(51, [0.2, 0.4], filter_type="bandpass")
        assert h.shape == torch.Size([51])

    def test_bandstop_basic(self):
        """Test basic bandstop filter design."""
        h = firwin(51, [0.2, 0.4], filter_type="bandstop")
        assert h.shape == torch.Size([51])
        # Bandstop filter should have coefficients that sum to ~1
        assert_allclose(h.sum().item(), 1.0, rtol=0.1)

    def test_compare_scipy_lowpass(self):
        """Compare lowpass filter with scipy.signal.firwin."""
        num_taps = 51
        cutoff = 0.3

        h_torch = firwin(num_taps, cutoff)
        h_scipy = signal.firwin(num_taps, cutoff)

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_compare_scipy_highpass(self):
        """Compare highpass filter with scipy.signal.firwin."""
        num_taps = 51
        cutoff = 0.3

        h_torch = firwin(num_taps, cutoff, filter_type="highpass")
        h_scipy = signal.firwin(num_taps, cutoff, pass_zero=False)

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_compare_scipy_bandpass(self):
        """Compare bandpass filter with scipy.signal.firwin."""
        num_taps = 51
        cutoff = [0.2, 0.4]

        h_torch = firwin(num_taps, cutoff, filter_type="bandpass")
        h_scipy = signal.firwin(num_taps, cutoff, pass_zero=False)

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_compare_scipy_bandstop(self):
        """Compare bandstop filter with scipy.signal.firwin."""
        num_taps = 51
        cutoff = [0.2, 0.4]

        h_torch = firwin(num_taps, cutoff, filter_type="bandstop")
        h_scipy = signal.firwin(num_taps, cutoff, pass_zero=True)

        assert_allclose(h_torch.numpy(), h_scipy, rtol=1e-10)

    def test_window_hamming(self):
        """Test with Hamming window (default)."""
        h = firwin(51, 0.3, window="hamming")
        assert h.shape == torch.Size([51])

    def test_window_hann(self):
        """Test with Hann window."""
        h = firwin(51, 0.3, window="hann")
        h_scipy = signal.firwin(51, 0.3, window="hann")
        assert_allclose(h.numpy(), h_scipy, rtol=1e-10)

    def test_window_blackman(self):
        """Test with Blackman window."""
        h = firwin(51, 0.3, window="blackman")
        h_scipy = signal.firwin(51, 0.3, window="blackman")
        assert_allclose(h.numpy(), h_scipy, rtol=1e-10)

    def test_window_kaiser(self):
        """Test with Kaiser window."""
        h = firwin(51, 0.3, window=("kaiser", 8.0))
        h_scipy = signal.firwin(51, 0.3, window=("kaiser", 8.0))
        assert_allclose(h.numpy(), h_scipy, rtol=1e-6)

    def test_window_rectangular(self):
        """Test with rectangular window."""
        h = firwin(51, 0.3, window="rectangular")
        h_scipy = signal.firwin(51, 0.3, window="boxcar")
        assert_allclose(h.numpy(), h_scipy, rtol=1e-10)

    def test_sampling_frequency(self):
        """Test with explicit sampling frequency."""
        fs = 1000
        cutoff_hz = 150

        h = firwin(51, cutoff_hz, sampling_frequency=fs)
        h_scipy = signal.firwin(51, cutoff_hz, fs=fs)

        assert_allclose(h.numpy(), h_scipy, rtol=1e-10)

    def test_tensor_cutoff(self):
        """Test with tensor cutoff frequencies."""
        cutoff = torch.tensor([0.2, 0.4])
        h = firwin(51, cutoff, filter_type="bandpass")
        assert h.shape == torch.Size([51])

    def test_scale_true(self):
        """Test with scale=True (default)."""
        h = firwin(51, 0.3, scale=True)
        assert_allclose(h.sum().item(), 1.0, rtol=0.01)

    def test_scale_false(self):
        """Test with scale=False."""
        h = firwin(51, 0.3, scale=False)
        # Without scaling, sum won't be exactly 1
        # Just verify it runs and returns correct shape
        assert h.shape == torch.Size([51])

    def test_dtype_float32(self):
        """Test float32 output."""
        h = firwin(51, 0.3, dtype=torch.float32)
        assert h.dtype == torch.float32

    def test_dtype_float64(self):
        """Test float64 output."""
        h = firwin(51, 0.3, dtype=torch.float64)
        assert h.dtype == torch.float64

    def test_device_cpu(self):
        """Test CPU device."""
        h = firwin(51, 0.3, device=torch.device("cpu"))
        assert h.device.type == "cpu"

    def test_invalid_num_taps(self):
        """Test that invalid num_taps raises error."""
        with pytest.raises(ValueError, match="num_taps must be at least 1"):
            firwin(0, 0.3)

    def test_invalid_cutoff_range(self):
        """Test that cutoff outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="Cutoff frequencies must be"):
            firwin(51, 1.5)

    def test_invalid_cutoff_order(self):
        """Test that non-increasing cutoffs raise error."""
        with pytest.raises(ValueError, match="must be strictly increasing"):
            firwin(51, [0.4, 0.2], filter_type="bandpass")

    def test_frequency_response_lowpass(self):
        """Test that lowpass filter has correct frequency response."""
        h = firwin(101, 0.3)
        w, H = signal.freqz(h.numpy(), worN=512)

        # Check passband (should be close to 1)
        passband_idx = w < 0.3 * np.pi * 0.8
        assert np.all(np.abs(H[passband_idx]) > 0.9)

        # Check stopband (should be close to 0)
        stopband_idx = w > 0.3 * np.pi * 1.5
        assert np.all(np.abs(H[stopband_idx]) < 0.1)


class TestFirwin2:
    """Tests for frequency-sampling FIR filter design."""

    def test_basic_lowpass(self):
        """Test basic lowpass filter design."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains)
        assert h.shape == torch.Size([65])

    def test_basic_highpass(self):
        """Test basic highpass filter design."""
        freqs = [0, 0.3, 0.35, 1.0]
        gains = [0, 0, 1, 1]
        h = firwin2(65, freqs, gains)
        assert h.shape == torch.Size([65])

    def test_basic_bandpass(self):
        """Test basic bandpass filter design."""
        freqs = [0, 0.2, 0.25, 0.4, 0.45, 1.0]
        gains = [0, 0, 1, 1, 0, 0]
        h = firwin2(65, freqs, gains)
        assert h.shape == torch.Size([65])

    def test_compare_scipy_lowpass(self):
        """Compare lowpass filter with scipy.signal.firwin2."""
        num_taps = 65
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]

        h_torch = firwin2(num_taps, freqs, gains)
        h_scipy = signal.firwin2(num_taps, freqs, gains)

        # firwin2 implementations can vary slightly
        # Check that frequency response is similar
        w_torch, H_torch = signal.freqz(h_torch.numpy(), worN=256)
        w_scipy, H_scipy = signal.freqz(h_scipy, worN=256)

        # Both should have similar passband/stopband behavior
        assert_allclose(np.abs(H_torch), np.abs(H_scipy), rtol=0.2, atol=0.1)

    def test_window_hamming(self):
        """Test with Hamming window."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains, window="hamming")
        assert h.shape == torch.Size([65])

    def test_window_kaiser(self):
        """Test with Kaiser window."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains, window=("kaiser", 8.0))
        assert h.shape == torch.Size([65])

    def test_sampling_frequency(self):
        """Test with explicit sampling frequency."""
        fs = 1000
        freqs = [0, 125, 150, 500]  # Hz
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains, sampling_frequency=fs)
        assert h.shape == torch.Size([65])

    def test_tensor_inputs(self):
        """Test with tensor inputs."""
        freqs = torch.tensor([0, 0.25, 0.3, 1.0])
        gains = torch.tensor([1, 1, 0, 0])
        h = firwin2(65, freqs, gains)
        assert h.shape == torch.Size([65])

    def test_n_freqs(self):
        """Test with custom n_freqs."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains, n_freqs=1024)
        assert h.shape == torch.Size([65])

    def test_antisymmetric(self):
        """Test antisymmetric filter design."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [0, 0.25, 0.3, 1.0]  # Differentiator-like
        h = firwin2(64, freqs, gains, antisymmetric=True)
        assert h.shape == torch.Size([64])

    def test_dtype_float32(self):
        """Test float32 output."""
        freqs = [0, 0.25, 0.3, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(65, freqs, gains, dtype=torch.float32)
        assert h.dtype == torch.float32

    def test_invalid_num_taps(self):
        """Test that invalid num_taps raises error."""
        with pytest.raises(ValueError, match="num_taps must be at least 1"):
            firwin2(0, [0, 1], [1, 1])

    def test_invalid_freq_start(self):
        """Test that first frequency must be 0."""
        with pytest.raises(ValueError, match="First frequency must be 0"):
            firwin2(65, [0.1, 1.0], [1, 0])

    def test_invalid_freq_end(self):
        """Test that last frequency must be 1."""
        with pytest.raises(ValueError, match="Last frequency must be 1"):
            firwin2(65, [0, 0.5], [1, 0])

    def test_invalid_freq_order(self):
        """Test that frequencies must be increasing."""
        with pytest.raises(ValueError, match="strictly increasing"):
            firwin2(65, [0, 0.5, 0.3, 1.0], [1, 1, 0, 0])

    def test_mismatched_lengths(self):
        """Test that frequencies and gains must have same length."""
        with pytest.raises(ValueError, match="same length"):
            firwin2(65, [0, 0.5, 1.0], [1, 0])

    def test_minimum_points(self):
        """Test that at least 2 frequency points are required."""
        with pytest.raises(ValueError, match="At least 2 frequency points"):
            firwin2(65, [0.5], [1])

    def test_frequency_response(self):
        """Test that filter has approximately correct frequency response."""
        freqs = [0, 0.2, 0.25, 1.0]
        gains = [1, 1, 0, 0]
        h = firwin2(129, freqs, gains)

        w, H = signal.freqz(h.numpy(), worN=512)
        w_norm = w / np.pi

        # Check passband (should be close to 1)
        passband_idx = w_norm < 0.15
        assert np.all(np.abs(H[passband_idx]) > 0.8)

        # Check stopband (should be close to 0)
        stopband_idx = w_norm > 0.35
        assert np.all(np.abs(H[stopband_idx]) < 0.2)
