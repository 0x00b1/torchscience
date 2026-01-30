import pytest
import torch
import torch.testing

from torchscience.signal_processing.spectral_estimation import welch


class TestWelchForward:
    """Tests for Welch's method forward computation."""

    def test_known_sine_peak(self):
        """Test that a pure sine wave produces a peak at the correct frequency."""
        fs = 1000.0
        n = 10000
        t = torch.arange(n, dtype=torch.float64) / fs
        freq = 50.0
        x = torch.sin(2 * torch.pi * freq * t)

        freqs, psd = welch(x, fs=fs, nperseg=256)

        peak_freq = freqs[psd.argmax()]
        # Frequency resolution is fs/nperseg ~ 3.9 Hz
        torch.testing.assert_close(
            peak_freq,
            torch.tensor(freq, dtype=torch.float64),
            atol=4.0,
            rtol=0.0,
        )

    def test_output_shapes(self):
        """Test output shape matches nperseg."""
        nperseg = 128
        x = torch.randn(1024, dtype=torch.float64)
        freqs, psd = welch(x, nperseg=nperseg)
        expected_len = nperseg // 2 + 1
        assert freqs.shape == (expected_len,)
        assert psd.shape == (expected_len,)

    def test_batch_shape(self):
        """Test batch dimensions are preserved."""
        x = torch.randn(3, 5, 1024, dtype=torch.float64)
        freqs, psd = welch(x, nperseg=128)
        assert psd.shape == (3, 5, 65)

    def test_lower_variance_than_periodogram(self):
        """Test that Welch has lower variance than a single periodogram."""
        torch.manual_seed(42)
        from torchscience.signal_processing.spectral_estimation import (
            periodogram,
        )

        n_trials = 50
        periodogram_vars = []
        welch_vars = []
        for _ in range(n_trials):
            x = torch.randn(1024, dtype=torch.float64)
            _, psd_p = periodogram(x)
            _, psd_w = welch(x, nperseg=256)
            periodogram_vars.append(psd_p.var().item())
            welch_vars.append(psd_w.var().item())
        # Welch's average variance should be lower
        assert sum(welch_vars) / n_trials < sum(periodogram_vars) / n_trials

    def test_non_negative_output(self):
        """Test PSD is non-negative."""
        x = torch.randn(1024, dtype=torch.float64)
        _, psd = welch(x, nperseg=128)
        assert (psd >= -1e-10).all()

    def test_custom_overlap(self):
        """Test with custom overlap."""
        x = torch.randn(1024, dtype=torch.float64)
        freqs, psd = welch(x, nperseg=128, noverlap=64)
        assert psd.shape == (65,)

    def test_no_overlap(self):
        """Test with zero overlap (Bartlett's method)."""
        x = torch.randn(1024, dtype=torch.float64)
        freqs, psd = welch(x, nperseg=128, noverlap=0)
        assert psd.shape == (65,)


class TestWelchGradient:
    """Tests for gradient support."""

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x = torch.randn(512, dtype=torch.float64, requires_grad=True)

        def fn(x):
            _, psd = welch(x, nperseg=64)
            return psd

        torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Test that gradients flow through welch."""
        x = torch.randn(512, dtype=torch.float64, requires_grad=True)
        _, psd = welch(x, nperseg=64)
        loss = psd.sum()
        loss.backward()
        assert x.grad is not None


class TestWelchMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(1024, device="meta")
        window = torch.empty(128, device="meta")
        freqs, psd = welch(x, window=window, nperseg=128)
        assert psd.shape == (65,)


class TestWelchDtype:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype):
        """Test PSD output dtype matches input."""
        x = torch.randn(512, dtype=dtype)
        _, psd = welch(x, nperseg=64)
        assert psd.dtype == dtype


class TestWelchErrors:
    """Tests for error handling."""

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x = torch.tensor([1, 2, 3, 4, 5])
        # Error is raised either from hann_window or C++ check
        with pytest.raises(RuntimeError):
            welch(x, nperseg=4)

    def test_signal_too_short_raises(self):
        """Test that signal shorter than nperseg raises an error."""
        x = torch.randn(10)
        with pytest.raises(RuntimeError, match="too short"):
            welch(x, nperseg=256)
