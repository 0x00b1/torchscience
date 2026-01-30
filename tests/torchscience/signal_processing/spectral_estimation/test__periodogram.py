import pytest
import torch
import torch.testing

from torchscience.signal_processing.spectral_estimation import periodogram


class TestPeriodogramForward:
    """Tests for periodogram forward computation."""

    def test_known_sine_peak(self):
        """Test that a pure sine wave produces a peak at the correct frequency."""
        fs = 1000.0
        n = 1000
        t = torch.arange(n, dtype=torch.float64) / fs
        freq = 50.0
        x = torch.sin(2 * torch.pi * freq * t)

        freqs, psd = periodogram(x, fs=fs)

        # Peak should be at 50 Hz
        peak_freq = freqs[psd.argmax()]
        torch.testing.assert_close(
            peak_freq,
            torch.tensor(freq, dtype=torch.float64),
            atol=1.0,
            rtol=0.0,
        )

    def test_dc_signal(self):
        """Test that a constant signal has all power at DC."""
        x = torch.ones(100, dtype=torch.float64) * 3.0
        freqs, psd = periodogram(x)

        # DC component should dominate
        assert psd[0] > psd[1:].max() * 10

    def test_output_shapes(self):
        """Test output shape: N samples -> N//2+1 frequency bins."""
        for n in [100, 101, 256]:
            x = torch.randn(n, dtype=torch.float64)
            freqs, psd = periodogram(x)
            expected_len = n // 2 + 1
            assert freqs.shape == (expected_len,), (
                f"n={n}: freqs shape {freqs.shape}"
            )
            assert psd.shape == (expected_len,), (
                f"n={n}: psd shape {psd.shape}"
            )

    def test_batch_shape(self):
        """Test that batch dimensions are preserved."""
        x = torch.randn(3, 5, 100, dtype=torch.float64)
        freqs, psd = periodogram(x)
        assert psd.shape == (3, 5, 51)

    def test_density_vs_spectrum_scaling(self):
        """Test that density and spectrum scaling differ by expected factor."""
        x = torch.randn(256, dtype=torch.float64)
        _, psd_density = periodogram(x, fs=100.0, scaling="density")
        _, psd_spectrum = periodogram(x, fs=100.0, scaling="spectrum")
        # They should not be equal
        assert not torch.allclose(psd_density, psd_spectrum)

    def test_parseval_spectrum_scaling(self):
        """Test Parseval's theorem: sum of spectrum approx equals mean of x^2."""
        torch.manual_seed(42)
        x = torch.randn(1024, dtype=torch.float64)
        _, psd = periodogram(x, scaling="spectrum")
        signal_power = (x**2).sum()
        spectral_power = psd.sum()
        torch.testing.assert_close(
            spectral_power, signal_power / x.size(-1), atol=1e-6, rtol=1e-4
        )

    def test_custom_window(self):
        """Test with a Hann window."""
        n = 256
        x = torch.randn(n, dtype=torch.float64)
        window = torch.hann_window(n, dtype=torch.float64)
        freqs, psd = periodogram(x, window=window)
        assert psd.shape == (n // 2 + 1,)
        # PSD should be non-negative
        assert (psd >= -1e-10).all()


class TestPeriodogramGradient:
    """Tests for gradient support."""

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)

        def fn(x):
            _, psd = periodogram(x)
            return psd

        torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)

    def test_gradient_flows(self):
        """Test that gradients flow through periodogram."""
        x = torch.randn(128, dtype=torch.float64, requires_grad=True)
        _, psd = periodogram(x)
        loss = psd.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPeriodogramMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(256, device="meta")
        window = torch.empty(256, device="meta")
        freqs, psd = periodogram(x, window=window)
        assert freqs.shape == (129,)
        assert psd.shape == (129,)
        assert freqs.device.type == "meta"
        assert psd.device.type == "meta"


class TestPeriodogramDtype:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype):
        """Test that PSD output dtype matches input."""
        x = torch.randn(100, dtype=dtype)
        _, psd = periodogram(x)
        assert psd.dtype == dtype


class TestPeriodogramErrors:
    """Tests for error handling."""

    def test_integer_input_raises(self):
        """Test that integer input raises an error."""
        x = torch.tensor([1, 2, 3])
        with pytest.raises(
            RuntimeError, match="periodogram requires real floating"
        ):
            periodogram(x)

    def test_wrong_window_length_raises(self):
        """Test that mismatched window length raises an error."""
        x = torch.randn(100)
        window = torch.ones(50)
        with pytest.raises(RuntimeError, match="window must be 1-D"):
            periodogram(x, window=window)
