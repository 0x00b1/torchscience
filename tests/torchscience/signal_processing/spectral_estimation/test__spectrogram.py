import pytest
import torch
import torch.testing

from torchscience.signal_processing.spectral_estimation import spectrogram


class TestSpectrogramForward:
    """Tests for spectrogram forward computation."""

    def test_output_shapes(self):
        """Test output shapes with default parameters."""
        x = torch.randn(2048, dtype=torch.float64)
        nperseg = 256
        noverlap = 128
        freqs, times, Sxx = spectrogram(x, nperseg=nperseg, noverlap=noverlap)

        n_freqs = nperseg // 2 + 1
        step = nperseg - noverlap
        n_segments = (x.size(-1) - nperseg) // step + 1

        assert freqs.shape == (n_freqs,)
        assert times.shape == (n_segments,)
        assert Sxx.shape == (n_freqs, n_segments)

    def test_batch_shapes(self):
        """Test batch dimension preservation."""
        x = torch.randn(3, 2, 2048, dtype=torch.float64)
        freqs, times, Sxx = spectrogram(x, nperseg=128)
        assert Sxx.shape[:2] == (3, 2)
        assert Sxx.shape[2] == 65  # 128//2 + 1

    def test_chirp_time_frequency(self):
        """Test that a chirp shows increasing frequency over time."""
        fs = 1000.0
        n = 10000
        t = torch.arange(n, dtype=torch.float64) / fs
        # Linear chirp from 10 Hz to 100 Hz
        x = torch.sin(2 * torch.pi * (10 + 45 * t) * t)

        freqs, times, Sxx = spectrogram(x, fs=fs, nperseg=256)

        # Find peak frequency at early vs late times
        early_peak = freqs[Sxx[:, 0].argmax()]
        late_peak = freqs[Sxx[:, -1].argmax()]
        assert late_peak > early_peak

    def test_non_negative_output(self):
        """Test spectrogram values are non-negative."""
        x = torch.randn(1024, dtype=torch.float64)
        _, _, Sxx = spectrogram(x, nperseg=128)
        assert (Sxx >= -1e-10).all()

    def test_time_bins_monotonic(self):
        """Test that time bins are monotonically increasing."""
        x = torch.randn(2048, dtype=torch.float64)
        _, times, _ = spectrogram(x, fs=100.0, nperseg=256)
        assert (times[1:] > times[:-1]).all()

    def test_mean_approximates_welch(self):
        """Test that averaging spectrogram over time approximates Welch."""
        from torchscience.signal_processing.spectral_estimation import welch

        torch.manual_seed(42)
        x = torch.randn(4096, dtype=torch.float64)
        nperseg = 256
        noverlap = 128

        _, _, Sxx = spectrogram(x, nperseg=nperseg, noverlap=noverlap)
        sxx_mean = Sxx.mean(dim=-1)

        _, psd_welch = welch(x, nperseg=nperseg, noverlap=noverlap)

        torch.testing.assert_close(sxx_mean, psd_welch, atol=1e-10, rtol=1e-10)


class TestSpectrogramGradient:
    """Tests for gradient support."""

    def test_gradient_flows(self):
        """Test that gradients flow through spectrogram."""
        x = torch.randn(512, dtype=torch.float64, requires_grad=True)
        _, _, Sxx = spectrogram(x, nperseg=64)
        loss = Sxx.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self):
        """Test gradient correctness using finite differences."""
        x = torch.randn(256, dtype=torch.float64, requires_grad=True)

        def fn(x):
            _, _, Sxx = spectrogram(x, nperseg=64)
            return Sxx

        torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4)


class TestSpectrogramMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(2048, device="meta")
        window = torch.empty(256, device="meta")
        freqs, times, Sxx = spectrogram(x, window=window, nperseg=256)
        assert Sxx.shape[0] == 129  # 256//2 + 1


class TestSpectrogramDtype:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype):
        """Test Sxx output dtype matches input."""
        x = torch.randn(1024, dtype=dtype)
        _, _, Sxx = spectrogram(x, nperseg=128)
        assert Sxx.dtype == dtype


class TestSpectrogramErrors:
    """Tests for error handling."""

    def test_integer_input_raises(self):
        """Test that integer input raises."""
        x = torch.tensor([1, 2, 3, 4, 5])
        # Error is raised either from hann_window or C++ check
        with pytest.raises(RuntimeError):
            spectrogram(x, nperseg=4)

    def test_signal_too_short_raises(self):
        """Test that a too-short signal raises."""
        x = torch.randn(10)
        with pytest.raises(RuntimeError, match="too short"):
            spectrogram(x, nperseg=256)
