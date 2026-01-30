import pytest
import torch
import torch.testing

from torchscience.signal_processing.spectral_estimation import (
    cross_spectral_density,
    welch,
)


class TestCSDForward:
    """Tests for cross spectral density forward computation."""

    def test_auto_spectrum_matches_welch(self):
        """CSD of x with itself should equal Welch PSD."""
        torch.manual_seed(42)
        x = torch.randn(2048, dtype=torch.float64)
        nperseg = 128

        freqs_w, psd = welch(x, nperseg=nperseg)
        freqs_c, csd = cross_spectral_density(x, x, nperseg=nperseg)

        torch.testing.assert_close(freqs_w, freqs_c)
        # CSD(x, x) should be real and equal to PSD(x)
        torch.testing.assert_close(csd.real, psd, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            csd.imag,
            torch.zeros_like(csd.imag),
            atol=1e-10,
            rtol=0.0,
        )

    def test_output_is_complex(self):
        """CSD output should be complex."""
        x = torch.randn(1024, dtype=torch.float64)
        y = torch.randn(1024, dtype=torch.float64)
        _, csd = cross_spectral_density(x, y, nperseg=128)
        assert csd.is_complex()

    def test_output_shape(self):
        """Test output shape."""
        x = torch.randn(1024, dtype=torch.float64)
        y = torch.randn(1024, dtype=torch.float64)
        freqs, csd = cross_spectral_density(x, y, nperseg=128)
        assert freqs.shape == (65,)
        assert csd.shape == (65,)

    def test_batch_shape(self):
        """Test batch dimensions."""
        x = torch.randn(3, 1024, dtype=torch.float64)
        y = torch.randn(3, 1024, dtype=torch.float64)
        freqs, csd = cross_spectral_density(x, y, nperseg=128)
        assert csd.shape == (3, 65)

    def test_correlated_signals_peak(self):
        """Test that correlated signals show a peak at shared frequency."""
        fs = 1000.0
        n = 10000
        t = torch.arange(n, dtype=torch.float64) / fs
        freq = 50.0
        x = torch.sin(2 * torch.pi * freq * t)
        y = torch.sin(2 * torch.pi * freq * t + 0.3)

        freqs, csd = cross_spectral_density(x, y, fs=fs, nperseg=256)
        # Peak magnitude should be at shared frequency
        peak_freq = freqs[csd.abs().argmax()]
        torch.testing.assert_close(
            peak_freq,
            torch.tensor(freq, dtype=torch.float64),
            atol=4.0,
            rtol=0.0,
        )

    def test_hermitian_symmetry(self):
        """CSD(x,y) should equal conj(CSD(y,x))."""
        torch.manual_seed(42)
        x = torch.randn(1024, dtype=torch.float64)
        y = torch.randn(1024, dtype=torch.float64)
        _, csd_xy = cross_spectral_density(x, y, nperseg=128)
        _, csd_yx = cross_spectral_density(y, x, nperseg=128)
        torch.testing.assert_close(
            csd_xy, csd_yx.conj(), atol=1e-10, rtol=1e-10
        )


class TestCSDGradient:
    """Tests for gradient support."""

    def test_gradient_flows_x(self):
        """Test gradients flow through x."""
        x = torch.randn(512, dtype=torch.float64, requires_grad=True)
        y = torch.randn(512, dtype=torch.float64)
        _, csd = cross_spectral_density(x, y, nperseg=64)
        loss = csd.abs().sum()
        loss.backward()
        assert x.grad is not None

    def test_gradient_flows_y(self):
        """Test gradients flow through y."""
        x = torch.randn(512, dtype=torch.float64)
        y = torch.randn(512, dtype=torch.float64, requires_grad=True)
        _, csd = cross_spectral_density(x, y, nperseg=64)
        loss = csd.abs().sum()
        loss.backward()
        assert y.grad is not None


class TestCSDMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.empty(1024, device="meta")
        y = torch.empty(1024, device="meta")
        window = torch.empty(128, device="meta")
        freqs, csd = cross_spectral_density(x, y, window=window, nperseg=128)
        assert csd.shape == (65,)


class TestCSDErrors:
    """Tests for error handling."""

    def test_integer_input_raises(self):
        """Test that integer input raises."""
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([1, 2, 3, 4, 5])
        # Error is raised either from hann_window or C++ check
        with pytest.raises(RuntimeError):
            cross_spectral_density(x, y, nperseg=4)
