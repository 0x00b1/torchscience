"""Tests for FFT-based FIR filtering."""

import torch
from scipy.signal import fftconvolve

from torchscience.filter import fftfilt, firwin


class TestFftfilt:
    """Test fftfilt function."""

    def test_matches_scipy_fftconvolve(self) -> None:
        """Output should match scipy.signal.fftconvolve mode='same'."""
        # Create a lowpass filter
        b = firwin(51, 0.3, filter_type="lowpass", dtype=torch.float64)
        # Create a test signal
        x = torch.randn(1000, dtype=torch.float64)

        # Apply our fftfilt
        y = fftfilt(b, x)

        # Apply scipy's fftconvolve with mode='same'
        # Note: scipy's fftconvolve returns size of first argument with mode='same'
        # so we call fftconvolve(x, b) to get output same size as x
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")

        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_matches_scipy_short_filter(self) -> None:
        """Should match scipy for short filters."""
        b = firwin(11, 0.4, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = fftfilt(b, x)
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")

        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_matches_scipy_long_filter(self) -> None:
        """Should match scipy for long filters."""
        b = firwin(201, 0.2, dtype=torch.float64)
        x = torch.randn(500, dtype=torch.float64)

        y = fftfilt(b, x)
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")

        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_gradcheck_filter_coeffs(self) -> None:
        """Gradient check for filter coefficients."""
        b = torch.randn(21, dtype=torch.float64, requires_grad=True)
        x = torch.randn(100, dtype=torch.float64)

        torch.autograd.gradcheck(
            lambda b_: fftfilt(b_, x),
            (b,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_signal(self) -> None:
        """Gradient check for input signal."""
        b = torch.randn(21, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda x_: fftfilt(b, x_),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_both(self) -> None:
        """Gradient check for both filter and signal."""
        b = torch.randn(21, dtype=torch.float64, requires_grad=True)
        x = torch.randn(100, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            fftfilt,
            (b, x),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_batch_signal_2d(self) -> None:
        """Should handle batched 2D signals."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y = fftfilt(b, x, axis=-1)

        assert y.shape == x.shape

        # Each row should match individual filtering
        for i in range(10):
            y_expected = fftconvolve(x[i].numpy(), b.numpy(), mode="same")
            torch.testing.assert_close(
                y[i],
                torch.from_numpy(y_expected),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_batch_signal_3d(self) -> None:
        """Should handle batched 3D signals."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(5, 8, 100, dtype=torch.float64)

        y = fftfilt(b, x, axis=-1)

        assert y.shape == x.shape

        # Check a few samples
        for i in range(5):
            for j in range(8):
                y_expected = fftconvolve(
                    x[i, j].numpy(), b.numpy(), mode="same"
                )
                torch.testing.assert_close(
                    y[i, j],
                    torch.from_numpy(y_expected),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_axis_parameter(self) -> None:
        """Should filter along specified axis."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(100, 50, dtype=torch.float64)

        # Filter along axis 0
        y = fftfilt(b, x, axis=0)
        assert y.shape == x.shape

        # Each column should be filtered
        for j in range(50):
            y_expected = fftconvolve(x[:, j].numpy(), b.numpy(), mode="same")
            torch.testing.assert_close(
                y[:, j],
                torch.from_numpy(y_expected),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_axis_negative(self) -> None:
        """Should handle negative axis."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y_neg = fftfilt(b, x, axis=-1)
        y_pos = fftfilt(b, x, axis=1)

        torch.testing.assert_close(y_neg, y_pos)

    def test_dtype_float32(self) -> None:
        """Should handle float32 inputs."""
        b = torch.randn(21, dtype=torch.float32)
        x = torch.randn(100, dtype=torch.float32)

        y = fftfilt(b, x)

        assert y.dtype == torch.float32
        assert y.shape == x.shape

    def test_dtype_float64(self) -> None:
        """Should handle float64 inputs."""
        b = torch.randn(21, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = fftfilt(b, x)

        assert y.dtype == torch.float64
        assert y.shape == x.shape

    def test_output_shape_preserved(self) -> None:
        """Output shape should match input signal shape."""
        b = firwin(51, 0.3, dtype=torch.float64)

        for shape in [(100,), (10, 100), (5, 10, 100), (2, 3, 4, 100)]:
            x = torch.randn(*shape, dtype=torch.float64)
            y = fftfilt(b, x)
            assert y.shape == x.shape

    def test_identity_filter(self) -> None:
        """Delta function filter should return input (with slight shift)."""
        # Create a delta function filter
        b = torch.zeros(21, dtype=torch.float64)
        b[10] = 1.0  # Center tap

        x = torch.randn(100, dtype=torch.float64)
        y = fftfilt(b, x)

        # Should be close to identity (the delta is centered)
        torch.testing.assert_close(y, x, rtol=1e-10, atol=1e-10)

    def test_device_preservation_cpu(self) -> None:
        """Should preserve device (CPU test)."""
        b = torch.randn(21, dtype=torch.float64, device="cpu")
        x = torch.randn(100, dtype=torch.float64, device="cpu")

        y = fftfilt(b, x)

        assert y.device == x.device

    def test_filter_longer_than_signal(self) -> None:
        """Should handle filter longer than signal."""
        b = firwin(51, 0.3, dtype=torch.float64)
        x = torch.randn(30, dtype=torch.float64)

        y = fftfilt(b, x)
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")

        assert y.shape == x.shape
        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_1d_signal(self) -> None:
        """Should handle 1D signals correctly."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = fftfilt(b, x)

        assert y.ndim == 1
        assert y.shape == x.shape

    def test_complex_filter(self) -> None:
        """Should handle complex filter coefficients."""
        b_real = torch.randn(21, dtype=torch.float64)
        b = b_real + 1j * torch.randn(21, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = fftfilt(b, x)

        # Result should be complex
        assert y.is_complex()

        # Check against scipy
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")
        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complex_signal(self) -> None:
        """Should handle complex input signals."""
        b = firwin(21, 0.3, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64) + 1j * torch.randn(
            100, dtype=torch.float64
        )

        y = fftfilt(b, x)

        # Result should be complex
        assert y.is_complex()

        # Check against scipy
        y_scipy = fftconvolve(x.numpy(), b.numpy(), mode="same")
        torch.testing.assert_close(
            y,
            torch.from_numpy(y_scipy),
            rtol=1e-10,
            atol=1e-10,
        )
