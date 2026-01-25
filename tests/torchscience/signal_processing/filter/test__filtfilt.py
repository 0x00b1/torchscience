"""Tests for filtfilt zero-phase digital filtering."""

import numpy as np
import scipy.signal
import torch

from torchscience.signal_processing.filter import filtfilt


class TestFiltfiltMatchesScipy:
    """Test filtfilt matches scipy.signal.filtfilt."""

    def test_filtfilt_butterworth(self) -> None:
        """Test filtfilt with a Butterworth filter."""
        b_np, a_np = scipy.signal.butter(5, 0.25)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = filtfilt(b, a, x)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_filtfilt_chebyshev(self) -> None:
        """Test filtfilt with a Chebyshev type I filter."""
        b_np, a_np = scipy.signal.cheby1(4, 0.5, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = filtfilt(b, a, x)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_filtfilt_elliptic(self) -> None:
        """Test filtfilt with an elliptic filter."""
        b_np, a_np = scipy.signal.ellip(3, 0.5, 40, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_filtfilt_fir_only(self) -> None:
        """Test filtfilt with FIR filter (a = [1])."""
        b = torch.tensor([0.2, 0.3, 0.3, 0.2], dtype=torch.float64)
        a = torch.tensor([1.0], dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x)
        y_scipy = scipy.signal.filtfilt(b.numpy(), a.numpy(), x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestFiltfiltZeroPhase:
    """Test filtfilt produces zero phase distortion."""

    def test_zero_phase_sinusoid(self) -> None:
        """Test that a sinusoidal signal has zero phase shift."""
        # Create a sinusoidal signal
        t = torch.linspace(0, 1, 1000, dtype=torch.float64)
        freq = 10.0  # Hz
        x = torch.sin(2 * torch.pi * freq * t)

        # Design a lowpass filter that passes 10 Hz
        b_np, a_np = scipy.signal.butter(4, 0.05)  # Cutoff at Nyquist/20
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        # Apply zero-phase filter
        y = filtfilt(b, a, x)

        # Find peaks of input and output (should be at same locations)
        # Use a simple peak detection: compare with neighbors
        middle_region = slice(100, 900)  # Avoid edge effects
        x_mid = x[middle_region]
        y_mid = y[middle_region]

        # The filtered signal should have peaks at the same locations
        # This verifies zero phase distortion
        # For a sinusoid, the peaks of x and y should be aligned

        # Cross-correlation should peak at zero lag
        x_centered = x_mid - x_mid.mean()
        y_centered = y_mid - y_mid.mean()

        # Compute correlation at lag 0
        corr_0 = (x_centered * y_centered).sum()
        # Compute correlation at lag 1 and -1
        corr_1 = (x_centered[:-1] * y_centered[1:]).sum()
        corr_m1 = (x_centered[1:] * y_centered[:-1]).sum()

        # Maximum should be at lag 0 (zero phase)
        assert corr_0 > corr_1
        assert corr_0 > corr_m1


class TestFiltfiltMagnitudeSquared:
    """Test filtfilt squares the magnitude response."""

    def test_magnitude_squared_dc_gain(self) -> None:
        """Test that DC gain is squared."""
        # Design a filter with specific DC gain
        b_np, a_np = scipy.signal.butter(4, 0.25)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        # DC gain of original filter
        dc_gain_single = np.sum(b_np) / np.sum(a_np)

        # Apply filtfilt to constant signal
        x = torch.ones(100, dtype=torch.float64)
        y = filtfilt(b, a, x)

        # Output should be dc_gain^2 (approximately, due to edge effects)
        # Check middle of signal to avoid edge effects
        expected_dc_gain = dc_gain_single**2
        actual_dc_gain = y[40:60].mean().item()

        np.testing.assert_allclose(actual_dc_gain, expected_dc_gain, rtol=1e-6)


class TestFiltfiltPadtype:
    """Test filtfilt padding options."""

    def test_padtype_odd(self) -> None:
        """Test filtfilt with odd padding (default)."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padtype="odd")
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy(), padtype="odd")

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_even(self) -> None:
        """Test filtfilt with even padding."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padtype="even")
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy(), padtype="even")

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_constant(self) -> None:
        """Test filtfilt with constant padding."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padtype="constant")
        y_scipy = scipy.signal.filtfilt(
            b_np, a_np, x.numpy(), padtype="constant"
        )

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_none(self) -> None:
        """Test filtfilt with no padding."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padtype=None)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy(), padtype=None)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestFiltfiltPadlen:
    """Test filtfilt padlen parameter."""

    def test_custom_padlen(self) -> None:
        """Test filtfilt with custom padding length."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padlen=20)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy(), padlen=20)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padlen_zero(self) -> None:
        """Test filtfilt with padlen=0 (no padding)."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x, padlen=0)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy(), padlen=0)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestFiltfiltAxis:
    """Test filtfilt axis parameter."""

    def test_axis_minus_1(self) -> None:
        """Test filtering along last axis (default)."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y = filtfilt(b, a, x, axis=-1)

        # Check each row
        for i in range(10):
            y_scipy = scipy.signal.filtfilt(b_np, a_np, x[i].numpy())
            torch.testing.assert_close(
                y[i],
                torch.from_numpy(np.ascontiguousarray(y_scipy)),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_axis_0(self) -> None:
        """Test filtering along axis 0."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, 10, dtype=torch.float64)

        y = filtfilt(b, a, x, axis=0)

        # Check each column
        for j in range(10):
            y_scipy = scipy.signal.filtfilt(
                b_np, a_np, x[:, j].numpy(), axis=0
            )
            torch.testing.assert_close(
                y[:, j],
                torch.from_numpy(np.ascontiguousarray(y_scipy)),
                rtol=1e-10,
                atol=1e-10,
            )


class TestFiltfiltGradients:
    """Test filtfilt gradient computation."""

    def test_gradcheck_signal(self) -> None:
        """Gradient check for input signal."""
        b = torch.tensor([0.5, 0.5], dtype=torch.float64)
        a = torch.tensor([1.0, -0.3], dtype=torch.float64)
        x = torch.randn(50, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda x_: filtfilt(b, a, x_, padtype=None),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestFiltfiltDtypeAndDevice:
    """Test filtfilt dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float32)
        a = torch.tensor(a_np, dtype=torch.float32)
        x = torch.randn(100, dtype=torch.float32)

        y = filtfilt(b, a, x)

        assert y.dtype == torch.float32
        assert y.shape == x.shape

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = filtfilt(b, a, x)

        assert y.dtype == torch.float64
        assert y.shape == x.shape

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64, device="cpu")
        a = torch.tensor(a_np, dtype=torch.float64, device="cpu")
        x = torch.randn(100, dtype=torch.float64, device="cpu")

        y = filtfilt(b, a, x)

        assert y.device == x.device


class TestFiltfiltEdgeCases:
    """Test filtfilt edge cases."""

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape."""
        b_np, a_np = scipy.signal.butter(3, 0.3)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)

        for shape in [(100,), (10, 100), (5, 10, 100)]:
            x = torch.randn(*shape, dtype=torch.float64)
            y = filtfilt(b, a, x)
            assert y.shape == x.shape

    def test_long_signal(self) -> None:
        """Test with a long signal."""
        b_np, a_np = scipy.signal.butter(4, 0.2)
        b = torch.tensor(b_np, dtype=torch.float64)
        a = torch.tensor(a_np, dtype=torch.float64)
        x = torch.randn(10000, dtype=torch.float64)

        y = filtfilt(b, a, x)
        y_scipy = scipy.signal.filtfilt(b_np, a_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )
