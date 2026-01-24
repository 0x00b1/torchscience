"""Tests for sosfiltfilt zero-phase digital filtering using SOS."""

import numpy as np
import scipy.signal
import torch

from torchscience.filter_design import sosfiltfilt


class TestSosfiltfiltMatchesScipy:
    """Test sosfiltfilt matches scipy.signal.sosfiltfilt."""

    def test_sosfiltfilt_butterworth(self) -> None:
        """Test sosfiltfilt with a Butterworth filter."""
        sos_np = scipy.signal.butter(5, 0.25, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(200, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sosfiltfilt_chebyshev(self) -> None:
        """Test sosfiltfilt with a Chebyshev type I filter."""
        sos_np = scipy.signal.cheby1(4, 0.5, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(150, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sosfiltfilt_elliptic(self) -> None:
        """Test sosfiltfilt with an elliptic filter."""
        sos_np = scipy.signal.ellip(3, 0.5, 40, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sosfiltfilt_bessel(self) -> None:
        """Test sosfiltfilt with a Bessel filter."""
        sos_np = scipy.signal.bessel(4, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(120, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestSosfiltfiltZeroPhase:
    """Test sosfiltfilt produces zero phase distortion."""

    def test_zero_phase_sinusoid(self) -> None:
        """Test that a sinusoidal signal has zero phase shift."""
        # Create a sinusoidal signal
        t = torch.linspace(0, 1, 1000, dtype=torch.float64)
        freq = 10.0  # Hz
        x = torch.sin(2 * torch.pi * freq * t)

        # Design a lowpass filter that passes 10 Hz
        sos_np = scipy.signal.butter(4, 0.05, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        # Apply zero-phase filter
        y = sosfiltfilt(sos, x)

        # Cross-correlation should peak at zero lag
        middle_region = slice(100, 900)
        x_mid = x[middle_region]
        y_mid = y[middle_region]

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


class TestSosfiltfiltMagnitudeSquared:
    """Test sosfiltfilt squares the magnitude response."""

    def test_magnitude_squared_dc_gain(self) -> None:
        """Test that DC gain is squared."""
        # Design a filter with specific DC gain
        sos_np = scipy.signal.butter(4, 0.25, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        # DC gain of original filter (product of section DC gains)
        dc_gain_single = 1.0
        for section in sos_np:
            b_sum = section[:3].sum()
            a_sum = section[3:].sum()
            dc_gain_single *= b_sum / a_sum

        # Apply sosfiltfilt to constant signal
        x = torch.ones(100, dtype=torch.float64)
        y = sosfiltfilt(sos, x)

        # Output should be dc_gain^2 (approximately, due to edge effects)
        expected_dc_gain = dc_gain_single**2
        actual_dc_gain = y[40:60].mean().item()

        np.testing.assert_allclose(actual_dc_gain, expected_dc_gain, rtol=1e-6)


class TestSosfiltfiltPadtype:
    """Test sosfiltfilt padding options."""

    def test_padtype_odd(self) -> None:
        """Test sosfiltfilt with odd padding (default)."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padtype="odd")
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy(), padtype="odd")

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_even(self) -> None:
        """Test sosfiltfilt with even padding."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padtype="even")
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy(), padtype="even")

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_constant(self) -> None:
        """Test sosfiltfilt with constant padding."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padtype="constant")
        y_scipy = scipy.signal.sosfiltfilt(
            sos_np, x.numpy(), padtype="constant"
        )

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padtype_none(self) -> None:
        """Test sosfiltfilt with no padding."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padtype=None)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy(), padtype=None)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestSosfiltfiltPadlen:
    """Test sosfiltfilt padlen parameter."""

    def test_custom_padlen(self) -> None:
        """Test sosfiltfilt with custom padding length."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padlen=20)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy(), padlen=20)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_padlen_zero(self) -> None:
        """Test sosfiltfilt with padlen=0 (no padding)."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, padlen=0)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy(), padlen=0)

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestSosfiltfiltAxis:
    """Test sosfiltfilt axis parameter."""

    def test_axis_minus_1(self) -> None:
        """Test filtering along last axis (default)."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(10, 100, dtype=torch.float64)

        y = sosfiltfilt(sos, x, axis=-1)

        # Check each row
        for i in range(10):
            y_scipy = scipy.signal.sosfiltfilt(sos_np, x[i].numpy())
            torch.testing.assert_close(
                y[i],
                torch.from_numpy(np.ascontiguousarray(y_scipy)),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_axis_0(self) -> None:
        """Test filtering along axis 0."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, 10, dtype=torch.float64)

        y = sosfiltfilt(sos, x, axis=0)

        # Check each column
        for j in range(10):
            y_scipy = scipy.signal.sosfiltfilt(sos_np, x[:, j].numpy(), axis=0)
            torch.testing.assert_close(
                y[:, j],
                torch.from_numpy(np.ascontiguousarray(y_scipy)),
                rtol=1e-10,
                atol=1e-10,
            )


class TestSosfiltfiltGradients:
    """Test sosfiltfilt gradient computation."""

    def test_gradcheck_signal(self) -> None:
        """Gradient check for input signal."""
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(50, dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda x_: sosfiltfilt(sos, x_, padtype=None),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestSosfiltfiltDtypeAndDevice:
    """Test sosfiltfilt dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float32)
        x = torch.randn(100, dtype=torch.float32)

        y = sosfiltfilt(sos, x)

        assert y.dtype == torch.float32
        assert y.shape == x.shape

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x)

        assert y.dtype == torch.float64
        assert y.shape == x.shape

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64, device="cpu")
        x = torch.randn(100, dtype=torch.float64, device="cpu")

        y = sosfiltfilt(sos, x)

        assert y.device == x.device


class TestSosfiltfiltEdgeCases:
    """Test sosfiltfilt edge cases."""

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape."""
        sos_np = scipy.signal.butter(3, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)

        for shape in [(100,), (10, 100), (5, 10, 100)]:
            x = torch.randn(*shape, dtype=torch.float64)
            y = sosfiltfilt(sos, x)
            assert y.shape == x.shape

    def test_long_signal(self) -> None:
        """Test with a long signal."""
        sos_np = scipy.signal.butter(4, 0.2, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(10000, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_single_section(self) -> None:
        """Test with a single SOS section (second-order filter)."""
        sos_np = scipy.signal.butter(2, 0.3, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(100, dtype=torch.float64)

        y = sosfiltfilt(sos, x)
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())

        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-10,
            atol=1e-10,
        )


class TestSosfiltfiltNumericalStability:
    """Test sosfiltfilt numerical stability vs filtfilt."""

    def test_high_order_filter_stability(self) -> None:
        """Test that SOS is more stable than ba for high-order filters."""
        # High order filter - SOS should be more stable
        order = 12
        sos_np = scipy.signal.butter(order, 0.1, output="sos")
        sos = torch.tensor(sos_np, dtype=torch.float64)
        x = torch.randn(500, dtype=torch.float64)

        # Should not raise or produce NaN/Inf
        y = sosfiltfilt(sos, x)

        assert torch.isfinite(y).all()

        # Compare with scipy
        y_scipy = scipy.signal.sosfiltfilt(sos_np, x.numpy())
        torch.testing.assert_close(
            y,
            torch.from_numpy(np.ascontiguousarray(y_scipy)),
            rtol=1e-8,
            atol=1e-8,
        )
