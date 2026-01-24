"""Tests for the butterworth_design filter design function."""

import math

import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.filter_design import butterworth_design


class TestButterworthDesignLowpass:
    """Test butterworth_design lowpass filter design."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("cutoff", [0.1, 0.25, 0.5, 0.75])
    def test_matches_scipy_sos(self, order: int, cutoff: float) -> None:
        """SOS output should match scipy.signal.butter."""
        sos = butterworth_design(
            order, cutoff, filter_type="lowpass", output="sos"
        )

        # Scipy reference
        sos_scipy = scipy_signal.butter(
            order, cutoff, btype="lowpass", output="sos"
        )

        # Compare frequency response
        w = torch.linspace(0, math.pi, 100)
        h_ts = _sos_freqz(sos, w)
        h_scipy = _sos_freqz(torch.from_numpy(sos_scipy), w)

        # Use larger atol for near-zero stopband values
        torch.testing.assert_close(
            h_ts.abs(), h_scipy.abs(), rtol=1e-3, atol=1e-6
        )

    def test_cutoff_at_minus_3db(self) -> None:
        """Magnitude at cutoff should be -3dB (1/sqrt(2))."""
        order = 4
        cutoff = 0.3
        sos = butterworth_design(order, cutoff, filter_type="lowpass")

        # Evaluate at cutoff frequency
        w = torch.tensor([cutoff * math.pi])
        h = _sos_freqz(sos, w)

        # Should be approximately -3dB
        expected = 1.0 / math.sqrt(2)
        assert abs(h[0].abs() - expected) < 0.01


class TestButterworthDesignHighpass:
    """Test butterworth_design highpass filter design."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    @pytest.mark.parametrize("cutoff", [0.1, 0.25, 0.5])
    def test_matches_scipy_sos(self, order: int, cutoff: float) -> None:
        """SOS output should match scipy.signal.butter."""
        sos = butterworth_design(
            order, cutoff, filter_type="highpass", output="sos"
        )

        sos_scipy = scipy_signal.butter(
            order, cutoff, btype="highpass", output="sos"
        )

        w = torch.linspace(0.01, math.pi, 100)  # Avoid DC
        h_ts = _sos_freqz(sos, w)
        h_scipy = _sos_freqz(torch.from_numpy(sos_scipy), w)

        # Use larger atol for near-zero stopband values
        torch.testing.assert_close(
            h_ts.abs(), h_scipy.abs(), rtol=1e-3, atol=1e-6
        )


class TestButterworthDesignBandpass:
    """Test butterworth_design bandpass filter design."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_matches_scipy_sos(self, order: int) -> None:
        """SOS output should match scipy.signal.butter."""
        cutoff = [0.2, 0.5]
        sos = butterworth_design(
            order, cutoff, filter_type="bandpass", output="sos"
        )

        sos_scipy = scipy_signal.butter(
            order, cutoff, btype="bandpass", output="sos"
        )

        # Bandpass doubles order
        assert sos.shape[0] == sos_scipy.shape[0]

        w = torch.linspace(0.01, math.pi, 100)
        h_ts = _sos_freqz(sos, w)
        h_scipy = _sos_freqz(torch.from_numpy(sos_scipy), w)

        # Use larger atol for near-zero stopband values
        torch.testing.assert_close(
            h_ts.abs(), h_scipy.abs(), rtol=1e-3, atol=1e-6
        )


class TestButterworthDesignBandstop:
    """Test butterworth_design bandstop filter design."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_matches_scipy_sos(self, order: int) -> None:
        """SOS output should match scipy.signal.butter."""
        cutoff = [0.2, 0.5]
        sos = butterworth_design(
            order, cutoff, filter_type="bandstop", output="sos"
        )

        sos_scipy = scipy_signal.butter(
            order, cutoff, btype="bandstop", output="sos"
        )

        assert sos.shape[0] == sos_scipy.shape[0]

        w = torch.linspace(0.01, math.pi, 100)
        h_ts = _sos_freqz(sos, w)
        h_scipy = _sos_freqz(torch.from_numpy(sos_scipy), w)

        # Use larger atol for near-zero stopband values
        torch.testing.assert_close(
            h_ts.abs(), h_scipy.abs(), rtol=1e-3, atol=1e-6
        )


class TestButterworthDesignOutputFormats:
    """Test different output formats."""

    def test_output_sos_shape(self) -> None:
        """SOS output should have correct shape."""
        sos = butterworth_design(4, 0.3, output="sos")
        assert sos.shape == (2, 6)

    def test_output_zpk(self) -> None:
        """ZPK output should return tuple of 3 tensors."""
        zeros, poles, gain = butterworth_design(4, 0.3, output="zpk")
        assert zeros.numel() == 4  # 4 zeros
        assert poles.numel() == 4  # 4 poles
        assert gain.numel() == 1  # scalar gain

    def test_output_ba(self) -> None:
        """BA output should return tuple of 2 tensors."""
        numerator, denominator = butterworth_design(4, 0.3, output="ba")
        assert numerator.numel() == 5  # order + 1 coefficients
        assert denominator.numel() == 5


class TestButterworthDesignGradients:
    """Test gradient support."""

    def test_gradient_wrt_cutoff(self) -> None:
        """Should have gradient w.r.t. cutoff frequency."""
        cutoff = torch.tensor(0.3, requires_grad=True)
        sos = butterworth_design(4, cutoff, output="sos")

        loss = sos.sum()
        loss.backward()

        assert cutoff.grad is not None
        assert not torch.isnan(cutoff.grad)


class TestButterworthDesignSamplingFrequency:
    """Test sampling_frequency parameter."""

    def test_sampling_frequency_converts_hz_to_normalized(self) -> None:
        """sampling_frequency parameter should convert Hz to normalized frequency."""
        sampling_frequency = 1000.0
        cutoff_hz = 100.0

        sos_hz = butterworth_design(
            4, cutoff_hz, sampling_frequency=sampling_frequency
        )
        sos_norm = butterworth_design(4, cutoff_hz / (sampling_frequency / 2))

        torch.testing.assert_close(sos_hz, sos_norm)


def _sos_freqz(sos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute frequency response of SOS filter."""
    h = torch.ones(len(w), dtype=torch.complex128)
    z = torch.exp(1j * w)

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section.to(torch.float64)
        num = b0 + b1 * z**-1 + b2 * z**-2
        den = a0 + a1 * z**-1 + a2 * z**-2
        h = h * num / den

    return h
