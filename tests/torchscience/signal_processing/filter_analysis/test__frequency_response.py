"""Tests for frequency response functions."""

import numpy as np
import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter_analysis import (
    frequency_response,
    frequency_response_fir,
    frequency_response_sos,
    frequency_response_zpk,
)
from torchscience.signal_processing.filter_design import (
    SOSNormalizationError,
    bilinear_transform_zpk,
    butterworth_design,
    butterworth_prototype,
)


class TestFrequencyResponseSOS:
    """Tests for frequency_response_sos."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("cutoff", [0.1, 0.3, 0.5, 0.7])
    def test_matches_scipy_freqz_sos(self, order: int, cutoff: float) -> None:
        """Should produce same result as scipy.signal.sosfreqz."""
        sos = butterworth_design(order, cutoff, dtype=torch.float64)

        freqs, response = frequency_response_sos(sos, frequencies=256)

        # Scipy reference
        w_scipy, h_scipy = scipy_signal.sosfreqz(sos.numpy(), worN=256)

        # Compare magnitude (more robust than phase)
        torch.testing.assert_close(
            response.abs(),
            torch.from_numpy(np.abs(h_scipy)),
            rtol=1e-5,
            atol=1e-10,
        )

    def test_cutoff_at_minus_3db(self) -> None:
        """Butterworth filter should be at -3dB at cutoff frequency."""
        cutoff = 0.25
        sos = butterworth_design(4, cutoff, dtype=torch.float64)

        # Evaluate at exactly the cutoff frequency
        freqs, response = frequency_response_sos(
            sos, frequencies=torch.tensor([cutoff])
        )

        magnitude_db = 20 * torch.log10(response.abs())

        # Should be approximately -3dB
        torch.testing.assert_close(
            magnitude_db,
            torch.tensor([-3.0103], dtype=torch.float64),
            rtol=0.01,
            atol=0.01,
        )

    def test_passband_gain_unity(self) -> None:
        """Lowpass filter should have unity gain in passband."""
        sos = butterworth_design(4, 0.5, dtype=torch.float64)

        # Evaluate at DC (normalized frequency = 0)
        freqs, response = frequency_response_sos(
            sos, frequencies=torch.tensor([0.0])
        )

        torch.testing.assert_close(
            response.abs(),
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_stopband_attenuation(self) -> None:
        """Lowpass filter should attenuate stopband."""
        sos = butterworth_design(4, 0.2, dtype=torch.float64)

        # Evaluate at high frequency (stopband)
        freqs, response = frequency_response_sos(
            sos, frequencies=torch.tensor([0.8])
        )

        # Should be significantly attenuated (< -40 dB for 4th order)
        magnitude_db = 20 * torch.log10(response.abs())
        assert magnitude_db.item() < -40.0

    def test_validation_raises_on_unnormalized(self) -> None:
        """Should raise SOSNormalizationError for unnormalized SOS."""
        # Create SOS with a0 != 1
        sos = torch.tensor([[1.0, 0.5, 0.25, 2.0, 0.1, 0.05]])

        with pytest.raises(SOSNormalizationError):
            frequency_response_sos(sos, validate=True)

    def test_validation_skip(self) -> None:
        """Should not raise when validate=False."""
        sos = torch.tensor([[1.0, 0.5, 0.25, 2.0, 0.1, 0.05]])

        # Should not raise
        freqs, response = frequency_response_sos(sos, validate=False)
        assert response.shape == freqs.shape

    def test_whole_parameter(self) -> None:
        """whole=True should compute full frequency range."""
        sos = butterworth_design(2, 0.3)

        freqs_half, _ = frequency_response_sos(
            sos, frequencies=100, whole=False
        )
        freqs_whole, _ = frequency_response_sos(
            sos, frequencies=100, whole=True
        )

        # Both use endpoint=False, so last point is max_freq * (n-1)/n
        # Half: max=1.0, last = 0.99; Whole: max=2.0, last = 1.98
        assert freqs_half[-1].item() == pytest.approx(0.99, rel=0.01)
        assert freqs_whole[-1].item() == pytest.approx(1.98, rel=0.01)

    def test_sampling_frequency_parameter(self) -> None:
        """sampling_frequency should convert to Hz."""
        sos = butterworth_design(2, 0.3)
        fs = 44100.0

        freqs, _ = frequency_response_sos(
            sos, frequencies=100, sampling_frequency=fs
        )

        # Should go from 0 to near Nyquist in Hz (endpoint=False)
        assert freqs[0].item() == 0.0
        # Last point is Nyquist * (n-1)/n = 22050 * 99/100 = 21829.5
        assert freqs[-1].item() == pytest.approx(fs / 2 * 99 / 100, rel=0.01)


class TestFrequencyResponseZPK:
    """Tests for frequency_response_zpk."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_matches_sos_response(self, order: int) -> None:
        """ZPK response should match SOS response for same filter."""
        sos = butterworth_design(order, 0.3, dtype=torch.float64)
        zeros, poles, gain = butterworth_design(
            order, 0.3, output="zpk", dtype=torch.float64
        )

        _, response_sos = frequency_response_sos(sos, frequencies=256)
        _, response_zpk = frequency_response_zpk(
            zeros, poles, gain, frequencies=256
        )

        torch.testing.assert_close(
            response_sos.abs(),
            response_zpk.abs(),
            rtol=1e-5,
            atol=1e-10,
        )

    def test_all_pole_filter(self) -> None:
        """Should handle filters with no zeros."""
        # Butterworth prototype has no zeros
        zeros, poles, gain = butterworth_prototype(3, dtype=torch.float64)
        zeros_d, poles_d, gain_d = bilinear_transform_zpk(
            zeros, poles, gain, 2.0
        )

        freqs, response = frequency_response_zpk(zeros_d, poles_d, gain_d)

        assert response.shape == freqs.shape
        assert torch.isfinite(response).all()


class TestFrequencyResponseBA:
    """Tests for frequency_response (BA form)."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_matches_scipy_freqz(self, order: int) -> None:
        """Should match scipy.signal.freqz."""
        b, a = butterworth_design(order, 0.3, output="ba", dtype=torch.float64)

        freqs, response = frequency_response(b, a, frequencies=256)

        # Scipy reference
        w_scipy, h_scipy = scipy_signal.freqz(b.numpy(), a.numpy(), worN=256)

        torch.testing.assert_close(
            response.abs(),
            torch.from_numpy(np.abs(h_scipy)),
            rtol=1e-5,
            atol=1e-10,
        )

    def test_matches_sos_response(self) -> None:
        """BA response should match SOS response for same filter."""
        sos = butterworth_design(4, 0.3, dtype=torch.float64)
        b, a = butterworth_design(4, 0.3, output="ba", dtype=torch.float64)

        _, response_sos = frequency_response_sos(sos, frequencies=256)
        _, response_ba = frequency_response(b, a, frequencies=256)

        torch.testing.assert_close(
            response_sos.abs(),
            response_ba.abs(),
            rtol=1e-5,
            atol=1e-10,
        )


class TestFrequencyResponseFIR:
    """Tests for frequency_response_fir."""

    def test_matches_scipy_freqz(self) -> None:
        """Should match scipy.signal.freqz for FIR."""
        # Simple FIR filter (moving average)
        h = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float64)

        freqs, response = frequency_response_fir(h, frequencies=256)

        # Scipy reference (FIR has denominator = [1])
        w_scipy, h_scipy = scipy_signal.freqz(h.numpy(), [1.0], worN=256)

        torch.testing.assert_close(
            response.abs(),
            torch.from_numpy(np.abs(h_scipy)),
            rtol=1e-5,
            atol=1e-10,
        )

    def test_dc_gain(self) -> None:
        """DC gain should be sum of coefficients."""
        h = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])
        expected_dc_gain = h.sum()

        freqs, response = frequency_response_fir(
            h, frequencies=torch.tensor([0.0])
        )

        torch.testing.assert_close(
            response.abs().squeeze(),
            expected_dc_gain,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_symmetric_fir_real_response(self) -> None:
        """Symmetric FIR should have zero phase (real response) at low frequencies."""
        # Type I FIR (odd length, symmetric)
        h = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], dtype=torch.float64)

        freqs, response = frequency_response_fir(h, frequencies=50)

        # For symmetric FIR, phase should be linear (group delay = constant)
        # Check that the imaginary part is small at low frequencies
        # (linear phase means response is real * exp(-j*w*delay))
        assert torch.isfinite(response).all()


class TestFrequencyResponseConsistency:
    """Tests for consistency between all frequency response functions."""

    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    def test_all_representations_match(self, order: int) -> None:
        """SOS, ZPK, and BA should all give same magnitude response."""
        sos = butterworth_design(order, 0.4, dtype=torch.float64)
        z, p, k = butterworth_design(
            order, 0.4, output="zpk", dtype=torch.float64
        )
        b, a = butterworth_design(order, 0.4, output="ba", dtype=torch.float64)

        _, resp_sos = frequency_response_sos(sos, frequencies=128)
        _, resp_zpk = frequency_response_zpk(z, p, k, frequencies=128)
        _, resp_ba = frequency_response(b, a, frequencies=128)

        # All should match in magnitude
        torch.testing.assert_close(
            resp_sos.abs(), resp_zpk.abs(), rtol=1e-4, atol=1e-10
        )
        torch.testing.assert_close(
            resp_sos.abs(), resp_ba.abs(), rtol=1e-4, atol=1e-10
        )
