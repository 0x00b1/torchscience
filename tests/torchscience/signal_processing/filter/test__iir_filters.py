"""Tests for IIR filter design functions."""

import numpy as np
import pytest
import torch
from scipy import signal as scipy_signal

from torchscience.signal_processing.filter import (
    bessel_design,
    bessel_prototype,
    butterworth_design,
    butterworth_minimum_order,
    butterworth_prototype,
    chebyshev_type_1_design,
    chebyshev_type_1_minimum_order,
    chebyshev_type_1_prototype,
    chebyshev_type_2_design,
    chebyshev_type_2_minimum_order,
    chebyshev_type_2_prototype,
    elliptic_design,
    elliptic_minimum_order,
    elliptic_prototype,
    iirnotch,
    iirpeak,
)


class TestButterworthPrototype:
    """Tests for butterworth_prototype function."""

    def test_basic_order_4(self):
        """Test 4th order prototype."""
        z, p, k = butterworth_prototype(4)
        assert z.numel() == 0
        assert p.numel() == 4
        assert k.item() == pytest.approx(1.0)

    def test_poles_on_unit_circle(self):
        """All poles should be on the unit circle."""
        z, p, k = butterworth_prototype(4)
        magnitudes = torch.abs(p)
        assert torch.allclose(
            magnitudes, torch.ones_like(magnitudes), atol=1e-6
        )

    def test_poles_in_left_half_plane(self):
        """All poles should be in the left half-plane."""
        z, p, k = butterworth_prototype(6)
        assert torch.all(p.real < 0)

    def test_scipy_comparison(self):
        """Compare poles with scipy."""
        for order in [2, 4, 6, 8]:
            z, p, k = butterworth_prototype(order)
            z_scipy, p_scipy, k_scipy = scipy_signal.buttap(order)

            # Sort poles by angle for comparison
            p_sorted = torch.sort(torch.angle(p)).values
            p_scipy_sorted = np.sort(np.angle(p_scipy))

            assert np.allclose(p_sorted.numpy(), p_scipy_sorted, atol=1e-10)

    def test_invalid_order_raises(self):
        """Order must be positive."""
        with pytest.raises(ValueError):
            butterworth_prototype(0)
        with pytest.raises(ValueError):
            butterworth_prototype(-1)


class TestChebyshevType1Prototype:
    """Tests for chebyshev_type_1_prototype function."""

    def test_basic_order_4(self):
        """Test 4th order prototype with 1dB ripple."""
        z, p, k = chebyshev_type_1_prototype(4, 1.0)
        assert z.numel() == 0
        assert p.numel() == 4

    def test_poles_in_left_half_plane(self):
        """All poles should be in the left half-plane."""
        z, p, k = chebyshev_type_1_prototype(6, 1.0)
        assert torch.all(p.real < 0)

    def test_scipy_comparison(self):
        """Compare poles with scipy."""
        for order, ripple in [(4, 1.0), (6, 0.5), (3, 3.0)]:
            z, p, k = chebyshev_type_1_prototype(order, ripple)
            z_scipy, p_scipy, k_scipy = scipy_signal.cheb1ap(order, ripple)

            # Sort poles for comparison
            p_sorted = np.sort(p.numpy())
            p_scipy_sorted = np.sort(p_scipy)

            assert np.allclose(p_sorted, p_scipy_sorted, rtol=1e-6)

    def test_invalid_order_raises(self):
        """Order must be positive."""
        with pytest.raises(ValueError):
            chebyshev_type_1_prototype(0, 1.0)

    def test_invalid_ripple_raises(self):
        """Ripple must be positive."""
        with pytest.raises(ValueError):
            chebyshev_type_1_prototype(4, 0.0)
        with pytest.raises(ValueError):
            chebyshev_type_1_prototype(4, -1.0)


class TestChebyshevType2Prototype:
    """Tests for chebyshev_type_2_prototype function."""

    def test_basic_order_4(self):
        """Test 4th order prototype with 40dB attenuation."""
        z, p, k = chebyshev_type_2_prototype(4, 40.0)
        # Type II has zeros
        assert z.numel() == 4
        assert p.numel() == 4

    def test_odd_order_fewer_zeros(self):
        """Odd order has one fewer zero."""
        z, p, k = chebyshev_type_2_prototype(5, 40.0)
        assert z.numel() == 4  # 5-1 = 4 zeros
        assert p.numel() == 5

    def test_zeros_on_imaginary_axis(self):
        """All zeros should be on the imaginary axis."""
        z, p, k = chebyshev_type_2_prototype(4, 40.0)
        assert torch.allclose(z.real, torch.zeros_like(z.real), atol=1e-10)

    def test_poles_in_left_half_plane(self):
        """All poles should be in the left half-plane."""
        z, p, k = chebyshev_type_2_prototype(6, 40.0)
        assert torch.all(p.real < 0)

    def test_scipy_comparison(self):
        """Compare with scipy."""
        for order, attn in [(4, 40.0), (6, 60.0), (3, 20.0)]:
            z, p, k = chebyshev_type_2_prototype(order, attn)
            z_scipy, p_scipy, k_scipy = scipy_signal.cheb2ap(order, attn)

            # Sort poles for comparison
            p_sorted = torch.sort(torch.angle(p)).values
            p_scipy_sorted = np.sort(np.angle(p_scipy))

            assert np.allclose(p_sorted.numpy(), p_scipy_sorted, atol=1e-10)

    def test_invalid_attenuation_raises(self):
        """Attenuation must be positive."""
        with pytest.raises(ValueError):
            chebyshev_type_2_prototype(4, 0.0)


class TestBesselPrototype:
    """Tests for bessel_prototype function."""

    def test_basic_order_4(self):
        """Test 4th order prototype."""
        z, p, k = bessel_prototype(4)
        assert z.numel() == 0
        assert p.numel() == 4

    def test_poles_in_left_half_plane(self):
        """All poles should be in the left half-plane."""
        z, p, k = bessel_prototype(6)
        assert torch.all(p.real < 0)

    def test_scipy_comparison_phase_norm(self):
        """Compare with scipy using phase normalization (default)."""
        for order in [2, 4, 6]:
            z, p, k = bessel_prototype(order, normalization="phase")
            z_scipy, p_scipy, k_scipy = scipy_signal.besselap(
                order, norm="phase"
            )

            # Sort poles for comparison
            p_sorted = np.sort(p.numpy())
            p_scipy_sorted = np.sort(p_scipy)

            # Allow 3% tolerance due to different iterative algorithms
            assert np.allclose(p_sorted, p_scipy_sorted, rtol=0.03)

    def test_scipy_comparison_delay_norm(self):
        """Compare with scipy using delay normalization."""
        for order in [2, 4, 6]:
            z, p, k = bessel_prototype(order, normalization="delay")
            z_scipy, p_scipy, k_scipy = scipy_signal.besselap(
                order, norm="delay"
            )

            # Sort poles for comparison
            p_sorted = np.sort(p.numpy())
            p_scipy_sorted = np.sort(p_scipy)

            assert np.allclose(p_sorted, p_scipy_sorted, rtol=0.03)

    def test_scipy_comparison_magnitude_norm(self):
        """Compare with scipy using magnitude normalization."""
        for order in [2, 4, 6]:
            z, p, k = bessel_prototype(order, normalization="magnitude")
            z_scipy, p_scipy, k_scipy = scipy_signal.besselap(
                order, norm="mag"
            )

            # Sort poles for comparison
            p_sorted = np.sort(p.numpy())
            p_scipy_sorted = np.sort(p_scipy)

            assert np.allclose(p_sorted, p_scipy_sorted, rtol=0.03)


class TestButterworthDesign:
    """Tests for butterworth_design function."""

    def test_lowpass_sos_output(self):
        """Test lowpass filter with SOS output."""
        sos = butterworth_design(4, 0.3)
        assert sos.shape == (2, 6)  # 4th order = 2 sections

    def test_highpass_sos_output(self):
        """Test highpass filter with SOS output."""
        sos = butterworth_design(4, 0.3, filter_type="highpass")
        assert sos.shape == (2, 6)

    def test_bandpass_sos_output(self):
        """Test bandpass filter with SOS output."""
        sos = butterworth_design(2, [0.2, 0.4], filter_type="bandpass")
        # Bandpass doubles the order
        assert sos.shape == (2, 6)

    def test_bandstop_sos_output(self):
        """Test bandstop filter with SOS output."""
        sos = butterworth_design(2, [0.2, 0.4], filter_type="bandstop")
        assert sos.shape == (2, 6)

    def test_zpk_output(self):
        """Test ZPK output format."""
        z, p, k = butterworth_design(4, 0.3, output="zpk")
        assert z.numel() == 4
        assert p.numel() == 4

    def test_ba_output(self):
        """Test BA output format."""
        b, a = butterworth_design(4, 0.3, output="ba")
        assert b.numel() == 5  # order + 1
        assert a.numel() == 5

    def test_scipy_comparison_lowpass(self):
        """Compare lowpass filter frequency response with scipy."""
        sos = butterworth_design(4, 0.3)
        sos_scipy = scipy_signal.butter(4, 0.3, output="sos")

        # Compare frequency response instead of raw coefficients
        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)
        assert np.allclose(np.angle(h_ours), np.angle(h_scipy), atol=1e-6)

    def test_scipy_comparison_highpass(self):
        """Compare highpass filter frequency response with scipy."""
        sos = butterworth_design(4, 0.3, filter_type="highpass")
        sos_scipy = scipy_signal.butter(4, 0.3, btype="highpass", output="sos")

        # Compare frequency response
        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)
        assert np.allclose(np.angle(h_ours), np.angle(h_scipy), atol=1e-6)


class TestChebyshevType1Design:
    """Tests for chebyshev_type_1_design function."""

    def test_lowpass_sos_output(self):
        """Test lowpass filter with SOS output."""
        sos = chebyshev_type_1_design(4, 0.3, passband_ripple_db=1.0)
        assert sos.shape == (2, 6)

    def test_scipy_comparison_lowpass(self):
        """Compare lowpass filter frequency response with scipy."""
        sos = chebyshev_type_1_design(4, 0.3, passband_ripple_db=1.0)
        sos_scipy = scipy_signal.cheby1(4, 1.0, 0.3, output="sos")

        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)

    def test_scipy_comparison_highpass(self):
        """Compare highpass filter frequency response with scipy."""
        sos = chebyshev_type_1_design(
            4, 0.3, passband_ripple_db=1.0, filter_type="highpass"
        )
        sos_scipy = scipy_signal.cheby1(
            4, 1.0, 0.3, btype="highpass", output="sos"
        )

        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)


class TestChebyshevType2Design:
    """Tests for chebyshev_type_2_design function."""

    def test_lowpass_sos_output(self):
        """Test lowpass filter with SOS output."""
        sos = chebyshev_type_2_design(4, 0.3, stopband_attenuation_db=40.0)
        assert sos.shape == (2, 6)

    def test_scipy_comparison_lowpass(self):
        """Compare lowpass filter frequency response with scipy."""
        sos = chebyshev_type_2_design(4, 0.3, stopband_attenuation_db=40.0)
        sos_scipy = scipy_signal.cheby2(4, 40.0, 0.3, output="sos")

        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)


class TestBesselDesign:
    """Tests for bessel_design function."""

    def test_lowpass_sos_output(self):
        """Test lowpass filter with SOS output."""
        sos = bessel_design(4, 0.3)
        assert sos.shape == (2, 6)

    def test_scipy_comparison_lowpass(self):
        """Compare lowpass filter frequency response with scipy."""
        sos = bessel_design(4, 0.3)
        sos_scipy = scipy_signal.bessel(4, 0.3, output="sos")

        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        # Allow 5% tolerance due to differences in Bessel pole computation algorithms
        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=0.05)

    def test_scipy_comparison_highpass(self):
        """Compare highpass filter frequency response with scipy."""
        sos = bessel_design(4, 0.3, filter_type="highpass")
        sos_scipy = scipy_signal.bessel(4, 0.3, btype="highpass", output="sos")

        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        # Allow 5% tolerance due to differences in Bessel pole computation algorithms
        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=0.05)


class TestButterworthMinimumOrder:
    """Tests for butterworth_minimum_order function."""

    def test_lowpass_order(self):
        """Test lowpass order estimation."""
        order, wn = butterworth_minimum_order(0.2, 0.3, 3, 40)
        scipy_order, _ = scipy_signal.buttord(0.2, 0.3, 3, 40)
        assert order == scipy_order

    def test_highpass_order(self):
        """Test highpass order estimation."""
        # For highpass, passband freq > stopband freq
        order, wn = butterworth_minimum_order(
            0.3, 0.2, 3, 40, filter_type="highpass"
        )
        # scipy infers highpass from freq relationship
        scipy_order, _ = scipy_signal.buttord(0.3, 0.2, 3, 40)
        assert order == scipy_order


class TestChebyshevType1MinimumOrder:
    """Tests for chebyshev_type_1_minimum_order function."""

    def test_lowpass_order(self):
        """Test lowpass order estimation."""
        order, wn = chebyshev_type_1_minimum_order(0.2, 0.3, 3, 40)
        scipy_order, _ = scipy_signal.cheb1ord(0.2, 0.3, 3, 40)
        assert order == scipy_order


class TestChebyshevType2MinimumOrder:
    """Tests for chebyshev_type_2_minimum_order function."""

    def test_lowpass_order(self):
        """Test lowpass order estimation."""
        order, wn = chebyshev_type_2_minimum_order(0.2, 0.3, 3, 40)
        scipy_order, _ = scipy_signal.cheb2ord(0.2, 0.3, 3, 40)
        assert order == scipy_order


class TestEllipticPrototype:
    """Tests for elliptic_prototype function."""

    def test_basic_order_4(self):
        """Test 4th order prototype."""
        z, p, k = elliptic_prototype(4, 1.0, 40.0)
        assert z.numel() == 4  # Even order has n zeros
        assert p.numel() == 4

    def test_odd_order_fewer_zeros(self):
        """Odd order has one fewer zero pair."""
        z, p, k = elliptic_prototype(5, 1.0, 40.0)
        assert z.numel() == 4  # 5-1 = 4 zeros
        assert p.numel() == 5

    def test_zeros_on_imaginary_axis(self):
        """All zeros should be on the imaginary axis."""
        z, p, k = elliptic_prototype(4, 1.0, 40.0)
        assert torch.allclose(z.real, torch.zeros_like(z.real), atol=1e-10)

    def test_poles_in_left_half_plane(self):
        """All poles should be in the left half-plane."""
        z, p, k = elliptic_prototype(6, 1.0, 40.0)
        assert torch.all(p.real < 0)

    def test_scipy_comparison(self):
        """Compare with scipy."""
        for order, rp, rs in [(4, 1.0, 40.0), (5, 0.5, 60.0), (3, 3.0, 20.0)]:
            z, p, k = elliptic_prototype(order, rp, rs, dtype=torch.float64)
            z_scipy, p_scipy, k_scipy = scipy_signal.ellipap(order, rp, rs)

            # Sort zeros for comparison
            if z.numel() > 0:
                z_sorted = torch.sort(z.imag).values
                z_scipy_sorted = np.sort(np.imag(z_scipy))
                assert np.allclose(z_sorted.numpy(), z_scipy_sorted, rtol=1e-6)

            # Sort poles by magnitude and imaginary part for stable comparison
            p_sorted = np.sort_complex(p.numpy())
            p_scipy_sorted = np.sort_complex(p_scipy)
            assert np.allclose(p_sorted, p_scipy_sorted, rtol=1e-6)

    def test_invalid_order_raises(self):
        """Order must be positive."""
        with pytest.raises(ValueError):
            elliptic_prototype(0, 1.0, 40.0)

    def test_invalid_ripple_raises(self):
        """Ripple must be positive."""
        with pytest.raises(ValueError):
            elliptic_prototype(4, 0.0, 40.0)
        with pytest.raises(ValueError):
            elliptic_prototype(4, -1.0, 40.0)

    def test_invalid_attenuation_raises(self):
        """Attenuation must be positive."""
        with pytest.raises(ValueError):
            elliptic_prototype(4, 1.0, 0.0)


class TestEllipticDesign:
    """Tests for elliptic_design function."""

    def test_lowpass_sos_output(self):
        """Test lowpass filter with SOS output."""
        sos = elliptic_design(
            4, 0.3, passband_ripple_db=1.0, stopband_attenuation_db=40.0
        )
        assert sos.shape == (2, 6)  # 4th order = 2 sections

    def test_highpass_sos_output(self):
        """Test highpass filter with SOS output."""
        sos = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            filter_type="highpass",
        )
        assert sos.shape == (2, 6)

    def test_bandpass_sos_output(self):
        """Test bandpass filter with SOS output."""
        sos = elliptic_design(
            2,
            [0.2, 0.4],
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            filter_type="bandpass",
        )
        # Bandpass doubles the order
        assert sos.shape == (2, 6)

    def test_zpk_output(self):
        """Test ZPK output format."""
        z, p, k = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            output="zpk",
        )
        assert z.numel() == 4
        assert p.numel() == 4

    def test_ba_output(self):
        """Test BA output format."""
        b, a = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            output="ba",
        )
        assert b.numel() == 5  # order + 1
        assert a.numel() == 5

    def test_scipy_comparison_lowpass(self):
        """Compare lowpass filter frequency response with scipy."""
        sos = elliptic_design(
            4, 0.3, passband_ripple_db=1.0, stopband_attenuation_db=40.0
        )
        sos_scipy = scipy_signal.ellip(4, 1.0, 40.0, 0.3, output="sos")

        # Compare frequency response
        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-6)

    def test_scipy_comparison_highpass(self):
        """Compare highpass filter frequency response with scipy."""
        sos = elliptic_design(
            4,
            0.3,
            passband_ripple_db=1.0,
            stopband_attenuation_db=40.0,
            filter_type="highpass",
            dtype=torch.float64,
        )
        sos_scipy = scipy_signal.ellip(
            4, 1.0, 40.0, 0.3, btype="highpass", output="sos"
        )

        # Compare frequency response
        w, h_ours = scipy_signal.sosfreqz(sos.numpy(), worN=128)
        w, h_scipy = scipy_signal.sosfreqz(sos_scipy, worN=128)

        assert np.allclose(np.abs(h_ours), np.abs(h_scipy), rtol=1e-5)


class TestEllipticMinimumOrder:
    """Tests for elliptic_minimum_order function."""

    def test_lowpass_order(self):
        """Test lowpass order estimation."""
        order, wn = elliptic_minimum_order(0.2, 0.3, 3, 40)
        scipy_order, _ = scipy_signal.ellipord(0.2, 0.3, 3, 40)
        assert order == scipy_order

    def test_highpass_order(self):
        """Test highpass order estimation."""
        # For highpass, passband freq > stopband freq
        order, wn = elliptic_minimum_order(
            0.3, 0.2, 3, 40, filter_type="highpass"
        )
        # scipy infers highpass from freq relationship
        scipy_order, _ = scipy_signal.ellipord(0.3, 0.2, 3, 40)
        assert order == scipy_order


class TestIirNotch:
    """Tests for iirnotch function."""

    def test_basic_output(self):
        """Test basic output shape."""
        b, a = iirnotch(0.1, 30.0)
        assert b.shape == (3,)
        assert a.shape == (3,)

    def test_coefficient_normalization(self):
        """Denominator should start with 1."""
        b, a = iirnotch(0.1, 30.0)
        assert a[0].item() == pytest.approx(1.0)

    def test_scipy_comparison(self):
        """Compare with scipy iirnotch."""
        for w0, Q in [(0.1, 30.0), (0.25, 20.0), (0.4, 50.0)]:
            b, a = iirnotch(w0, Q)
            b_scipy, a_scipy = scipy_signal.iirnotch(w0, Q)

            assert np.allclose(b.numpy(), b_scipy, rtol=1e-6)
            assert np.allclose(a.numpy(), a_scipy, rtol=1e-6)

    def test_scipy_comparison_with_fs(self):
        """Compare with scipy using explicit sampling frequency."""
        b, a = iirnotch(60.0, 30.0, sampling_frequency=1000.0)
        b_scipy, a_scipy = scipy_signal.iirnotch(60.0, 30.0, fs=1000.0)

        assert np.allclose(b.numpy(), b_scipy, rtol=1e-6)
        assert np.allclose(a.numpy(), a_scipy, rtol=1e-6)

    def test_notch_at_frequency(self):
        """Verify filter has a notch (zero response) at notch frequency."""
        w0 = 0.1
        b, a = iirnotch(w0, 30.0)

        # Evaluate frequency response at notch frequency
        w, h = scipy_signal.freqz(b.numpy(), a.numpy(), worN=[np.pi * w0])
        # At notch, response should be very small
        assert np.abs(h[0]) < 0.1

    def test_invalid_frequency_raises(self):
        """Notch frequency must be in valid range."""
        with pytest.raises(ValueError):
            iirnotch(0.0, 30.0)
        with pytest.raises(ValueError):
            iirnotch(1.0, 30.0)  # At Nyquist
        with pytest.raises(ValueError):
            iirnotch(-0.1, 30.0)

    def test_invalid_quality_factor_raises(self):
        """Quality factor must be positive."""
        with pytest.raises(ValueError):
            iirnotch(0.1, 0.0)
        with pytest.raises(ValueError):
            iirnotch(0.1, -1.0)


class TestIirPeak:
    """Tests for iirpeak function."""

    def test_basic_output(self):
        """Test basic output shape."""
        b, a = iirpeak(0.1, 30.0)
        assert b.shape == (3,)
        assert a.shape == (3,)

    def test_coefficient_normalization(self):
        """Denominator should start with 1."""
        b, a = iirpeak(0.1, 30.0)
        assert a[0].item() == pytest.approx(1.0)

    def test_scipy_comparison(self):
        """Compare with scipy iirpeak."""
        for w0, Q in [(0.1, 30.0), (0.25, 20.0), (0.4, 50.0)]:
            b, a = iirpeak(w0, Q)
            b_scipy, a_scipy = scipy_signal.iirpeak(w0, Q)

            assert np.allclose(b.numpy(), b_scipy, rtol=1e-6)
            assert np.allclose(a.numpy(), a_scipy, rtol=1e-6)

    def test_scipy_comparison_with_fs(self):
        """Compare with scipy using explicit sampling frequency."""
        b, a = iirpeak(60.0, 30.0, sampling_frequency=1000.0)
        b_scipy, a_scipy = scipy_signal.iirpeak(60.0, 30.0, fs=1000.0)

        assert np.allclose(b.numpy(), b_scipy, rtol=1e-6)
        assert np.allclose(a.numpy(), a_scipy, rtol=1e-6)

    def test_peak_at_frequency(self):
        """Verify filter has a peak at peak frequency."""
        w0 = 0.1
        b, a = iirpeak(w0, 30.0)

        # Evaluate frequency response at peak frequency
        w, h = scipy_signal.freqz(b.numpy(), a.numpy(), worN=[np.pi * w0])
        # At peak, response should be close to 1
        assert np.abs(h[0]) > 0.9

    def test_complement_of_notch(self):
        """Peak + notch should equal 1 at all frequencies."""
        w0 = 0.1
        Q = 30.0
        b_peak, a_peak = iirpeak(w0, Q, dtype=torch.float64)
        b_notch, a_notch = iirnotch(w0, Q, dtype=torch.float64)

        # Get frequency responses
        w, h_peak = scipy_signal.freqz(
            b_peak.numpy(), a_peak.numpy(), worN=128
        )
        w, h_notch = scipy_signal.freqz(
            b_notch.numpy(), a_notch.numpy(), worN=128
        )

        # Sum should be 1
        assert np.allclose(h_peak + h_notch, np.ones_like(h_peak), rtol=1e-5)

    def test_invalid_frequency_raises(self):
        """Peak frequency must be in valid range."""
        with pytest.raises(ValueError):
            iirpeak(0.0, 30.0)
        with pytest.raises(ValueError):
            iirpeak(1.0, 30.0)  # At Nyquist

    def test_invalid_quality_factor_raises(self):
        """Quality factor must be positive."""
        with pytest.raises(ValueError):
            iirpeak(0.1, 0.0)
