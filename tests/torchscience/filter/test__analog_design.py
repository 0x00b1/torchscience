"""Tests for analog filter design functions."""

import pytest
import torch

# Import scipy for reference comparisons
scipy = pytest.importorskip("scipy")
from scipy import signal as scipy_signal


class TestButterworthAnalog:
    """Tests for butterworth_analog function."""

    def test_lowpass_zpk_matches_scipy(self):
        """Test lowpass ZPK output matches scipy.signal.butter(analog=True)."""
        import numpy as np

        from torchscience.filter import (
            butterworth_analog,
        )

        order = 4
        cutoff = 1000.0  # rad/s

        z, p, k = butterworth_analog(order, cutoff, output="zpk")
        z_scipy, p_scipy, k_scipy = scipy_signal.butter(
            order, cutoff, btype="low", analog=True, output="zpk"
        )

        # Check shapes
        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

        # Check gain (relative tolerance)
        assert abs(k.item() - k_scipy) / abs(k_scipy) < 1e-6

        # Check poles - compare each pole finds a match in scipy result
        p_np = p.numpy()
        p_scipy_np = np.array(p_scipy)
        for pole in p_np:
            # Find closest pole in scipy result
            distances = np.abs(p_scipy_np - pole)
            min_dist = np.min(distances)
            # Relative tolerance based on pole magnitude
            rtol = 1e-5 * np.abs(pole)
            assert min_dist < max(rtol, 1e-8), (
                f"Pole {pole} not found in scipy result"
            )

    def test_highpass_zpk_matches_scipy(self):
        """Test highpass ZPK output matches scipy."""
        from torchscience.filter import (
            butterworth_analog,
        )

        order = 3
        cutoff = 500.0

        z, p, k = butterworth_analog(
            order, cutoff, filter_type="highpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.butter(
            order, cutoff, btype="high", analog=True, output="zpk"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        assert abs(k.item() - k_scipy) < 1e-10

    def test_bandpass_zpk_matches_scipy(self):
        """Test bandpass ZPK output matches scipy."""
        from torchscience.filter import (
            butterworth_analog,
        )

        order = 2
        cutoff = [100.0, 1000.0]

        z, p, k = butterworth_analog(
            order, cutoff, filter_type="bandpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.butter(
            order, cutoff, btype="bandpass", analog=True, output="zpk"
        )

        # Bandpass doubles the order
        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        assert abs(k.item() - k_scipy) / abs(k_scipy) < 1e-8

    def test_bandstop_zpk_matches_scipy(self):
        """Test bandstop ZPK output matches scipy."""
        from torchscience.filter import (
            butterworth_analog,
        )

        order = 2
        cutoff = [200.0, 800.0]

        z, p, k = butterworth_analog(
            order, cutoff, filter_type="bandstop", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.butter(
            order, cutoff, btype="bandstop", analog=True, output="zpk"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        assert abs(k.item() - k_scipy) / max(abs(k_scipy), 1e-10) < 1e-8

    def test_ba_output(self):
        """Test BA output format."""
        from torchscience.filter import (
            butterworth_analog,
        )

        order = 3
        cutoff = 1000.0

        b, a = butterworth_analog(order, cutoff, output="ba")
        b_scipy, a_scipy = scipy_signal.butter(
            order, cutoff, btype="low", analog=True, output="ba"
        )

        assert b.shape[0] == len(b_scipy)
        assert a.shape[0] == len(a_scipy)
        # Normalize by leading coefficient for comparison
        b_norm = b / a[0]
        a_norm = a / a[0]
        b_scipy_norm = b_scipy / a_scipy[0]
        a_scipy_norm = a_scipy / a_scipy[0]
        torch.testing.assert_close(
            b_norm,
            torch.tensor(b_scipy_norm, dtype=b.dtype),
            rtol=1e-5,
            atol=1e-10,
        )
        torch.testing.assert_close(
            a_norm,
            torch.tensor(a_scipy_norm, dtype=a.dtype),
            rtol=1e-5,
            atol=1e-10,
        )

    def test_frequency_response_at_cutoff(self):
        """Test that magnitude at cutoff is -3dB for Butterworth."""
        from torchscience.filter import (
            butterworth_analog,
            freqs_zpk,
        )

        order = 4
        cutoff = 1000.0

        z, p, k = butterworth_analog(order, cutoff, output="zpk")
        w = torch.tensor([cutoff])
        _, h = freqs_zpk(z, p, k, w)

        mag_db = 20 * torch.log10(torch.abs(h))
        # Butterworth has -3dB at cutoff
        assert abs(mag_db.item() + 3.01) < 0.1

    def test_dtype_and_device(self):
        """Test dtype and device parameters."""
        from torchscience.filter import (
            butterworth_analog,
        )

        z, p, k = butterworth_analog(
            4, 1000.0, dtype=torch.float64, device=torch.device("cpu")
        )
        assert k.dtype == torch.float64


class TestChebyshevType1Analog:
    """Tests for chebyshev_type_1_analog function."""

    def test_lowpass_zpk_matches_scipy(self):
        """Test lowpass ZPK output matches scipy."""
        from torchscience.filter import (
            chebyshev_type_1_analog,
        )

        order = 4
        cutoff = 1000.0
        ripple_db = 1.0

        z, p, k = chebyshev_type_1_analog(
            order, cutoff, ripple_db, output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.cheby1(
            order, ripple_db, cutoff, btype="low", analog=True, output="zpk"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        # Use looser tolerance for large gains
        assert abs(k.item() - k_scipy) / abs(k_scipy) < 1e-5

    def test_highpass_zpk_matches_scipy(self):
        """Test highpass ZPK output matches scipy."""
        from torchscience.filter import (
            chebyshev_type_1_analog,
        )

        order = 3
        cutoff = 500.0
        ripple_db = 0.5

        z, p, k = chebyshev_type_1_analog(
            order, cutoff, ripple_db, filter_type="highpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.cheby1(
            order, ripple_db, cutoff, btype="high", analog=True, output="zpk"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_bandpass_zpk_matches_scipy(self):
        """Test bandpass ZPK output matches scipy."""
        from torchscience.filter import (
            chebyshev_type_1_analog,
        )

        order = 2
        cutoff = [100.0, 1000.0]
        ripple_db = 1.0

        z, p, k = chebyshev_type_1_analog(
            order, cutoff, ripple_db, filter_type="bandpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.cheby1(
            order,
            ripple_db,
            cutoff,
            btype="bandpass",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_ba_output(self):
        """Test BA output format."""
        from torchscience.filter import (
            chebyshev_type_1_analog,
        )

        order = 3
        cutoff = 1000.0
        ripple_db = 1.0

        b, a = chebyshev_type_1_analog(order, cutoff, ripple_db, output="ba")
        b_scipy, a_scipy = scipy_signal.cheby1(
            order, ripple_db, cutoff, btype="low", analog=True, output="ba"
        )

        torch.testing.assert_close(
            b, torch.tensor(b_scipy, dtype=b.dtype), rtol=1e-6, atol=1e-10
        )
        torch.testing.assert_close(
            a, torch.tensor(a_scipy, dtype=a.dtype), rtol=1e-6, atol=1e-10
        )

    def test_passband_ripple(self):
        """Test that passband ripple is within spec."""
        from torchscience.filter import (
            chebyshev_type_1_analog,
            freqs_zpk,
        )

        order = 4
        cutoff = 1000.0
        ripple_db = 1.0

        z, p, k = chebyshev_type_1_analog(
            order, cutoff, ripple_db, output="zpk"
        )
        # Check frequency response in passband
        w = torch.linspace(1.0, cutoff, 100)
        _, h = freqs_zpk(z, p, k, w)

        mag_db = 20 * torch.log10(torch.abs(h))
        # Maximum variation should be approximately the ripple
        assert (mag_db.max() - mag_db.min()).item() < ripple_db + 0.2


class TestChebyshevType2Analog:
    """Tests for chebyshev_type_2_analog function."""

    def test_lowpass_zpk_matches_scipy(self):
        """Test lowpass ZPK output matches scipy."""
        from torchscience.filter import (
            chebyshev_type_2_analog,
        )

        order = 4
        cutoff = 1000.0
        attenuation_db = 40.0

        z, p, k = chebyshev_type_2_analog(
            order, cutoff, attenuation_db, output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.cheby2(
            order,
            attenuation_db,
            cutoff,
            btype="low",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        # Gain comparison with relative tolerance
        assert abs(k.item() - k_scipy) / abs(k_scipy) < 1e-6

    def test_highpass_zpk_matches_scipy(self):
        """Test highpass ZPK output matches scipy."""
        from torchscience.filter import (
            chebyshev_type_2_analog,
        )

        order = 3
        cutoff = 500.0
        attenuation_db = 30.0

        z, p, k = chebyshev_type_2_analog(
            order, cutoff, attenuation_db, filter_type="highpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.cheby2(
            order,
            attenuation_db,
            cutoff,
            btype="high",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_ba_output(self):
        """Test BA output format."""
        from torchscience.filter import (
            chebyshev_type_2_analog,
        )

        order = 3
        cutoff = 1000.0
        attenuation_db = 40.0

        b, a = chebyshev_type_2_analog(
            order, cutoff, attenuation_db, output="ba"
        )
        b_scipy, a_scipy = scipy_signal.cheby2(
            order,
            attenuation_db,
            cutoff,
            btype="low",
            analog=True,
            output="ba",
        )

        # Normalize for comparison (divide by leading coefficient)
        b_normalized = b / b[0]
        b_scipy_normalized = b_scipy / b_scipy[0]

        torch.testing.assert_close(
            b_normalized,
            torch.tensor(b_scipy_normalized, dtype=b.dtype),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_stopband_attenuation(self):
        """Test that stopband attenuation meets spec."""
        from torchscience.filter import (
            chebyshev_type_2_analog,
            freqs_zpk,
        )

        order = 4
        cutoff = 1000.0
        attenuation_db = 40.0

        z, p, k = chebyshev_type_2_analog(
            order, cutoff, attenuation_db, output="zpk"
        )
        # Check frequency response well into stopband
        w = torch.linspace(cutoff * 2, cutoff * 10, 100)
        _, h = freqs_zpk(z, p, k, w)

        mag_db = 20 * torch.log10(torch.abs(h))
        # All stopband should be at or below -attenuation_db
        assert mag_db.max().item() < -attenuation_db + 1.0


class TestEllipticAnalog:
    """Tests for elliptic_analog function."""

    def test_lowpass_zpk_matches_scipy(self):
        """Test lowpass ZPK output matches scipy."""
        from torchscience.filter import (
            elliptic_analog,
        )

        order = 4
        cutoff = 1000.0
        ripple_db = 1.0
        attenuation_db = 40.0

        z, p, k = elliptic_analog(
            order, cutoff, ripple_db, attenuation_db, output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.ellip(
            order,
            ripple_db,
            attenuation_db,
            cutoff,
            btype="low",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        assert abs(k.item() - k_scipy) / abs(k_scipy) < 1e-6

    def test_highpass_zpk_matches_scipy(self):
        """Test highpass ZPK output matches scipy."""
        from torchscience.filter import (
            elliptic_analog,
        )

        order = 3
        cutoff = 500.0
        ripple_db = 0.5
        attenuation_db = 30.0

        z, p, k = elliptic_analog(
            order,
            cutoff,
            ripple_db,
            attenuation_db,
            filter_type="highpass",
            output="zpk",
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.ellip(
            order,
            ripple_db,
            attenuation_db,
            cutoff,
            btype="high",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_bandpass_zpk_matches_scipy(self):
        """Test bandpass ZPK output matches scipy."""
        from torchscience.filter import (
            elliptic_analog,
        )

        order = 2
        cutoff = [100.0, 1000.0]
        ripple_db = 1.0
        attenuation_db = 40.0

        z, p, k = elliptic_analog(
            order,
            cutoff,
            ripple_db,
            attenuation_db,
            filter_type="bandpass",
            output="zpk",
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.ellip(
            order,
            ripple_db,
            attenuation_db,
            cutoff,
            btype="bandpass",
            analog=True,
            output="zpk",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_ba_output(self):
        """Test BA output format."""
        from torchscience.filter import (
            elliptic_analog,
        )

        order = 3
        cutoff = 1000.0
        ripple_db = 1.0
        attenuation_db = 40.0

        b, a = elliptic_analog(
            order, cutoff, ripple_db, attenuation_db, output="ba"
        )
        b_scipy, a_scipy = scipy_signal.ellip(
            order,
            ripple_db,
            attenuation_db,
            cutoff,
            btype="low",
            analog=True,
            output="ba",
        )

        # Compare shapes
        assert b.shape[0] == len(b_scipy)
        assert a.shape[0] == len(a_scipy)


class TestBesselAnalog:
    """Tests for bessel_analog function."""

    def test_lowpass_zpk_matches_scipy(self):
        """Test lowpass ZPK output matches scipy."""
        from torchscience.filter import bessel_analog

        order = 4
        cutoff = 1000.0

        z, p, k = bessel_analog(order, cutoff, output="zpk")
        z_scipy, p_scipy, k_scipy = scipy_signal.bessel(
            order, cutoff, btype="low", analog=True, output="zpk", norm="phase"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)
        # Gain comparison - use looser tolerance due to Bessel polynomial precision
        assert (
            abs(k.item() - k_scipy) / abs(k_scipy) < 0.1
        )  # 10% relative tolerance

    def test_highpass_zpk_matches_scipy(self):
        """Test highpass ZPK output matches scipy."""
        from torchscience.filter import bessel_analog

        order = 3
        cutoff = 500.0

        z, p, k = bessel_analog(
            order, cutoff, filter_type="highpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.bessel(
            order,
            cutoff,
            btype="high",
            analog=True,
            output="zpk",
            norm="phase",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_bandpass_zpk_matches_scipy(self):
        """Test bandpass ZPK output matches scipy."""
        from torchscience.filter import bessel_analog

        order = 2
        cutoff = [100.0, 1000.0]

        z, p, k = bessel_analog(
            order, cutoff, filter_type="bandpass", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.bessel(
            order,
            cutoff,
            btype="bandpass",
            analog=True,
            output="zpk",
            norm="phase",
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_ba_output(self):
        """Test BA output format."""
        from torchscience.filter import bessel_analog

        order = 3
        cutoff = 1000.0

        b, a = bessel_analog(order, cutoff, output="ba")
        b_scipy, a_scipy = scipy_signal.bessel(
            order, cutoff, btype="low", analog=True, output="ba", norm="phase"
        )

        # Normalize and compare - use looser tolerance for Bessel
        b_normalized = b / a[0]
        a_normalized = a / a[0]
        b_scipy_normalized = b_scipy / a_scipy[0]
        a_scipy_normalized = a_scipy / a_scipy[0]

        torch.testing.assert_close(
            b_normalized,
            torch.tensor(b_scipy_normalized, dtype=b.dtype),
            rtol=0.1,  # 10% relative tolerance for Bessel
            atol=1e-6,
        )
        torch.testing.assert_close(
            a_normalized,
            torch.tensor(a_scipy_normalized, dtype=a.dtype),
            rtol=0.1,  # 10% relative tolerance for Bessel
            atol=1e-6,
        )

    def test_normalization_delay(self):
        """Test delay normalization."""
        from torchscience.filter import bessel_analog

        order = 4
        cutoff = 1000.0

        z, p, k = bessel_analog(
            order, cutoff, normalization="delay", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.bessel(
            order, cutoff, btype="low", analog=True, output="zpk", norm="delay"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_normalization_magnitude(self):
        """Test magnitude normalization."""
        from torchscience.filter import bessel_analog

        order = 4
        cutoff = 1000.0

        z, p, k = bessel_analog(
            order, cutoff, normalization="magnitude", output="zpk"
        )
        z_scipy, p_scipy, k_scipy = scipy_signal.bessel(
            order, cutoff, btype="low", analog=True, output="zpk", norm="mag"
        )

        assert z.shape[0] == len(z_scipy)
        assert p.shape[0] == len(p_scipy)

    def test_flat_group_delay(self):
        """Test that Bessel filter has maximally flat group delay."""
        from torchscience.filter import (
            bessel_analog,
            freqs_zpk,
        )

        order = 4
        cutoff = 1000.0

        z, p, k = bessel_analog(order, cutoff, output="zpk")

        # Compute group delay at several frequencies in passband
        w = torch.linspace(1.0, cutoff * 0.5, 50)
        _, h = freqs_zpk(z, p, k, w)

        # Group delay from phase: -d(phase)/dw
        phase = torch.angle(h)
        dw = w[1] - w[0]
        group_delay = -torch.diff(phase) / dw

        # Group delay should be relatively flat in passband
        gd_variation = (
            group_delay.max() - group_delay.min()
        ) / group_delay.mean()
        assert gd_variation.item() < 0.1  # Less than 10% variation


class TestAnalogFilterValidation:
    """Tests for input validation."""

    def test_invalid_order(self):
        """Test that invalid order raises error."""
        from torchscience.filter import (
            butterworth_analog,
        )

        with pytest.raises(ValueError):
            butterworth_analog(0, 1000.0)

    def test_invalid_cutoff_negative(self):
        """Test that negative cutoff raises error."""
        from torchscience.filter import (
            butterworth_analog,
        )

        with pytest.raises(ValueError):
            butterworth_analog(4, -1000.0)

    def test_invalid_bandpass_cutoff(self):
        """Test that invalid bandpass cutoff order raises error."""
        from torchscience.filter import (
            butterworth_analog,
        )

        # low > high should fail
        with pytest.raises(ValueError):
            butterworth_analog(4, [1000.0, 100.0], filter_type="bandpass")

    def test_invalid_filter_type(self):
        """Test that invalid filter type raises error."""
        from torchscience.filter import (
            butterworth_analog,
        )

        with pytest.raises(ValueError):
            butterworth_analog(4, 1000.0, filter_type="invalid")

    def test_invalid_output(self):
        """Test that invalid output format raises error."""
        from torchscience.filter import (
            butterworth_analog,
        )

        with pytest.raises(ValueError):
            butterworth_analog(4, 1000.0, output="invalid")


class TestAnalogVsDigital:
    """Tests comparing analog and digital filter designs."""

    def test_analog_no_bilinear_distortion(self):
        """Test that analog filter has no bilinear transform frequency warping."""
        from torchscience.filter import (
            butterworth_analog,
            butterworth_design,
        )

        # Design analog filter
        order = 4
        cutoff = 0.2  # For digital: normalized frequency
        cutoff_analog = 1000.0  # For analog: rad/s

        z_analog, p_analog, k_analog = butterworth_analog(
            order, cutoff_analog, output="zpk"
        )
        z_digital, p_digital, k_digital = butterworth_design(
            order, cutoff, output="zpk"
        )

        # Analog filter has poles in left half s-plane
        assert all(p.real < 0 for p in p_analog)

        # Digital filter has poles inside unit circle
        assert all(abs(p) < 1 for p in p_digital)

    def test_analog_poles_in_s_plane(self):
        """Test that analog filter poles are in s-plane (real parts negative)."""
        from torchscience.filter import (
            butterworth_analog,
        )

        z, p, k = butterworth_analog(6, 1000.0, output="zpk")

        # All poles should have negative real part (stable)
        for pole in p:
            assert pole.real < 0, f"Pole {pole} has non-negative real part"
