"""Tests for FilterDesigner high-level interface."""

from __future__ import annotations

import pytest
import torch


class TestFilterDesigner:
    """Tests for FilterDesigner class."""

    def test_init_default(self):
        """Test default initialization."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        assert designer.dtype is None
        assert designer.device is None

    def test_init_with_dtype_device(self):
        """Test initialization with dtype and device."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner(
            dtype=torch.float64, device=torch.device("cpu")
        )
        assert designer.dtype == torch.float64
        assert designer.device == torch.device("cpu")


class TestFilterDesignerLowpass:
    """Tests for lowpass filter design."""

    def test_lowpass_butterworth_default(self):
        """Test lowpass design with default Butterworth method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4)

        assert filt.is_iir
        assert filt.sos is not None
        assert filt.sos.shape[0] == 2  # 4th order = 2 biquads
        assert filt.sos.shape[1] == 6

    def test_lowpass_butterworth_explicit(self):
        """Test lowpass design with explicit Butterworth method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        assert filt.is_iir
        assert filt.sos is not None

    def test_lowpass_chebyshev1(self):
        """Test lowpass design with Chebyshev Type I method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(
            cutoff=0.3, order=4, method="chebyshev1", ripple=1.0
        )

        assert filt.is_iir
        assert filt.sos is not None
        assert filt.sos.shape[0] == 2

    def test_lowpass_chebyshev2(self):
        """Test lowpass design with Chebyshev Type II method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(
            cutoff=0.3, order=4, method="chebyshev2", attenuation=40.0
        )

        assert filt.is_iir
        assert filt.sos is not None

    def test_lowpass_elliptic(self):
        """Test lowpass design with elliptic method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(
            cutoff=0.3,
            order=4,
            method="elliptic",
            ripple=1.0,
            attenuation=40.0,
        )

        assert filt.is_iir
        assert filt.sos is not None

    def test_lowpass_bessel(self):
        """Test lowpass design with Bessel method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="bessel")

        assert filt.is_iir
        assert filt.sos is not None

    def test_lowpass_fir(self):
        """Test lowpass FIR design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")

        assert not filt.is_iir
        assert filt.fir is not None
        assert filt.fir.shape[0] == 51


class TestFilterDesignerHighpass:
    """Tests for highpass filter design."""

    def test_highpass_butterworth(self):
        """Test highpass design with Butterworth method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.highpass(cutoff=0.3, order=4, method="butterworth")

        assert filt.is_iir
        assert filt.sos is not None

    def test_highpass_fir(self):
        """Test highpass FIR design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        # FIR highpass needs odd number of taps
        filt = designer.highpass(cutoff=0.3, num_taps=51, method="firwin")

        assert not filt.is_iir
        assert filt.fir is not None
        assert filt.fir.shape[0] == 51


class TestFilterDesignerBandpass:
    """Tests for bandpass filter design."""

    def test_bandpass_butterworth(self):
        """Test bandpass design with Butterworth method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.bandpass(
            low=0.2, high=0.4, order=4, method="butterworth"
        )

        assert filt.is_iir
        assert filt.sos is not None
        # Bandpass doubles the order
        assert filt.sos.shape[0] == 4

    def test_bandpass_fir(self):
        """Test bandpass FIR design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.bandpass(
            low=0.2, high=0.4, num_taps=51, method="firwin"
        )

        assert not filt.is_iir
        assert filt.fir is not None


class TestFilterDesignerBandstop:
    """Tests for bandstop filter design."""

    def test_bandstop_butterworth(self):
        """Test bandstop design with Butterworth method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.bandstop(
            low=0.2, high=0.4, order=4, method="butterworth"
        )

        assert filt.is_iir
        assert filt.sos is not None

    def test_bandstop_fir(self):
        """Test bandstop FIR design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        # FIR bandstop needs odd number of taps
        filt = designer.bandstop(
            low=0.2, high=0.4, num_taps=51, method="firwin"
        )

        assert not filt.is_iir
        assert filt.fir is not None


class TestFilterDesignerNotchPeak:
    """Tests for notch and peak filter design."""

    def test_notch(self):
        """Test notch filter design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.notch(frequency=0.25, quality_factor=30.0)

        assert filt.is_iir
        # Notch filter returns BA format
        assert filt.ba is not None
        b, a = filt.ba
        assert b.shape[0] == 3
        assert a.shape[0] == 3

    def test_peak(self):
        """Test peak filter design."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.peak(frequency=0.25, quality_factor=30.0)

        assert filt.is_iir
        assert filt.ba is not None


class TestFilterApply:
    """Tests for filter application."""

    def test_apply_iir(self):
        """Test applying IIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        x = torch.randn(100, dtype=torch.float64)
        y = filt.apply(x)

        assert y.shape == x.shape

    def test_apply_fir(self):
        """Test applying FIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")

        x = torch.randn(100, dtype=torch.float64)
        y = filt.apply(x)

        assert y.shape == x.shape

    def test_apply_batched(self):
        """Test applying filter to batched signal."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        x = torch.randn(3, 100, dtype=torch.float64)
        y = filt.apply(x, axis=-1)

        assert y.shape == x.shape

    def test_apply_axis(self):
        """Test applying filter along different axis."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        x = torch.randn(100, 3, dtype=torch.float64)
        y = filt.apply(x, axis=0)

        assert y.shape == x.shape


class TestFilterZeroPhase:
    """Tests for zero-phase filtering."""

    def test_zero_phase_iir(self):
        """Test zero-phase filtering with IIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        x = torch.randn(100, dtype=torch.float64)
        y = filt.apply_zero_phase(x)

        assert y.shape == x.shape

    def test_zero_phase_fir(self):
        """Test zero-phase filtering with FIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")

        # Use longer signal to accommodate padding requirements
        x = torch.randn(200, dtype=torch.float64)
        y = filt.apply_zero_phase(x)

        assert y.shape == x.shape


class TestFilterFrequencyResponse:
    """Tests for frequency response computation."""

    def test_frequency_response_iir(self):
        """Test frequency response computation for IIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        w, h = filt.frequency_response(n_points=256)

        assert w.shape == (256,)
        assert h.shape == (256,)
        assert torch.is_complex(h)

    def test_frequency_response_fir(self):
        """Test frequency response computation for FIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")

        w, h = filt.frequency_response(n_points=256)

        assert w.shape == (256,)
        assert h.shape == (256,)

    def test_frequency_response_cutoff(self):
        """Test that frequency response is approximately -3dB at cutoff."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        w, h = filt.frequency_response(n_points=1024)

        # Find the response at the cutoff frequency
        cutoff_idx = int(0.3 * 1024)
        mag_db = 20 * torch.log10(torch.abs(h[cutoff_idx]))

        # Butterworth should be -3 dB at cutoff
        assert abs(mag_db.item() + 3.0) < 0.5


class TestFilterGroupDelay:
    """Tests for group delay computation."""

    def test_group_delay_iir(self):
        """Test group delay computation for IIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        w, gd = filt.group_delay(n_points=256)

        assert w.shape == (256,)
        assert gd.shape == (256,)

    def test_group_delay_fir(self):
        """Test group delay computation for FIR filter."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, num_taps=51, method="firwin")

        w, gd = filt.group_delay(n_points=256)

        assert w.shape == (256,)
        assert gd.shape == (256,)

        # Linear phase FIR should have constant group delay
        # (equal to (num_taps - 1) / 2)
        expected_delay = (51 - 1) / 2.0
        # Check that group delay is approximately constant in passband
        passband_gd = gd[: len(gd) // 4]  # Check first quarter
        assert torch.allclose(
            passband_gd, torch.full_like(passband_gd, expected_delay), atol=0.5
        )


class TestFilterConversions:
    """Tests for filter format conversions."""

    def test_to_sos(self):
        """Test conversion to SOS format."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        sos = filt.to_sos()
        assert sos.shape[0] == 2
        assert sos.shape[1] == 6

    def test_to_ba(self):
        """Test conversion to BA format."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        b, a = filt.to_ba()
        assert b.shape[0] == 5  # 4th order + 1
        assert a.shape[0] == 5

    def test_to_zpk(self):
        """Test conversion to ZPK format."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method="butterworth")

        z, p, k = filt.to_zpk()
        assert z.shape[0] == 4
        assert p.shape[0] == 4


class TestFilterDifferentMethods:
    """Tests for different filter design methods."""

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("butterworth", {}),
            ("chebyshev1", {"ripple": 1.0}),
            ("chebyshev2", {"attenuation": 40.0}),
            ("elliptic", {"ripple": 1.0, "attenuation": 40.0}),
            ("bessel", {}),
        ],
    )
    def test_iir_methods(self, method, kwargs):
        """Test all IIR design methods."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(cutoff=0.3, order=4, method=method, **kwargs)

        assert filt.is_iir
        assert filt.sos is not None

        # All filters should attenuate at high frequencies
        w, h = filt.frequency_response(n_points=256)
        assert torch.abs(h[-1]) < 0.1  # High frequency should be attenuated

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("firwin", {}),
            ("firwin", {"window": "hamming"}),
            ("firwin", {"window": "blackman"}),
            ("firwin", {"window": ("kaiser", 8.6)}),
        ],
    )
    def test_fir_methods(self, method, kwargs):
        """Test FIR design methods with different windows."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        filt = designer.lowpass(
            cutoff=0.3, num_taps=51, method=method, **kwargs
        )

        assert not filt.is_iir
        assert filt.fir is not None


class TestFilterSamplingFrequency:
    """Tests for filters with sampling frequency."""

    def test_lowpass_with_sampling_frequency(self):
        """Test lowpass design with explicit sampling frequency."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        # 100 Hz cutoff at 1000 Hz sample rate = 0.2 normalized
        filt = designer.lowpass(
            cutoff=100.0,
            order=4,
            method="butterworth",
            sampling_frequency=1000.0,
        )

        assert filt.is_iir
        assert filt.sos is not None

    def test_bandpass_with_sampling_frequency(self):
        """Test bandpass design with explicit sampling frequency."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        # 100-200 Hz passband at 1000 Hz sample rate
        filt = designer.bandpass(
            low=100.0,
            high=200.0,
            order=4,
            method="butterworth",
            sampling_frequency=1000.0,
        )

        assert filt.is_iir


class TestFilterErrors:
    """Tests for error handling."""

    def test_order_and_num_taps_error(self):
        """Test error when both order and num_taps specified."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="Cannot specify both"):
            designer.lowpass(cutoff=0.3, order=4, num_taps=51)

    def test_neither_order_nor_num_taps_error(self):
        """Test error when neither order nor num_taps specified."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="Must specify either"):
            designer.lowpass(cutoff=0.3)

    def test_invalid_method_error(self):
        """Test error for invalid method."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="Unknown method"):
            designer.lowpass(cutoff=0.3, order=4, method="invalid")

    def test_fir_method_with_order_error(self):
        """Test error when FIR method used with order instead of num_taps."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="requires num_taps"):
            designer.lowpass(cutoff=0.3, order=4, method="firwin")

    def test_iir_method_with_num_taps_error(self):
        """Test error when IIR method used with num_taps instead of order."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="requires order"):
            designer.lowpass(cutoff=0.3, num_taps=51, method="butterworth")

    def test_chebyshev1_missing_ripple(self):
        """Test error when Chebyshev1 specified without ripple."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="ripple"):
            designer.lowpass(cutoff=0.3, order=4, method="chebyshev1")

    def test_elliptic_missing_params(self):
        """Test error when elliptic specified without ripple and attenuation."""
        from torchscience.signal_processing.filter_design import FilterDesigner

        designer = FilterDesigner()
        with pytest.raises(ValueError, match="ripple.*attenuation"):
            designer.lowpass(cutoff=0.3, order=4, method="elliptic")


class TestFilter:
    """Tests for Filter class."""

    def test_filter_init_sos(self):
        """Test Filter initialization with SOS."""
        from torchscience.signal_processing.filter_design import Filter

        sos = torch.tensor(
            [
                [0.1, 0.2, 0.1, 1.0, -0.5, 0.2],
                [0.25, 0.5, 0.25, 1.0, -0.8, 0.4],
            ],
            dtype=torch.float64,
        )
        filt = Filter(sos=sos)

        assert filt.is_iir
        assert filt.sos is not None

    def test_filter_init_fir(self):
        """Test Filter initialization with FIR."""
        from torchscience.signal_processing.filter_design import Filter

        fir = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
        filt = Filter(fir=fir)

        assert not filt.is_iir
        assert filt.fir is not None

    def test_filter_init_ba(self):
        """Test Filter initialization with BA."""
        from torchscience.signal_processing.filter_design import Filter

        b = torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)
        a = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64)
        filt = Filter(ba=(b, a))

        assert filt.is_iir
        assert filt.ba is not None

    def test_filter_init_zpk(self):
        """Test Filter initialization with ZPK."""
        from torchscience.signal_processing.filter_design import Filter

        z = torch.tensor([-1.0, -1.0], dtype=torch.complex128)
        p = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.complex128)
        k = torch.tensor(0.25)
        filt = Filter(zpk=(z, p, k))

        assert filt.is_iir
        assert filt.zpk is not None
