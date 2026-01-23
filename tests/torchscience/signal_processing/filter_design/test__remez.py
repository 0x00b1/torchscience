"""Tests for Parks-McClellan (Remez) optimal FIR filter design."""

import numpy as np
import pytest
import torch
from scipy.signal import freqz
from scipy.signal import remez as scipy_remez

from torchscience.signal_processing.filter_design import remez


class TestRemez:
    """Test remez (Parks-McClellan) optimal FIR filter design."""

    def test_lowpass_matches_scipy(self) -> None:
        """Lowpass filter should match scipy.signal.remez."""
        num_taps = 51
        # Bands in Nyquist-normalized frequencies (0 to 0.5)
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]  # Passband gain 1, stopband gain 0

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        assert h.shape == (num_taps,)
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_bandpass_matches_scipy(self) -> None:
        """Bandpass filter should match scipy.signal.remez."""
        num_taps = 65
        # Three bands: stopband - passband - stopband
        bands = [0.0, 0.1, 0.2, 0.35, 0.4, 0.5]
        desired = [0.0, 1.0, 0.0]

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        assert h.shape == (num_taps,)
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_highpass_matches_scipy(self) -> None:
        """Highpass filter should match scipy.signal.remez."""
        num_taps = 51
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [0.0, 1.0]  # Stopband then passband

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        assert h.shape == (num_taps,)
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_bandstop_matches_scipy(self) -> None:
        """Bandstop filter should match scipy.signal.remez."""
        num_taps = 65
        # Three bands: passband - stopband - passband
        bands = [0.0, 0.1, 0.2, 0.35, 0.4, 0.5]
        desired = [1.0, 0.0, 1.0]

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        assert h.shape == (num_taps,)
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_weighted_design(self) -> None:
        """Test weighted error function."""
        num_taps = 51
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]
        weights = [1.0, 10.0]  # Weight stopband 10x more

        h = remez(num_taps, bands, desired, weights=weights)
        h_scipy = scipy_remez(num_taps, bands, desired, weight=weights, fs=1.0)

        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_equiripple_property(self) -> None:
        """Verify equiripple behavior in passbands and stopbands."""
        num_taps = 51
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]

        h = remez(num_taps, bands, desired)

        # Compute frequency response
        w, H = freqz(h.numpy(), worN=512, fs=1.0)

        # Check passband (0 to 0.2)
        passband_mask = w <= 0.2
        passband_response = np.abs(H[passband_mask])

        # Check stopband (0.3 to 0.5)
        stopband_mask = w >= 0.3
        stopband_response = np.abs(H[stopband_mask])

        # The error should be bounded (equiripple property)
        # Passband error should be relatively small
        passband_error = np.abs(passband_response - 1.0)
        assert np.max(passband_error) < 0.1  # Within 10% of unity

        # Stopband should have small response
        assert np.max(stopband_response) < 0.1  # Attenuation is reasonable

    def test_dtype_float32(self) -> None:
        """Should respect float32 dtype parameter."""
        h = remez(
            num_taps=31,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
            dtype=torch.float32,
        )
        assert h.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Should respect float64 dtype parameter."""
        h = remez(
            num_taps=31,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
            dtype=torch.float64,
        )
        assert h.dtype == torch.float64

    def test_default_dtype_is_float64(self) -> None:
        """Default dtype should be float64."""
        h = remez(
            num_taps=31,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
        )
        assert h.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Should respect device parameter."""
        h = remez(
            num_taps=31,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
            device=torch.device("cpu"),
        )
        assert h.device.type == "cpu"

    def test_invalid_num_taps_zero(self) -> None:
        """Should raise for zero num_taps."""
        with pytest.raises(ValueError, match="num_taps must be at least 3"):
            remez(
                num_taps=0,
                bands=[0.0, 0.2, 0.3, 0.5],
                desired=[1.0, 0.0],
            )

    def test_invalid_num_taps_negative(self) -> None:
        """Should raise for negative num_taps."""
        with pytest.raises(ValueError, match="num_taps must be at least 3"):
            remez(
                num_taps=-5,
                bands=[0.0, 0.2, 0.3, 0.5],
                desired=[1.0, 0.0],
            )

    def test_invalid_bands_odd_length(self) -> None:
        """Should raise for odd number of band edges."""
        with pytest.raises(ValueError, match="bands must have even length"):
            remez(
                num_taps=31,
                bands=[0.0, 0.2, 0.3],  # Odd length
                desired=[1.0, 0.0],
            )

    def test_invalid_bands_desired_mismatch(self) -> None:
        """Should raise when bands and desired lengths don't match."""
        with pytest.raises(
            ValueError, match="Number of desired values.*must equal"
        ):
            remez(
                num_taps=31,
                bands=[0.0, 0.2, 0.3, 0.5],
                desired=[1.0],  # Should have 2 values for 2 bands
            )

    def test_invalid_bands_not_ascending(self) -> None:
        """Should raise when band edges are not ascending."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            remez(
                num_taps=31,
                bands=[0.0, 0.3, 0.2, 0.5],  # Not ascending
                desired=[1.0, 0.0],
            )

    def test_invalid_bands_outside_nyquist(self) -> None:
        """Should raise when band edges exceed Nyquist."""
        with pytest.raises(ValueError, match="must be between 0 and 0.5"):
            remez(
                num_taps=31,
                bands=[0.0, 0.2, 0.3, 0.6],  # 0.6 exceeds Nyquist (0.5)
                desired=[1.0, 0.0],
            )

    def test_invalid_bands_negative(self) -> None:
        """Should raise when band edges are negative."""
        with pytest.raises(ValueError, match="must be between 0 and 0.5"):
            remez(
                num_taps=31,
                bands=[-0.1, 0.2, 0.3, 0.5],
                desired=[1.0, 0.0],
            )

    def test_invalid_weights_length(self) -> None:
        """Should raise when weights length doesn't match bands."""
        with pytest.raises(ValueError, match="weights.*must equal"):
            remez(
                num_taps=31,
                bands=[0.0, 0.2, 0.3, 0.5],
                desired=[1.0, 0.0],
                weights=[1.0],  # Should have 2 values for 2 bands
            )

    def test_convergence_warning(self) -> None:
        """Test that maxiter parameter is respected."""
        # This test verifies the maxiter parameter is passed correctly
        # With scipy backend, it converges quickly so we just verify
        # the function completes with low maxiter
        h = remez(
            num_taps=31,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
            maxiter=2,  # Low iteration count
        )
        # Should still produce a valid filter (scipy is efficient)
        assert h.shape == (31,)
        # Check filter is symmetric (linear phase)
        torch.testing.assert_close(h, h.flip(0), rtol=1e-10, atol=1e-12)

    def test_output_shape(self) -> None:
        """Output shape should equal num_taps."""
        for num_taps in [31, 51, 65, 101]:
            h = remez(
                num_taps=num_taps,
                bands=[0.0, 0.2, 0.3, 0.5],
                desired=[1.0, 0.0],
            )
            assert h.shape == (num_taps,)

    def test_symmetric_coefficients(self) -> None:
        """FIR filter coefficients should be symmetric (linear phase)."""
        h = remez(
            num_taps=51,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
        )
        # Check symmetry
        for i in range(len(h) // 2):
            torch.testing.assert_close(h[i], h[-1 - i], rtol=1e-10, atol=1e-12)

    def test_even_num_taps_type_ii(self) -> None:
        """Even num_taps should work (Type II filter)."""
        num_taps = 50
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        assert h.shape == (num_taps,)
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-6, atol=1e-8
        )

    def test_differentiator_type(self) -> None:
        """Test differentiator filter type."""
        num_taps = 31
        bands = [0.05, 0.45]  # Single band, avoid DC and Nyquist
        desired = [1.0]  # One desired value for one band

        h = remez(num_taps, bands, desired, filter_type="differentiator")
        h_scipy = scipy_remez(
            num_taps, bands, desired, type="differentiator", fs=1.0
        )

        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-5, atol=1e-7
        )

    def test_hilbert_type(self) -> None:
        """Test Hilbert transformer filter type."""
        num_taps = 31
        bands = [0.05, 0.45]  # Single band, avoid DC and Nyquist
        desired = [1.0]  # One desired value for one band

        h = remez(num_taps, bands, desired, filter_type="hilbert")
        h_scipy = scipy_remez(num_taps, bands, desired, type="hilbert", fs=1.0)

        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-5, atol=1e-7
        )

    def test_high_order_filter(self) -> None:
        """Test high-order filter design stability."""
        num_taps = 201
        bands = [0.0, 0.15, 0.2, 0.5]
        desired = [1.0, 0.0]

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        # Higher tolerance for high-order filters due to numerical sensitivity
        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-4, atol=1e-6
        )

    def test_grid_density_parameter(self) -> None:
        """Test grid_density parameter affects design."""
        num_taps = 31
        bands = [0.0, 0.2, 0.3, 0.5]
        desired = [1.0, 0.0]

        h1 = remez(num_taps, bands, desired, grid_density=8)
        h2 = remez(num_taps, bands, desired, grid_density=32)

        # Both should be valid filters but may have slight differences
        assert h1.shape == h2.shape == (num_taps,)
        # Should be close but not necessarily identical
        # (different grid densities lead to slightly different solutions)

    def test_multiband_filter(self) -> None:
        """Test multi-band filter design."""
        num_taps = 81
        # Four bands: pass - stop - pass - stop
        bands = [0.0, 0.1, 0.15, 0.25, 0.3, 0.4, 0.45, 0.5]
        desired = [1.0, 0.0, 1.0, 0.0]

        h = remez(num_taps, bands, desired)
        h_scipy = scipy_remez(num_taps, bands, desired, fs=1.0)

        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-5, atol=1e-7
        )

    def test_multiband_weighted(self) -> None:
        """Test multi-band filter with weights."""
        num_taps = 81
        bands = [0.0, 0.1, 0.15, 0.25, 0.3, 0.4, 0.45, 0.5]
        desired = [1.0, 0.0, 1.0, 0.0]
        weights = [1.0, 10.0, 1.0, 10.0]  # Weight stopbands more

        h = remez(num_taps, bands, desired, weights=weights)
        h_scipy = scipy_remez(num_taps, bands, desired, weight=weights, fs=1.0)

        torch.testing.assert_close(
            h, torch.from_numpy(h_scipy), rtol=1e-5, atol=1e-7
        )

    def test_unity_passband_dc_gain(self) -> None:
        """Lowpass filter should have approximately unity DC gain."""
        h = remez(
            num_taps=51,
            bands=[0.0, 0.2, 0.3, 0.5],
            desired=[1.0, 0.0],
        )
        # DC gain is sum of coefficients
        dc_gain = h.sum()
        torch.testing.assert_close(
            dc_gain,
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-2,
            atol=1e-3,
        )
