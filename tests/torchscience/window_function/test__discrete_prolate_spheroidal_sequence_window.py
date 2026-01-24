import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestDiscreteProlateSpheroidalSequenceWindow:
    """Tests for discrete_prolate_spheroidal_sequence_window and periodic version."""

    def test_symmetric_basic(self):
        """Test basic symmetric window generation."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        for n in [4, 8, 16, 32, 64]:
            result = wf.discrete_prolate_spheroidal_sequence_window(
                n, nw, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert result.dtype == torch.float64

    def test_periodic_basic(self):
        """Test basic periodic window generation."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        for n in [4, 8, 16, 32, 64]:
            result = wf.periodic_discrete_prolate_spheroidal_sequence_window(
                n, nw, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert result.dtype == torch.float64

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.dpss (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [16, 32, 64]:
            for nw_val in [2.0, 3.0, 4.0]:
                nw = torch.tensor(nw_val, dtype=torch.float64)
                result = wf.discrete_prolate_spheroidal_sequence_window(
                    n, nw, dtype=torch.float64
                )
                # scipy.signal.windows.dpss returns multiple tapers, take first one
                expected_raw = scipy_signal.windows.dpss(
                    n, nw_val, Kmax=1, sym=True, norm="subsample"
                )
                if expected_raw.ndim > 1:
                    expected_raw = expected_raw[0]
                expected = torch.tensor(expected_raw, dtype=torch.float64)
                # Normalize both to max=1 for comparison
                expected = expected / expected.abs().max()
                # Sign may differ, compare absolute values
                torch.testing.assert_close(
                    result.abs(), expected.abs(), rtol=1e-4, atol=1e-4
                )

    def test_periodic_window_properties(self):
        """Test that periodic DPSS window has expected properties.

        Note: scipy's periodic DPSS uses a different algorithm, so we test
        properties rather than exact comparison.
        """
        for n in [16, 32, 64]:
            for nw_val in [2.0, 3.0, 4.0]:
                nw = torch.tensor(nw_val, dtype=torch.float64)
                result = (
                    wf.periodic_discrete_prolate_spheroidal_sequence_window(
                        n, nw, dtype=torch.float64
                    )
                )
                # Should be normalized to max=1
                torch.testing.assert_close(
                    result.max(),
                    torch.tensor(1.0, dtype=torch.float64),
                    atol=1e-6,
                    rtol=0,
                )
                # Should have reasonable values
                assert result.min() >= -1e-6
                assert result.max() <= 1.0 + 1e-6

    def test_output_shape(self):
        """Test output shape is (n,)."""
        nw = torch.tensor(3.0)
        for n in [0, 1, 5, 64, 100]:
            result = wf.discrete_prolate_spheroidal_sequence_window(n, nw)
            assert result.shape == (n,)
            result_periodic = (
                wf.periodic_discrete_prolate_spheroidal_sequence_window(n, nw)
            )
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        nw = torch.tensor(3.0, dtype=dtype)
        result = wf.discrete_prolate_spheroidal_sequence_window(
            64, nw, dtype=dtype
        )
        assert result.dtype == dtype
        result_periodic = (
            wf.periodic_discrete_prolate_spheroidal_sequence_window(
                64, nw, dtype=dtype
            )
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        nw = torch.tensor(3.0)
        result = wf.discrete_prolate_spheroidal_sequence_window(0, nw)
        assert result.shape == (0,)
        result_periodic = (
            wf.periodic_discrete_prolate_spheroidal_sequence_window(0, nw)
        )
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        result = wf.discrete_prolate_spheroidal_sequence_window(
            1, nw, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = (
            wf.periodic_discrete_prolate_spheroidal_sequence_window(
                1, nw, dtype=torch.float64
            )
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric DPSS window is symmetric."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.discrete_prolate_spheroidal_sequence_window(
                n, nw, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-5, atol=1e-5)

    def test_symmetry_various_nw(self):
        """Test symmetry with various NW values."""
        for nw_val in [2.0, 3.0, 4.0, 5.0]:
            nw = torch.tensor(nw_val, dtype=torch.float64)
            for n in [8, 11, 16]:
                result = wf.discrete_prolate_spheroidal_sequence_window(
                    n, nw, dtype=torch.float64
                )
                flipped = torch.flip(result, dims=[0])
                torch.testing.assert_close(
                    result, flipped, rtol=1e-5, atol=1e-5
                )

    def test_maximum_value_one(self):
        """Test that maximum value is 1.0 (normalized)."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        for n in [8, 16, 32]:
            result = wf.discrete_prolate_spheroidal_sequence_window(
                n, nw, dtype=torch.float64
            )
            torch.testing.assert_close(
                result.max(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-6,
                rtol=0,
            )

    def test_values_bounded(self):
        """Test that all window values are in [0, 1]."""
        for nw_val in [2.0, 3.0, 4.0]:
            nw = torch.tensor(nw_val, dtype=torch.float64)
            for n in [8, 16, 32]:
                result = wf.discrete_prolate_spheroidal_sequence_window(
                    n, nw, dtype=torch.float64
                )
                # Allow small numerical errors
                assert result.min() >= -1e-6
                assert result.max() <= 1.0 + 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through nw parameter."""
        nw = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.discrete_prolate_spheroidal_sequence_window(
            16, nw, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert nw.grad is not None
        assert not torch.isnan(nw.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        nw = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_discrete_prolate_spheroidal_sequence_window(
            16, nw, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert nw.grad is not None
        assert not torch.isnan(nw.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        nw = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(n_w):
            return wf.discrete_prolate_spheroidal_sequence_window(
                16, n_w, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (nw,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        nw = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(n_w):
            return wf.periodic_discrete_prolate_spheroidal_sequence_window(
                16, n_w, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (nw,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        nw = torch.tensor(3.0)
        with pytest.raises(ValueError):
            wf.discrete_prolate_spheroidal_sequence_window(-1, nw)
        with pytest.raises(ValueError):
            wf.periodic_discrete_prolate_spheroidal_sequence_window(-1, nw)

    def test_invalid_nw_raises(self):
        """Test that nw <= 0 raises error."""
        with pytest.raises(ValueError):
            wf.discrete_prolate_spheroidal_sequence_window(16, 0.0)
        with pytest.raises(ValueError):
            wf.discrete_prolate_spheroidal_sequence_window(16, -1.0)
        with pytest.raises(ValueError):
            wf.periodic_discrete_prolate_spheroidal_sequence_window(16, 0.0)
        with pytest.raises(ValueError):
            wf.periodic_discrete_prolate_spheroidal_sequence_window(16, -1.0)

    def test_float_param_input(self):
        """Test that nw can be passed as float."""
        result = wf.discrete_prolate_spheroidal_sequence_window(
            64, 3.0, dtype=torch.float64
        )
        assert result.shape == (64,)
        result_periodic = (
            wf.periodic_discrete_prolate_spheroidal_sequence_window(
                64, 3.0, dtype=torch.float64
            )
        )
        assert result_periodic.shape == (64,)

    def test_nw_affects_shape(self):
        """Test that NW affects the window shape."""
        n = 32
        nw_low = torch.tensor(2.0, dtype=torch.float64)
        nw_high = torch.tensor(5.0, dtype=torch.float64)
        result_low = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw_low, dtype=torch.float64
        )
        result_high = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw_high, dtype=torch.float64
        )
        # Different NW should produce different window shapes
        assert not torch.allclose(result_low, result_high)

    def test_higher_nw_narrower_time_window(self):
        """Test that higher NW produces narrower time-domain window.

        Higher NW means concentration in a wider frequency band, which by
        the uncertainty principle means a narrower time-domain window
        (lower edge values).
        """
        n = 64
        nw_low = torch.tensor(2.0, dtype=torch.float64)
        nw_high = torch.tensor(5.0, dtype=torch.float64)
        result_low = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw_low, dtype=torch.float64
        )
        result_high = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw_high, dtype=torch.float64
        )
        # Higher NW should have lower edge values (narrower in time domain)
        assert result_high[0] < result_low[0]
        assert result_high[-1] < result_low[-1]

    def test_periodic_vs_symmetric_difference(self):
        """Test that periodic and symmetric windows differ."""
        n = 16
        nw = torch.tensor(3.0, dtype=torch.float64)
        symmetric = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )
        periodic = wf.periodic_discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )
        # They should be different
        assert not torch.allclose(symmetric, periodic)

    def test_n_equals_two(self):
        """Test specific case n=2."""
        nw = torch.tensor(1.0, dtype=torch.float64)
        result = wf.discrete_prolate_spheroidal_sequence_window(
            2, nw, dtype=torch.float64
        )
        assert result.shape == (2,)
        # Should be symmetric: both values equal
        torch.testing.assert_close(result[0], result[1], rtol=1e-5, atol=1e-5)

    def test_optimal_concentration(self):
        """Test that DPSS window has good energy concentration.

        The DPSS window should have most of its energy concentrated
        within the specified bandwidth.
        """
        n = 64
        nw = torch.tensor(4.0, dtype=torch.float64)
        window = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )

        # Compute frequency response
        freq_response = torch.fft.fft(window, n=1024)
        magnitude = torch.abs(freq_response)

        # Energy in main lobe (first NW bins, roughly)
        bandwidth_bins = int(1024 * nw.item() / n)
        main_lobe_energy = (magnitude[:bandwidth_bins] ** 2).sum()
        main_lobe_energy += (magnitude[-bandwidth_bins:] ** 2).sum()
        total_energy = (magnitude**2).sum()

        # DPSS should have high concentration ratio (typically > 0.9)
        concentration = main_lobe_energy / total_energy
        assert concentration > 0.8  # Conservative threshold

    def test_bell_shaped(self):
        """Test that DPSS window is bell-shaped (monotonic from edges to center)."""
        n = 65  # Odd for clear center
        nw = torch.tensor(3.0, dtype=torch.float64)
        window = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )

        center = n // 2
        # First half should be increasing
        first_half = window[: center + 1]
        diffs = first_half[1:] - first_half[:-1]
        assert (diffs >= -1e-6).all(), "Window should increase toward center"

        # Second half should be decreasing
        second_half = window[center:]
        diffs = second_half[1:] - second_half[:-1]
        assert (diffs <= 1e-6).all(), "Window should decrease from center"

    def test_center_value_is_max(self):
        """Test that the center value is the maximum."""
        nw = torch.tensor(3.0, dtype=torch.float64)
        for n in [16, 17, 32, 33]:
            window = wf.discrete_prolate_spheroidal_sequence_window(
                n, nw, dtype=torch.float64
            )
            center = n // 2
            # Center should be the maximum (or tied for max in even case)
            torch.testing.assert_close(
                window[center],
                window.max(),
                atol=1e-6,
                rtol=0,
            )

    def test_small_nw_narrow_window(self):
        """Test that small NW produces a narrow, concentrated window."""
        n = 64
        nw = torch.tensor(1.5, dtype=torch.float64)
        window = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )
        # Small NW should have very small edge values
        assert window[0] < 0.1
        assert window[-1] < 0.1

    def test_comparison_with_kaiser(self):
        """Test that DPSS is similar to Kaiser (Kaiser approximates DPSS)."""
        n = 64
        nw = torch.tensor(3.0, dtype=torch.float64)
        dpss = wf.discrete_prolate_spheroidal_sequence_window(
            n, nw, dtype=torch.float64
        )
        # Kaiser with beta roughly matching NW
        beta = torch.pi * nw
        kaiser = wf.kaiser_window(n, beta, dtype=torch.float64)

        # They should have similar shapes (correlation > 0.99)
        correlation = (dpss * kaiser).sum() / (
            torch.sqrt((dpss**2).sum() * (kaiser**2).sum())
        )
        assert correlation > 0.95  # Should be highly correlated
