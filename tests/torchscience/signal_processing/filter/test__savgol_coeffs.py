"""Tests for Savitzky-Golay filter coefficient computation."""

import pytest
import torch
from scipy.signal import savgol_coeffs as scipy_savgol_coeffs

from torchscience.signal_processing.filter import savgol_coeffs


class TestSavgolCoeffs:
    """Test savgol_coeffs function."""

    @pytest.mark.parametrize("window_length", [5, 7, 11, 21])
    @pytest.mark.parametrize("polyorder", [2, 3, 4])
    def test_matches_scipy(self, window_length: int, polyorder: int) -> None:
        """Coefficients should match scipy.signal.savgol_coeffs."""
        if polyorder >= window_length:
            pytest.skip("polyorder must be less than window_length")

        coeffs = savgol_coeffs(window_length, polyorder)
        coeffs_scipy = scipy_savgol_coeffs(window_length, polyorder)

        torch.testing.assert_close(
            coeffs, torch.from_numpy(coeffs_scipy), rtol=1e-10, atol=1e-12
        )

    @pytest.mark.parametrize("deriv", [0, 1, 2])
    def test_derivative_matches_scipy(self, deriv: int) -> None:
        """Derivative coefficients should match scipy."""
        window_length = 11
        polyorder = 4

        coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv)
        coeffs_scipy = scipy_savgol_coeffs(
            window_length, polyorder, deriv=deriv
        )

        torch.testing.assert_close(
            coeffs, torch.from_numpy(coeffs_scipy), rtol=1e-10, atol=1e-12
        )

    def test_delta_matches_scipy(self) -> None:
        """Delta parameter should match scipy."""
        window_length = 11
        polyorder = 3
        delta = 0.5

        # Test smoothing with delta
        coeffs = savgol_coeffs(window_length, polyorder, delta=delta)
        coeffs_scipy = scipy_savgol_coeffs(
            window_length, polyorder, delta=delta
        )
        torch.testing.assert_close(
            coeffs, torch.from_numpy(coeffs_scipy), rtol=1e-10, atol=1e-12
        )

        # Test derivative with delta
        coeffs_deriv = savgol_coeffs(
            window_length, polyorder, deriv=1, delta=delta
        )
        coeffs_scipy_deriv = scipy_savgol_coeffs(
            window_length, polyorder, deriv=1, delta=delta
        )
        torch.testing.assert_close(
            coeffs_deriv,
            torch.from_numpy(coeffs_scipy_deriv),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_pos_matches_scipy(self) -> None:
        """Position parameter should match scipy."""
        window_length = 11
        polyorder = 3

        for pos in [0, 3, 5, 10]:
            coeffs = savgol_coeffs(window_length, polyorder, pos=pos)
            coeffs_scipy = scipy_savgol_coeffs(
                window_length, polyorder, pos=pos
            )
            torch.testing.assert_close(
                coeffs, torch.from_numpy(coeffs_scipy), rtol=1e-10, atol=1e-12
            )

    def test_dtype_float32(self) -> None:
        """Should respect float32 dtype parameter."""
        coeffs = savgol_coeffs(11, 3, dtype=torch.float32)
        assert coeffs.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Should respect float64 dtype parameter."""
        coeffs = savgol_coeffs(11, 3, dtype=torch.float64)
        assert coeffs.dtype == torch.float64

    def test_default_dtype_is_float64(self) -> None:
        """Default dtype should be float64."""
        coeffs = savgol_coeffs(11, 3)
        assert coeffs.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Should respect device parameter."""
        coeffs = savgol_coeffs(11, 3, device=torch.device("cpu"))
        assert coeffs.device.type == "cpu"

    def test_invalid_window_length_even(self) -> None:
        """Should raise for even window_length."""
        with pytest.raises(
            ValueError, match="window_length must be a positive odd"
        ):
            savgol_coeffs(4, 2)

    def test_invalid_window_length_zero(self) -> None:
        """Should raise for zero window_length."""
        with pytest.raises(
            ValueError, match="window_length must be a positive odd"
        ):
            savgol_coeffs(0, 2)

    def test_invalid_window_length_negative(self) -> None:
        """Should raise for negative window_length."""
        with pytest.raises(
            ValueError, match="window_length must be a positive odd"
        ):
            savgol_coeffs(-5, 2)

    def test_invalid_polyorder_too_large(self) -> None:
        """Should raise for polyorder >= window_length."""
        with pytest.raises(ValueError, match="polyorder must be less than"):
            savgol_coeffs(5, 5)

    def test_invalid_polyorder_equal_to_window(self) -> None:
        """Should raise when polyorder equals window_length."""
        with pytest.raises(ValueError, match="polyorder must be less than"):
            savgol_coeffs(7, 7)

    def test_invalid_polyorder_negative(self) -> None:
        """Should raise for negative polyorder."""
        with pytest.raises(ValueError, match="polyorder must be non-negative"):
            savgol_coeffs(5, -1)

    def test_invalid_deriv_negative(self) -> None:
        """Should raise for negative deriv."""
        with pytest.raises(ValueError, match="deriv must be non-negative"):
            savgol_coeffs(5, 2, deriv=-1)

    def test_deriv_greater_than_polyorder(self) -> None:
        """Should return zeros when deriv > polyorder."""
        coeffs = savgol_coeffs(5, 2, deriv=3)
        assert torch.all(coeffs == 0)
        assert coeffs.shape == (5,)

    def test_coefficients_sum_to_one_for_smoothing(self) -> None:
        """Smoothing coefficients should sum to approximately 1."""
        coeffs = savgol_coeffs(11, 3, deriv=0)
        torch.testing.assert_close(
            coeffs.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_coefficients_sum_to_zero_for_first_derivative(self) -> None:
        """First derivative coefficients should sum to approximately 0."""
        coeffs = savgol_coeffs(11, 3, deriv=1)
        torch.testing.assert_close(
            coeffs.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_smoothing_coefficients_symmetry(self) -> None:
        """Smoothing coefficients should be symmetric around center."""
        coeffs = savgol_coeffs(11, 3, deriv=0)
        # Check symmetry
        for i in range(len(coeffs) // 2):
            torch.testing.assert_close(
                coeffs[i], coeffs[-1 - i], rtol=1e-10, atol=1e-12
            )

    def test_first_derivative_coefficients_antisymmetry(self) -> None:
        """First derivative coefficients should be antisymmetric."""
        coeffs = savgol_coeffs(11, 3, deriv=1)
        # Check antisymmetry
        for i in range(len(coeffs) // 2):
            torch.testing.assert_close(
                coeffs[i], -coeffs[-1 - i], rtol=1e-10, atol=1e-12
            )

    def test_output_shape(self) -> None:
        """Output shape should equal window_length."""
        for window_length in [5, 7, 11, 21, 51]:
            coeffs = savgol_coeffs(window_length, 3)
            assert coeffs.shape == (window_length,)

    def test_polyorder_zero(self) -> None:
        """Polyorder 0 should give constant weights (moving average)."""
        window_length = 5
        coeffs = savgol_coeffs(window_length, 0)
        expected = torch.full(
            (window_length,), 1.0 / window_length, dtype=torch.float64
        )
        torch.testing.assert_close(coeffs, expected, rtol=1e-10, atol=1e-12)

    def test_second_derivative_matches_scipy(self) -> None:
        """Second derivative coefficients should match scipy."""
        window_length = 9
        polyorder = 4
        deriv = 2

        coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv)
        coeffs_scipy = scipy_savgol_coeffs(
            window_length, polyorder, deriv=deriv
        )

        torch.testing.assert_close(
            coeffs, torch.from_numpy(coeffs_scipy), rtol=1e-10, atol=1e-12
        )
