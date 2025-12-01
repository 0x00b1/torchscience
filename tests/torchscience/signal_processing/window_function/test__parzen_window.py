import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestParzenWindow:
    """Tests for parzen_window and periodic_parzen_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.parzen_window(n, dtype=torch.float64)
            expected = self._reference_parzen(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_parzen_window(n, dtype=torch.float64)
            expected = self._reference_parzen(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.parzen (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 5, 16, 17, 64, 65]:
            result = wf.parzen_window(n, dtype=torch.float64)
            expected = torch.from_numpy(
                scipy_signal.windows.parzen(n, sym=True)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.parzen (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 5, 16, 17, 64, 65]:
            result = wf.periodic_parzen_window(n, dtype=torch.float64)
            expected = torch.from_numpy(
                scipy_signal.windows.parzen(n, sym=False)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.parzen_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_parzen_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.parzen_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_parzen_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.parzen_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_parzen_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.parzen_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_parzen_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.parzen_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_parzen_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric parzen window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.parzen_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value_odd_symmetric(self):
        """Test that center value is 1.0 for odd-length symmetric windows."""
        for n in [5, 11, 65]:
            result = wf.parzen_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_max_value_is_center(self):
        """Test that max value is at center for both symmetric and periodic."""
        for n in [5, 10, 11, 64, 65]:
            sym_result = wf.parzen_window(n, dtype=torch.float64)
            per_result = wf.periodic_parzen_window(n, dtype=torch.float64)

            # Max should be at center
            sym_max_idx = torch.argmax(sym_result).item()
            per_max_idx = torch.argmax(per_result).item()

            # For symmetric windows with odd n, max is exactly at center
            if n % 2 == 1:
                assert sym_max_idx == n // 2
                assert per_max_idx == n // 2

    def test_piecewise_cubic_inner_region(self):
        """Test that inner region follows cubic polynomial."""
        n = 17  # Odd number for clear center
        result = wf.parzen_window(n, dtype=torch.float64)

        # For symmetric window, L = n = 17
        # center = (L-1)/2 = 8
        # quarter = (L-1)/4 = 4
        # Inner region: |k - 8| <= 4, so k in [4, 12]
        # The formula is 1 - 6*x^2 + 6*|x|^3 where x = (k - center) / (L/2)

        L = n
        center = (L - 1) / 2  # = 8
        half = L / 2  # = 8.5

        for k in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            x = (k - center) / half
            abs_x = abs(x)
            expected = 1 - 6 * x * x + 6 * abs_x**3
            torch.testing.assert_close(
                result[k],
                torch.tensor(expected, dtype=torch.float64),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_piecewise_cubic_outer_region(self):
        """Test that outer region follows cubic polynomial."""
        n = 17
        result = wf.parzen_window(n, dtype=torch.float64)

        # For symmetric window, L = n = 17
        # center = (L-1)/2 = 8
        # quarter = (L-1)/4 = 4
        # Outer region: |k - 8| > 4, so k in [0, 3] or [13, 16]
        # The formula is 2*(1 - |x|)^3 where x = (k - center) / (L/2)

        L = n
        center = (L - 1) / 2  # = 8
        half = L / 2  # = 8.5

        for k in [0, 1, 2, 3, 13, 14, 15, 16]:
            x = (k - center) / half
            abs_x = abs(x)
            expected = 2 * (1 - abs_x) ** 3
            torch.testing.assert_close(
                result[k],
                torch.tensor(expected, dtype=torch.float64),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_values_between_zero_and_one(self):
        """Test that all window values are in [0, 1]."""
        for n in [5, 10, 64]:
            result = wf.parzen_window(n, dtype=torch.float64)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
            result_periodic = wf.periodic_parzen_window(n, dtype=torch.float64)
            assert result_periodic.min() >= 0.0
            assert result_periodic.max() <= 1.0

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.parzen_window(64, dtype=torch.float64, requires_grad=True)
        assert result.requires_grad

    def test_negative_n_raises(self):
        """Test that negative n raises an error."""
        with pytest.raises(RuntimeError):
            wf.parzen_window(-1)
        with pytest.raises(RuntimeError):
            wf.periodic_parzen_window(-1)

    def test_periodic_vs_symmetric_relationship(self):
        """Test relationship between periodic and symmetric windows.

        For periodic window, it should be equivalent to taking first n points
        of an n+1 length symmetric window (using the same formula but with
        L = n + 1).
        """
        for n in [5, 10, 64]:
            periodic = wf.periodic_parzen_window(n, dtype=torch.float64)
            expected = self._reference_parzen(n, periodic=True)
            torch.testing.assert_close(
                periodic, expected, rtol=1e-10, atol=1e-10
            )

    def test_n_equals_two(self):
        """Test specific case n=2."""
        result = wf.parzen_window(2, dtype=torch.float64)
        # For n=2 symmetric, L=2, center = 0.5, half = 1, quarter = 0.25
        # k=0: x = (0 - 0.5) / 1 = -0.5, |x| = 0.5 > quarter
        #      w = 2*(1 - 0.5)^3 = 2*(0.5)^3 = 2*0.125 = 0.25
        # k=1: x = (1 - 0.5) / 1 = 0.5, |x| = 0.5 > quarter
        #      w = 2*(1 - 0.5)^3 = 0.25
        expected = torch.tensor([0.25, 0.25], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_endpoint_values(self):
        """Test that endpoints have small positive values (not zero)."""
        for n in [5, 10, 64]:
            result = wf.parzen_window(n, dtype=torch.float64)
            # Parzen window has very small but positive endpoints
            assert result[0] > 0, f"First endpoint should be > 0 for n={n}"
            assert result[-1] > 0, f"Last endpoint should be > 0 for n={n}"

    def test_differs_from_triangular(self):
        """Test that Parzen window differs from triangular window."""
        for n in [5, 10, 64]:
            parzen = wf.parzen_window(n, dtype=torch.float64)
            triangular = wf.triangular_window(n, dtype=torch.float64)
            # They should not be equal
            assert not torch.allclose(parzen, triangular), (
                f"Parzen and triangular should differ for n={n}"
            )

    def test_differs_from_welch(self):
        """Test that Parzen window differs from Welch window."""
        for n in [5, 10, 64]:
            parzen = wf.parzen_window(n, dtype=torch.float64)
            welch = wf.welch_window(n, dtype=torch.float64)
            # They should not be equal
            assert not torch.allclose(parzen, welch), (
                f"Parzen and Welch should differ for n={n}"
            )

    def test_small_window_sizes(self):
        """Test small window sizes (n=2, 3, 4, 5)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [2, 3, 4, 5]:
            result = wf.parzen_window(n, dtype=torch.float64)
            expected = torch.from_numpy(
                scipy_signal.windows.parzen(n, sym=True)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    @staticmethod
    def _reference_parzen(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation of Parzen window matching scipy."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        # For sym=True: L = n, for sym=False (periodic): L = n + 1
        L = (n + 1) if periodic else n

        # Position relative to center
        center = (L - 1) / 2.0
        k = torch.arange(n, dtype=torch.float64)
        pos = k - center
        abs_pos = torch.abs(pos)

        # Half-length for normalization
        half = L / 2.0
        quarter = (L - 1) / 4.0

        # Normalized positions
        x = pos / half
        abs_x = abs_pos / half

        # Piecewise formula
        inner_mask = abs_pos <= quarter
        outer_mask = ~inner_mask

        result = torch.zeros_like(k)
        result[inner_mask] = (
            1 - 6 * x[inner_mask] ** 2 + 6 * abs_x[inner_mask] ** 3
        )
        result[outer_mask] = 2 * (1 - abs_x[outer_mask]) ** 3

        return result
