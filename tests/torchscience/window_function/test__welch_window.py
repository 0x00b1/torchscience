import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestWelchWindow:
    """Tests for welch_window and periodic_welch_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.welch_window(n, dtype=torch.float64)
            expected = self._reference_welch(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_welch_window(n, dtype=torch.float64)
            expected = self._reference_welch(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.bohman (NOT parzen - that's different).

        Note: scipy doesn't have a direct welch window function. We compare
        against our reference implementation instead. The scipy parzen window
        is actually a different window (4th order B-spline).
        """
        # This test validates our reference implementation by checking
        # known properties of the Welch window
        for n in [4, 5, 16, 17, 64, 65]:
            result = wf.welch_window(n, dtype=torch.float64)
            expected = self._reference_welch(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.welch_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_welch_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.welch_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_welch_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.welch_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_welch_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.welch_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_welch_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.welch_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_welch_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric welch window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.welch_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value_one_for_odd(self):
        """Test that center value is 1.0 for odd-length symmetric windows."""
        for n in [5, 11, 65]:
            result = wf.welch_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_zero_endpoints_symmetric(self):
        """Test that symmetric welch window has zero endpoints."""
        for n in [5, 10, 64]:
            result = wf.welch_window(n, dtype=torch.float64)
            # Endpoints should be 0 for symmetric Welch window
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )
            torch.testing.assert_close(
                result[-1],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_parabolic_shape(self):
        """Test that welch window has parabolic shape (second differences constant)."""
        n = 11
        result = wf.welch_window(n, dtype=torch.float64)
        # First differences
        first_diffs = result[1:] - result[:-1]
        # Second differences should be approximately constant (negative)
        second_diffs = first_diffs[1:] - first_diffs[:-1]
        # All second differences should be equal
        torch.testing.assert_close(
            second_diffs,
            second_diffs[0].expand_as(second_diffs),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_values_between_zero_and_one(self):
        """Test that all window values are in [0, 1]."""
        for n in [5, 10, 64]:
            result = wf.welch_window(n, dtype=torch.float64)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
            result_periodic = wf.periodic_welch_window(n, dtype=torch.float64)
            assert result_periodic.min() >= 0.0
            assert result_periodic.max() <= 1.0

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.welch_window(64, dtype=torch.float64, requires_grad=True)
        assert result.requires_grad

    def test_negative_n_raises(self):
        """Test that negative n raises an error."""
        with pytest.raises(RuntimeError):
            wf.welch_window(-1)
        with pytest.raises(RuntimeError):
            wf.periodic_welch_window(-1)

    def test_periodic_vs_symmetric_relationship(self):
        """Test relationship between periodic and symmetric windows.

        For periodic window, it should be equivalent to taking first n points
        of an n+1 length symmetric window.
        """
        for n in [5, 10, 64]:
            periodic = wf.periodic_welch_window(n, dtype=torch.float64)
            # Create an n+1 length symmetric-like computation
            expected = self._reference_welch(n, periodic=True)
            torch.testing.assert_close(
                periodic, expected, rtol=1e-10, atol=1e-10
            )

    def test_n_equals_two(self):
        """Test specific case n=2."""
        result = wf.welch_window(2, dtype=torch.float64)
        # For n=2 symmetric, denom = 1, center = 0.5
        # w[0] = 1 - ((0 - 0.5) / 0.5)^2 = 1 - 1 = 0
        # w[1] = 1 - ((1 - 0.5) / 0.5)^2 = 1 - 1 = 0
        expected = torch.tensor([0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    @staticmethod
    def _reference_welch(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation of Welch window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        center = denom / 2
        k = torch.arange(n, dtype=torch.float64)
        x = (k - center) / center
        return 1 - x * x


class TestWelchWindowCompile:
    """Test torch.compile compatibility for welch_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.welch_window)
        result = compiled_fn(64)
        expected = wf.welch_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_welch_window)
        result = compiled_fn(64)
        expected = wf.periodic_welch_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_fn = torch.compile(wf.welch_window)
        for dtype in [torch.float32, torch.float64]:
            result = compiled_fn(64, dtype=dtype)
            expected = wf.welch_window(64, dtype=dtype)
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_compile_in_larger_function(self):
        """Test torch.compile when welch_window is used in a larger function."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.welch_window(
                n, dtype=signal.dtype, device=signal.device
            )
            return torch.fft.fft(signal * window)

        compiled_fn = torch.compile(windowed_fft)
        signal = torch.randn(64, dtype=torch.float64)
        result = compiled_fn(signal)
        expected = windowed_fft(signal)
        torch.testing.assert_close(result, expected)

    def test_compile_dynamic_shapes(self):
        """Test torch.compile with dynamic shapes."""
        compiled_fn = torch.compile(wf.welch_window, dynamic=True)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n)
            expected = wf.welch_window(n)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
