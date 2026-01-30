import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestFlatTopWindow:
    """Tests for flat_top_window and periodic_flat_top_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.flat_top_window(n, dtype=torch.float64)
            expected = self._reference_flat_top(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_flat_top_window(n, dtype=torch.float64)
            expected = self._reference_flat_top(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison(self):
        """Compare against scipy.signal.windows.flattop if available."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import flattop

        for n in [5, 10, 64, 128]:
            # scipy flattop is symmetric by default
            result = wf.flat_top_window(n, dtype=torch.float64)
            expected = torch.tensor(flattop(n), dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_periodic_comparison(self):
        """Compare against scipy with sym=False if available."""
        pytest.importorskip("scipy")
        from scipy.signal.windows import flattop

        for n in [5, 10, 64, 128]:
            result = wf.periodic_flat_top_window(n, dtype=torch.float64)
            expected = torch.tensor(flattop(n, sym=False), dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.flat_top_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_flat_top_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.flat_top_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_flat_top_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.flat_top_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_flat_top_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.flat_top_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_flat_top_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.flat_top_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_flat_top_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric flat-top window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.flat_top_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is approximately 1.0.

        Note: The flat-top coefficients don't sum to exactly 1.0 due to truncation,
        so the center value is approximately 1.000000003. This matches scipy behavior.
        """
        for n in [5, 11, 65]:
            result = wf.flat_top_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-8,
                rtol=0,
            )

    def test_coefficients_sum_to_one(self):
        """Verify that flat-top coefficients sum to 1 (at center)."""
        # Flat-top coefficients
        a0, a1, a2, a3, a4 = (
            0.21557895,
            0.41663158,
            0.277263158,
            0.083578947,
            0.006947368,
        )
        # At center, cos(pi) = -1, cos(2*pi) = 1, etc.
        # window = a0 + a1 + a2 + a3 + a4 = 1
        total = a0 + a1 + a2 + a3 + a4
        assert abs(total - 1.0) < 1e-7

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.flat_top_window(
            64, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

    def test_can_have_negative_values(self):
        """Test that flat-top window can have negative values.

        Unlike many other windows, the flat-top window is not bounded
        to [0, 1] and can have negative values at the edges.
        """
        for n in [10, 64, 128]:
            result = wf.flat_top_window(n, dtype=torch.float64)
            # Flat-top window should have negative values
            assert (result < 0.0).any(), f"Expected negative values for n={n}"

            result_periodic = wf.periodic_flat_top_window(
                n, dtype=torch.float64
            )
            assert (result_periodic < 0.0).any(), (
                f"Expected negative values for periodic n={n}"
            )

    def test_max_value_is_one(self):
        """Test that maximum value of odd-length flat-top window is approximately 1.0.

        Note: Due to coefficient truncation, max is approximately 1.000000003.
        For even-length windows, the maximum is not exactly at the center
        and may be less than 1.0.
        """
        for n in [5, 11, 65]:  # Only odd-length windows have max=1 at center
            result = wf.flat_top_window(n, dtype=torch.float64)
            torch.testing.assert_close(
                result.max(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-8,
                rtol=0,
            )

    @staticmethod
    def _reference_flat_top(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation using the same coefficients as scipy."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        # Flat-top coefficients (same as scipy.signal.windows.flattop)
        a0, a1, a2, a3, a4 = (
            0.21557895,
            0.41663158,
            0.277263158,
            0.083578947,
            0.006947368,
        )
        return (
            a0
            - a1 * torch.cos(2 * math.pi * k / denom)
            + a2 * torch.cos(4 * math.pi * k / denom)
            - a3 * torch.cos(6 * math.pi * k / denom)
            + a4 * torch.cos(8 * math.pi * k / denom)
        )


class TestFlatTopWindowCompile:
    """Test torch.compile compatibility for flat_top_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.flat_top_window)
        result = compiled_fn(64)
        expected = wf.flat_top_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_flat_top_window)
        result = compiled_fn(64)
        expected = wf.periodic_flat_top_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_fn = torch.compile(wf.flat_top_window)
        for dtype in [torch.float32, torch.float64]:
            result = compiled_fn(64, dtype=dtype)
            expected = wf.flat_top_window(64, dtype=dtype)
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_compile_in_larger_function(self):
        """Test torch.compile when flat_top_window is used in a larger function."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.flat_top_window(
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
        compiled_fn = torch.compile(wf.flat_top_window, dynamic=True)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n)
            expected = wf.flat_top_window(n)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
