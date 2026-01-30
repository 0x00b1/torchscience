import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestTriangularWindow:
    """Tests for triangular_window and periodic_triangular_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.triangular_window(n, dtype=torch.float64)
            expected = self._reference_triangular(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_triangular_window(n, dtype=torch.float64)
            expected = self._reference_triangular(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.triang (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 5, 16, 17, 64, 65]:
            result = wf.triangular_window(n, dtype=torch.float64)
            expected = torch.from_numpy(
                scipy_signal.windows.triang(n, sym=True)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.triang (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 5, 16, 17, 64, 65]:
            result = wf.periodic_triangular_window(n, dtype=torch.float64)
            expected = torch.from_numpy(
                scipy_signal.windows.triang(n, sym=False)
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.triangular_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_triangular_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.triangular_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_triangular_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.triangular_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_triangular_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.triangular_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_triangular_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.triangular_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_triangular_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric triangular window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.triangular_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_nonzero_endpoints(self):
        """Test that triangular window has non-zero endpoints (unlike Bartlett)."""
        for n in [5, 10, 64]:
            result = wf.triangular_window(n, dtype=torch.float64)
            # Endpoints should be > 0 (not exactly 0 like Bartlett)
            assert result[0] > 0, f"First endpoint should be > 0 for n={n}"
            assert result[-1] > 0, f"Last endpoint should be > 0 for n={n}"

    def test_center_value_odd(self):
        """Test that center of odd-length symmetric window is 1.0."""
        for n in [5, 11, 65]:
            result = wf.triangular_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_max_value_even(self):
        """Test that max value is < 1.0 for even-length symmetric windows."""
        for n in [4, 10, 64]:
            result = wf.triangular_window(n, dtype=torch.float64)
            # For even n, the max value should be less than 1.0
            assert result.max() < 1.0, (
                f"Max value should be < 1.0 for even n={n}"
            )

    def test_triangular_shape(self):
        """Test that triangular window has triangular shape (linear increase/decrease)."""
        n = 11
        result = wf.triangular_window(n, dtype=torch.float64)
        # First differences should be constant (positive in first half, negative in second)
        center = n // 2
        first_half_diffs = result[1 : center + 1] - result[0:center]
        second_half_diffs = result[center + 1 :] - result[center:-1]
        # All first differences in first half should be equal
        torch.testing.assert_close(
            first_half_diffs,
            first_half_diffs[0].expand_as(first_half_diffs),
            rtol=1e-10,
            atol=1e-10,
        )
        # All first differences in second half should be equal (and negative of first)
        torch.testing.assert_close(
            second_half_diffs,
            -first_half_diffs[0].expand_as(second_half_diffs),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_differs_from_bartlett(self):
        """Test that triangular window differs from Bartlett window."""
        for n in [5, 10, 64]:
            triangular = wf.triangular_window(n, dtype=torch.float64)
            bartlett = wf.bartlett_window(n, dtype=torch.float64)
            # They should not be equal
            assert not torch.allclose(triangular, bartlett), (
                f"Triangular and Bartlett should differ for n={n}"
            )
            # Triangular has non-zero endpoints while Bartlett has zero endpoints
            assert triangular[0] > bartlett[0]
            assert triangular[-1] > bartlett[-1]

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.triangular_window(
            64, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

    def test_negative_n_raises(self):
        """Test that negative n raises an error."""
        with pytest.raises(RuntimeError):
            wf.triangular_window(-1)
        with pytest.raises(RuntimeError):
            wf.periodic_triangular_window(-1)

    @staticmethod
    def _reference_triangular(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation matching scipy.signal.windows.triang."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        # For periodic window, compute as if length n+1 symmetric window
        L = n + 1 if periodic else n

        result = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            if L % 2 == 1:
                # Odd effective length
                half_L = (L + 1) / 2
                if i < L // 2 + 1:
                    result[i] = (i + 1) / half_L
                else:
                    result[i] = (L - i) / half_L
            else:
                # Even effective length
                if i < L // 2:
                    result[i] = (2 * i + 1) / L
                else:
                    result[i] = (2 * (L - i) - 1) / L

        return result


class TestTriangularWindowCompile:
    """Test torch.compile compatibility for triangular_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.triangular_window)
        result = compiled_fn(64)
        expected = wf.triangular_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_triangular_window)
        result = compiled_fn(64)
        expected = wf.periodic_triangular_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_fn = torch.compile(wf.triangular_window)
        for dtype in [torch.float32, torch.float64]:
            result = compiled_fn(64, dtype=dtype)
            expected = wf.triangular_window(64, dtype=dtype)
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_compile_in_larger_function(self):
        """Test torch.compile when triangular_window is used in FFT."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.triangular_window(
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
        compiled_fn = torch.compile(wf.triangular_window, dynamic=True)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n)
            expected = wf.triangular_window(n)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
