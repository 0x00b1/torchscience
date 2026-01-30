import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestBartlettWindow:
    """Tests for bartlett_window and periodic_bartlett_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.bartlett_window(n, dtype=torch.float64)
            expected = self._reference_bartlett(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_bartlett_window(n, dtype=torch.float64)
            expected = self._reference_bartlett(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_symmetric(self):
        """Compare with torch.signal.windows.bartlett (symmetric)."""
        for n in [4, 16, 64]:
            result = wf.bartlett_window(n, dtype=torch.float64)
            expected = torch.signal.windows.bartlett(
                n, sym=True, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_periodic(self):
        """Compare with torch.signal.windows.bartlett (periodic)."""
        for n in [4, 16, 64]:
            result = wf.periodic_bartlett_window(n, dtype=torch.float64)
            expected = torch.signal.windows.bartlett(
                n, sym=False, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.bartlett_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_bartlett_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.bartlett_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_bartlett_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.bartlett_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_bartlett_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.bartlett_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_bartlett_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.bartlett_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_bartlett_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Bartlett window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.bartlett_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that symmetric Bartlett window has zero endpoints."""
        for n in [5, 10, 64]:
            result = wf.bartlett_window(n, dtype=torch.float64)
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

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        for n in [5, 11, 65]:
            result = wf.bartlett_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_triangular_shape(self):
        """Test that Bartlett window has triangular shape (linear increase/decrease)."""
        n = 11
        result = wf.bartlett_window(n, dtype=torch.float64)
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

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.bartlett_window(
            64, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

    @staticmethod
    def _reference_bartlett(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        return 1 - torch.abs(k - denom / 2) / (denom / 2)


class TestBartlettWindowCompile:
    """Test torch.compile compatibility for bartlett_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.bartlett_window)
        result = compiled_fn(64)
        expected = wf.bartlett_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_bartlett_window)
        result = compiled_fn(64)
        expected = wf.periodic_bartlett_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_fn = torch.compile(wf.bartlett_window)
        for dtype in [torch.float32, torch.float64]:
            result = compiled_fn(64, dtype=dtype)
            expected = wf.bartlett_window(64, dtype=dtype)
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_compile_in_larger_function(self):
        """Test torch.compile when bartlett_window is used in a larger function."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.bartlett_window(
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
        compiled_fn = torch.compile(wf.bartlett_window, dynamic=True)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n)
            expected = wf.bartlett_window(n)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
