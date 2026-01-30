import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestBlackmanWindow:
    """Tests for blackman_window and periodic_blackman_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.blackman_window(n, dtype=torch.float64)
            expected = self._reference_blackman(n, periodic=False)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        for n in [1, 2, 5, 64, 128]:
            result = wf.periodic_blackman_window(n, dtype=torch.float64)
            expected = self._reference_blackman(n, periodic=True)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_symmetric(self):
        """Compare with torch.signal.windows.blackman (symmetric)."""
        for n in [4, 16, 64]:
            result = wf.blackman_window(n, dtype=torch.float64)
            expected = torch.signal.windows.blackman(
                n, sym=True, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_pytorch_comparison_periodic(self):
        """Compare with torch.signal.windows.blackman (periodic)."""
        for n in [4, 16, 64]:
            result = wf.periodic_blackman_window(n, dtype=torch.float64)
            expected = torch.signal.windows.blackman(
                n, sym=False, dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        for n in [0, 1, 5, 100]:
            result = wf.blackman_window(n)
            assert result.shape == (n,)
            result_periodic = wf.periodic_blackman_window(n)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_dtype_support(self, dtype):
        """Test all supported dtypes."""
        result = wf.blackman_window(64, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_blackman_window(64, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor for shape inference."""
        result = wf.blackman_window(64, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_blackman_window(64, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        result = wf.blackman_window(0)
        assert result.shape == (0,)
        result_periodic = wf.periodic_blackman_window(0)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        result = wf.blackman_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_blackman_window(1, dtype=torch.float64)
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Blackman window is symmetric."""
        for n in [5, 10, 11, 64]:
            result = wf.blackman_window(n, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_near_zero(self):
        """Test that symmetric Blackman window has near-zero endpoints."""
        # Blackman: 0.42 - 0.5 * cos(0) + 0.08 * cos(0) = 0.42 - 0.5 + 0.08 = 0.0
        for n in [5, 10, 64]:
            result = wf.blackman_window(n, dtype=torch.float64)
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
            result = wf.blackman_window(n, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_requires_grad(self):
        """Test requires_grad parameter."""
        result = wf.blackman_window(
            64, dtype=torch.float64, requires_grad=True
        )
        assert result.requires_grad

    @staticmethod
    def _reference_blackman(n: int, periodic: bool) -> torch.Tensor:
        """Reference implementation."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        denom = n if periodic else n - 1
        k = torch.arange(n, dtype=torch.float64)
        return (
            0.42
            - 0.5 * torch.cos(2 * math.pi * k / denom)
            + 0.08 * torch.cos(4 * math.pi * k / denom)
        )


class TestBlackmanWindowCompile:
    """Test torch.compile compatibility for blackman_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.blackman_window)
        result = compiled_fn(64)
        expected = wf.blackman_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_blackman_window)
        result = compiled_fn(64)
        expected = wf.periodic_blackman_window(64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_dtype(self):
        """Test torch.compile with explicit dtype."""
        compiled_fn = torch.compile(wf.blackman_window)
        for dtype in [torch.float32, torch.float64]:
            result = compiled_fn(64, dtype=dtype)
            expected = wf.blackman_window(64, dtype=dtype)
            torch.testing.assert_close(result, expected)
            assert result.dtype == dtype

    def test_compile_in_larger_function(self):
        """Test torch.compile when blackman_window is used in a larger function."""

        def windowed_fft(signal: torch.Tensor) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.blackman_window(
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
        compiled_fn = torch.compile(wf.blackman_window, dynamic=True)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n)
            expected = wf.blackman_window(n)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
