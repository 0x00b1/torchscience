import math

import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestTukeyWindow:
    """Tests for tukey_window and periodic_tukey_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(0.5, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.tukey_window(n, alpha, dtype=torch.float64)
            expected = self._reference_tukey(n, alpha.item(), periodic=False)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(0.5, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_tukey_window(n, alpha, dtype=torch.float64)
            expected = self._reference_tukey(n, alpha.item(), periodic=True)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.tukey (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for alpha_val in [0.0, 0.25, 0.5, 1.0]:
                alpha = torch.tensor(alpha_val, dtype=torch.float64)
                result = wf.tukey_window(n, alpha, dtype=torch.float64)
                expected = torch.tensor(
                    scipy_signal.windows.tukey(n, alpha_val, sym=True),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.tukey (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for alpha_val in [0.0, 0.25, 0.5, 1.0]:
                alpha = torch.tensor(alpha_val, dtype=torch.float64)
                result = wf.periodic_tukey_window(
                    n, alpha, dtype=torch.float64
                )
                expected = torch.tensor(
                    scipy_signal.windows.tukey(n, alpha_val, sym=False),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_alpha_zero_is_rectangular(self):
        """alpha=0 should produce rectangular window (all ones)."""
        alpha = torch.tensor(0.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.tukey_window(n, alpha, dtype=torch.float64)
            expected = torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_alpha_one_is_hann(self):
        """alpha=1 should produce Hann window."""
        alpha = torch.tensor(1.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.tukey_window(n, alpha, dtype=torch.float64)
            expected = wf.hann_window(n, dtype=torch.float64)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        alpha = torch.tensor(0.5)
        for n in [0, 1, 5, 100]:
            result = wf.tukey_window(n, alpha)
            assert result.shape == (n,)
            result_periodic = wf.periodic_tukey_window(n, alpha)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    )
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        alpha = torch.tensor(0.5, dtype=dtype)
        result = wf.tukey_window(64, alpha, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_tukey_window(64, alpha, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_meta_tensor(self):
        """Test meta tensor support."""
        alpha = torch.tensor(0.5, device="meta")
        result = wf.tukey_window(64, alpha, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (64,)
        result_periodic = wf.periodic_tukey_window(64, alpha, device="meta")
        assert result_periodic.device.type == "meta"
        assert result_periodic.shape == (64,)

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        alpha = torch.tensor(0.5)
        result = wf.tukey_window(0, alpha)
        assert result.shape == (0,)
        result_periodic = wf.periodic_tukey_window(0, alpha)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        alpha = torch.tensor(0.5, dtype=torch.float64)
        result = wf.tukey_window(1, alpha, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_tukey_window(
            1, alpha, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Tukey window is symmetric."""
        alpha = torch.tensor(0.5, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.tukey_window(n, alpha, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_flat_region(self):
        """Test that middle of window is flat for alpha < 1."""
        n = 64
        alpha = torch.tensor(0.25, dtype=torch.float64)
        result = wf.tukey_window(n, alpha, dtype=torch.float64)
        # Middle portion should be all 1s
        center = len(result) // 2
        flat_start = int(n * alpha / 2) + 1
        flat_end = n - flat_start
        for i in range(flat_start, flat_end):
            torch.testing.assert_close(
                result[i],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_requires_grad(self):
        """Test that requires_grad propagates."""
        alpha = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        result = wf.tukey_window(32, alpha, dtype=torch.float64)
        # Output should support backward pass
        loss = result.sum()
        loss.backward()
        # For alpha in the interior (0 < alpha < 1), gradient should be non-zero
        assert alpha.grad is not None

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        alpha = torch.tensor(0.5)
        with pytest.raises(RuntimeError):
            wf.tukey_window(-1, alpha)

    def test_float_alpha_input(self):
        """Test that alpha can be passed as float."""
        result = wf.tukey_window(64, 0.5, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_tukey_window(
            64, 0.5, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_alpha_affects_taper_width(self):
        """Test that larger alpha produces narrower flat region."""
        n = 64
        alpha_narrow = torch.tensor(0.25, dtype=torch.float64)
        alpha_wide = torch.tensor(0.75, dtype=torch.float64)
        result_narrow = wf.tukey_window(n, alpha_narrow, dtype=torch.float64)
        result_wide = wf.tukey_window(n, alpha_wide, dtype=torch.float64)
        # Wider alpha should have smaller values at positions near edges
        # (since taper region is larger)
        idx = n // 4  # quarter point
        assert result_wide[idx] < result_narrow[idx]

    @staticmethod
    def _reference_tukey(n: int, alpha: float, periodic: bool) -> torch.Tensor:
        """Reference implementation of Tukey window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n) if periodic else float(n - 1)
        if denom == 0:
            return torch.ones(1, dtype=torch.float64)

        # Clamp alpha
        if alpha <= 0:
            return torch.ones(n, dtype=torch.float64)

        result = torch.empty(n, dtype=torch.float64)
        width = alpha * denom / 2

        for i in range(n):
            x = float(i)
            if alpha >= 1:
                # Hann window
                result[i] = 0.5 * (1 - math.cos(2 * math.pi * x / denom))
            elif x < width:
                # Left taper
                result[i] = 0.5 * (1 - math.cos(math.pi * x / width))
            elif x <= denom - width:
                # Flat region
                result[i] = 1.0
            else:
                # Right taper
                result[i] = 0.5 * (1 - math.cos(math.pi * (denom - x) / width))

        return result


class TestTukeyWindowCompile:
    """Test torch.compile compatibility for tukey_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.tukey_window)
        alpha = torch.tensor(0.5, dtype=torch.float64)
        result = compiled_fn(64, alpha, dtype=torch.float64)
        expected = wf.tukey_window(64, alpha, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_tukey_window)
        alpha = torch.tensor(0.5, dtype=torch.float64)
        result = compiled_fn(64, alpha, dtype=torch.float64)
        expected = wf.periodic_tukey_window(64, alpha, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_different_alpha(self):
        """Test torch.compile with various alpha values."""
        compiled_fn = torch.compile(wf.tukey_window)
        for alpha_val in [0.0, 0.25, 0.5, 1.0]:
            alpha = torch.tensor(alpha_val, dtype=torch.float64)
            result = compiled_fn(64, alpha, dtype=torch.float64)
            expected = wf.tukey_window(64, alpha, dtype=torch.float64)
            torch.testing.assert_close(result, expected)

    def test_compile_in_larger_function(self):
        """Test torch.compile when tukey_window is used in a larger function."""

        def windowed_fft(
            signal: torch.Tensor, alpha: torch.Tensor
        ) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.tukey_window(
                n, alpha, dtype=signal.dtype, device=signal.device
            )
            return torch.fft.fft(signal * window)

        compiled_fn = torch.compile(windowed_fft)
        signal = torch.randn(64, dtype=torch.float64)
        alpha = torch.tensor(0.5, dtype=torch.float64)
        result = compiled_fn(signal, alpha)
        expected = windowed_fft(signal, alpha)
        torch.testing.assert_close(result, expected)

    def test_compile_dynamic_shapes(self):
        """Test torch.compile with dynamic shapes."""
        compiled_fn = torch.compile(wf.tukey_window, dynamic=True)
        alpha = torch.tensor(0.5, dtype=torch.float64)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n, alpha, dtype=torch.float64)
            expected = wf.tukey_window(n, alpha, dtype=torch.float64)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
