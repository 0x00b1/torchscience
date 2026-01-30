import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestKaiserWindow:
    """Tests for kaiser_window and periodic_kaiser_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        beta = torch.tensor(6.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.kaiser_window(n, beta, dtype=torch.float64)
            expected = self._reference_kaiser(n, beta.item(), periodic=False)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        beta = torch.tensor(6.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_kaiser_window(n, beta, dtype=torch.float64)
            expected = self._reference_kaiser(n, beta.item(), periodic=True)
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.kaiser (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for beta_val in [0.0, 5.0, 6.0, 8.6]:
                beta = torch.tensor(beta_val, dtype=torch.float64)
                result = wf.kaiser_window(n, beta, dtype=torch.float64)
                expected = torch.tensor(
                    scipy_signal.windows.kaiser(n, beta_val, sym=True),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-8, atol=1e-8
                )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.kaiser (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for beta_val in [0.0, 5.0, 6.0, 8.6]:
                beta = torch.tensor(beta_val, dtype=torch.float64)
                result = wf.periodic_kaiser_window(
                    n, beta, dtype=torch.float64
                )
                expected = torch.tensor(
                    scipy_signal.windows.kaiser(n, beta_val, sym=False),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-8, atol=1e-8
                )

    def test_beta_zero_is_rectangular(self):
        """beta=0 should produce rectangular window (all ones)."""
        beta = torch.tensor(0.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.kaiser_window(n, beta, dtype=torch.float64)
            expected = torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        beta = torch.tensor(6.0)
        for n in [0, 1, 5, 100]:
            result = wf.kaiser_window(n, beta)
            assert result.shape == (n,)
            result_periodic = wf.periodic_kaiser_window(n, beta)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        beta = torch.tensor(6.0, dtype=dtype)
        result = wf.kaiser_window(64, beta, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_kaiser_window(64, beta, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        beta = torch.tensor(6.0)
        result = wf.kaiser_window(0, beta)
        assert result.shape == (0,)
        result_periodic = wf.periodic_kaiser_window(0, beta)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        beta = torch.tensor(6.0, dtype=torch.float64)
        result = wf.kaiser_window(1, beta, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_kaiser_window(
            1, beta, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Kaiser window is symmetric."""
        beta = torch.tensor(6.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.kaiser_window(n, beta, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        beta = torch.tensor(6.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.kaiser_window(n, beta, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_gradient_flow(self):
        """Test that gradients flow through beta parameter."""
        beta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
        result = wf.kaiser_window(32, beta, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert beta.grad is not None
        assert not torch.isnan(beta.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        beta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_kaiser_window(32, beta, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert beta.grad is not None
        assert not torch.isnan(beta.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        beta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)

        def func(b):
            return wf.kaiser_window(32, b, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (beta,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        beta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)

        def func(b):
            return wf.periodic_kaiser_window(32, b, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (beta,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        beta = torch.tensor(6.0)
        with pytest.raises(ValueError):
            wf.kaiser_window(-1, beta)

    def test_float_beta_input(self):
        """Test that beta can be passed as float."""
        result = wf.kaiser_window(64, 6.0, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_kaiser_window(
            64, 6.0, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_beta_affects_shape(self):
        """Test that larger beta produces narrower window."""
        n = 64
        beta_small = torch.tensor(4.0, dtype=torch.float64)
        beta_large = torch.tensor(10.0, dtype=torch.float64)
        result_small = wf.kaiser_window(n, beta_small, dtype=torch.float64)
        result_large = wf.kaiser_window(n, beta_large, dtype=torch.float64)
        # Larger beta should have smaller values at edges
        assert result_large[0] < result_small[0]
        assert result_large[-1] < result_small[-1]

    @staticmethod
    def _reference_kaiser(n: int, beta: float, periodic: bool) -> torch.Tensor:
        """Reference implementation of Kaiser window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n) if periodic else float(n - 1)
        center = denom / 2.0

        k = torch.arange(n, dtype=torch.float64)
        x = (k - center) / center

        # Argument to I0: beta * sqrt(1 - x^2)
        arg = beta * torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

        return torch.i0(arg) / torch.i0(
            torch.tensor(beta, dtype=torch.float64)
        )


class TestKaiserWindowCompile:
    """Test torch.compile compatibility for kaiser_window."""

    def test_compile_basic(self):
        """Test basic torch.compile compatibility."""
        compiled_fn = torch.compile(wf.kaiser_window)
        beta = torch.tensor(6.0, dtype=torch.float64)
        result = compiled_fn(64, beta, dtype=torch.float64)
        expected = wf.kaiser_window(64, beta, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_compile_periodic(self):
        """Test torch.compile for periodic variant."""
        compiled_fn = torch.compile(wf.periodic_kaiser_window)
        beta = torch.tensor(6.0, dtype=torch.float64)
        result = compiled_fn(64, beta, dtype=torch.float64)
        expected = wf.periodic_kaiser_window(64, beta, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_compile_with_different_beta(self):
        """Test torch.compile with various beta values."""
        compiled_fn = torch.compile(wf.kaiser_window)
        for beta_val in [0.0, 4.0, 8.6, 14.0]:
            beta = torch.tensor(beta_val, dtype=torch.float64)
            result = compiled_fn(64, beta, dtype=torch.float64)
            expected = wf.kaiser_window(64, beta, dtype=torch.float64)
            torch.testing.assert_close(result, expected)

    def test_compile_in_larger_function(self):
        """Test torch.compile when kaiser_window is used in a larger function."""

        def windowed_fft(
            signal: torch.Tensor, beta: torch.Tensor
        ) -> torch.Tensor:
            n = signal.shape[-1]
            window = wf.kaiser_window(
                n, beta, dtype=signal.dtype, device=signal.device
            )
            return torch.fft.fft(signal * window)

        compiled_fn = torch.compile(windowed_fft)
        signal = torch.randn(64, dtype=torch.float64)
        beta = torch.tensor(6.0, dtype=torch.float64)
        result = compiled_fn(signal, beta)
        expected = windowed_fft(signal, beta)
        torch.testing.assert_close(result, expected)

    def test_compile_dynamic_shapes(self):
        """Test torch.compile with dynamic shapes."""
        compiled_fn = torch.compile(wf.kaiser_window, dynamic=True)
        beta = torch.tensor(6.0, dtype=torch.float64)
        for n in [16, 32, 64, 128]:
            result = compiled_fn(n, beta, dtype=torch.float64)
            expected = wf.kaiser_window(n, beta, dtype=torch.float64)
            torch.testing.assert_close(result, expected)
            assert result.shape == (n,)
