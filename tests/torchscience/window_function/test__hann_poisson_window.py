import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestHannPoissonWindow:
    """Tests for hann_poisson_window and periodic_hann_poisson_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
            expected = self._reference_hann_poisson(
                n, alpha.item(), periodic=False
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_hann_poisson_window(
                n, alpha, dtype=torch.float64
            )
            expected = self._reference_hann_poisson(
                n, alpha.item(), periodic=True
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        alpha = torch.tensor(2.0)
        for n in [0, 1, 5, 100]:
            result = wf.hann_poisson_window(n, alpha)
            assert result.shape == (n,)
            result_periodic = wf.periodic_hann_poisson_window(n, alpha)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        alpha = torch.tensor(2.0, dtype=dtype)
        result = wf.hann_poisson_window(64, alpha, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_hann_poisson_window(
            64, alpha, dtype=dtype
        )
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        alpha = torch.tensor(2.0)
        result = wf.hann_poisson_window(0, alpha)
        assert result.shape == (0,)
        result_periodic = wf.periodic_hann_poisson_window(0, alpha)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        result = wf.hann_poisson_window(1, alpha, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_hann_poisson_window(
            1, alpha, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric hann_poisson window is symmetric."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that endpoints are zero (from Hann component)."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
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

    def test_gradient_flow(self):
        """Test that gradients flow through alpha parameter."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        result = wf.hann_poisson_window(32, alpha, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert not torch.isnan(alpha.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_hann_poisson_window(
            32, alpha, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert alpha.grad is not None
        assert not torch.isnan(alpha.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def func(a):
            return wf.hann_poisson_window(32, a, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (alpha,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        alpha = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

        def func(a):
            return wf.periodic_hann_poisson_window(
                32, a, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (alpha,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        alpha = torch.tensor(2.0)
        with pytest.raises(ValueError):
            wf.hann_poisson_window(-1, alpha)

    def test_float_alpha_input(self):
        """Test that alpha can be passed as float."""
        result = wf.hann_poisson_window(64, 2.0, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_hann_poisson_window(
            64, 2.0, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_default_alpha(self):
        """Test default alpha value of 1.0."""
        result = wf.hann_poisson_window(64, dtype=torch.float64)
        expected = wf.hann_poisson_window(64, 1.0, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_alpha_zero_equals_hann(self):
        """Test that alpha=0 reduces to Hann window."""
        n = 64
        alpha = torch.tensor(0.0, dtype=torch.float64)
        result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
        # When alpha=0, poisson term is exp(0) = 1, so we get pure Hann
        expected = wf.hann_window(n, dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_alpha_affects_decay(self):
        """Test that larger alpha produces faster decay."""
        n = 64
        alpha_slow = torch.tensor(0.5, dtype=torch.float64)
        alpha_fast = torch.tensor(3.0, dtype=torch.float64)
        result_slow = wf.hann_poisson_window(
            n, alpha_slow, dtype=torch.float64
        )
        result_fast = wf.hann_poisson_window(
            n, alpha_fast, dtype=torch.float64
        )
        # Faster decay (larger alpha) should have smaller sum
        assert result_slow.sum() > result_fast.sum()

    def test_values_non_negative(self):
        """Test that all values are non-negative."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
            assert (result >= 0).all()

    def test_max_value_at_center(self):
        """Test that maximum value is at the center."""
        alpha = torch.tensor(2.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # For even n, max could be at center-1 or center
            assert abs(max_idx - expected_center) <= 1

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.hann (when alpha=0)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            alpha = torch.tensor(0.0, dtype=torch.float64)
            result = wf.hann_poisson_window(n, alpha, dtype=torch.float64)
            expected = torch.tensor(
                scipy_signal.windows.hann(n, sym=True), dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.hann periodic (when alpha=0)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            alpha = torch.tensor(0.0, dtype=torch.float64)
            result = wf.periodic_hann_poisson_window(
                n, alpha, dtype=torch.float64
            )
            expected = torch.tensor(
                scipy_signal.windows.hann(n, sym=False), dtype=torch.float64
            )
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    @staticmethod
    def _reference_hann_poisson(
        n: int, alpha: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation of Hann-Poisson window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n) if periodic else float(n - 1)

        k = torch.arange(n, dtype=torch.float64)

        # Hann component
        hann = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * k / denom))

        # Poisson component
        poisson = torch.exp(-alpha * torch.abs(denom - 2.0 * k) / denom)

        return hann * poisson
