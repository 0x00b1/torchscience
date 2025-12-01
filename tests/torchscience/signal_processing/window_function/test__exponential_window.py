import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestExponentialWindow:
    """Tests for exponential_window and periodic_exponential_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.exponential_window(n, tau, dtype=torch.float64)
            expected = self._reference_exponential(
                n, tau.item(), periodic=False
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_exponential_window(
                n, tau, dtype=torch.float64
            )
            expected = self._reference_exponential(
                n, tau.item(), periodic=True
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_comparison_symmetric(self):
        """Compare with scipy.signal.windows.exponential (symmetric)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for tau_val in [1.0, 3.0, 5.0]:
                tau = torch.tensor(tau_val, dtype=torch.float64)
                result = wf.exponential_window(n, tau, dtype=torch.float64)
                expected = torch.tensor(
                    scipy_signal.windows.exponential(n, tau=tau_val, sym=True),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_scipy_comparison_periodic(self):
        """Compare with scipy.signal.windows.exponential (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        for n in [4, 16, 64]:
            for tau_val in [1.0, 3.0, 5.0]:
                tau = torch.tensor(tau_val, dtype=torch.float64)
                result = wf.periodic_exponential_window(
                    n, tau, dtype=torch.float64
                )
                expected = torch.tensor(
                    scipy_signal.windows.exponential(
                        n, tau=tau_val, sym=False
                    ),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    result, expected, rtol=1e-10, atol=1e-10
                )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        tau = torch.tensor(3.0)
        for n in [0, 1, 5, 100]:
            result = wf.exponential_window(n, tau)
            assert result.shape == (n,)
            result_periodic = wf.periodic_exponential_window(n, tau)
            assert result_periodic.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        tau = torch.tensor(3.0, dtype=dtype)
        result = wf.exponential_window(64, tau, dtype=dtype)
        assert result.dtype == dtype
        result_periodic = wf.periodic_exponential_window(64, tau, dtype=dtype)
        assert result_periodic.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        tau = torch.tensor(3.0)
        result = wf.exponential_window(0, tau)
        assert result.shape == (0,)
        result_periodic = wf.periodic_exponential_window(0, tau)
        assert result_periodic.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        result = wf.exponential_window(1, tau, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )
        result_periodic = wf.periodic_exponential_window(
            1, tau, dtype=torch.float64
        )
        torch.testing.assert_close(
            result_periodic, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric exponential window is symmetric."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.exponential_window(n, tau, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length symmetric window is 1.0."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.exponential_window(n, tau, dtype=torch.float64)
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_gradient_flow(self):
        """Test that gradients flow through tau parameter."""
        tau = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.exponential_window(32, tau, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert tau.grad is not None
        assert not torch.isnan(tau.grad)

    def test_gradient_flow_periodic(self):
        """Test gradient flow for periodic version."""
        tau = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_exponential_window(32, tau, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert tau.grad is not None
        assert not torch.isnan(tau.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        tau = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(t):
            return wf.exponential_window(32, t, dtype=torch.float64).sum()

        torch.autograd.gradcheck(func, (tau,), raise_exception=True)

    def test_gradcheck_periodic(self):
        """Test gradient correctness for periodic version."""
        tau = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(t):
            return wf.periodic_exponential_window(
                32, t, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (tau,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        tau = torch.tensor(3.0)
        with pytest.raises(ValueError):
            wf.exponential_window(-1, tau)

    def test_float_tau_input(self):
        """Test that tau can be passed as float."""
        result = wf.exponential_window(64, 3.0, dtype=torch.float64)
        assert result.shape == (64,)
        result_periodic = wf.periodic_exponential_window(
            64, 3.0, dtype=torch.float64
        )
        assert result_periodic.shape == (64,)

    def test_default_tau(self):
        """Test default tau value of 1.0."""
        result = wf.exponential_window(64, dtype=torch.float64)
        expected = wf.exponential_window(64, 1.0, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_tau_affects_decay(self):
        """Test that larger tau produces slower decay."""
        n = 64
        tau_fast = torch.tensor(1.0, dtype=torch.float64)
        tau_slow = torch.tensor(5.0, dtype=torch.float64)
        result_fast = wf.exponential_window(n, tau_fast, dtype=torch.float64)
        result_slow = wf.exponential_window(n, tau_slow, dtype=torch.float64)
        # Slower decay (larger tau) should have larger values at edges
        assert result_slow[0] > result_fast[0]
        assert result_slow[-1] > result_fast[-1]
        # Sum should be larger for slower decay
        assert result_slow.sum() > result_fast.sum()

    def test_values_positive(self):
        """Test that all values are positive."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.exponential_window(n, tau, dtype=torch.float64)
            assert (result > 0).all()

    def test_max_value_at_center(self):
        """Test that maximum value is at the center."""
        tau = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.exponential_window(n, tau, dtype=torch.float64)
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # For even n, max could be at center-1 or center
            assert abs(max_idx - expected_center) <= 1

    @staticmethod
    def _reference_exponential(
        n: int, tau: float, periodic: bool
    ) -> torch.Tensor:
        """Reference implementation of exponential window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        center = n / 2.0 if periodic else (n - 1) / 2.0

        k = torch.arange(n, dtype=torch.float64)
        return torch.exp(-torch.abs(k - center) / tau)
