import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestGeneralizedNormalWindow:
    """Tests for generalized_normal_window."""

    def test_reference(self):
        """Compare against reference implementation."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            expected = self._reference_generalized_normal(
                n, p.item(), sigma.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_comparison(self):
        """Compare with scipy.signal.windows.general_gaussian."""
        scipy_signal = pytest.importorskip("scipy.signal")
        # scipy formula: exp(-0.5 * |x / sig|^(2*p_scipy))
        # Our formula: exp(-|x / sigma|^p_ours)
        # To match:
        #   - p_scipy = p_ours / 2
        #   - sigma = sig * 2^(1/p_ours)
        for n in [4, 16, 64]:
            for p_ours in [1.0, 2.0, 4.0]:
                for sig_scipy in [2.0, 5.0, 10.0]:
                    p_scipy = p_ours / 2.0
                    sigma_val = sig_scipy * (2.0 ** (1.0 / p_ours))
                    p = torch.tensor(p_ours, dtype=torch.float64)
                    sigma = torch.tensor(sigma_val, dtype=torch.float64)
                    result = wf.generalized_normal_window(
                        n, p, sigma, dtype=torch.float64
                    )
                    expected = torch.tensor(
                        scipy_signal.windows.general_gaussian(
                            n, p=p_scipy, sig=sig_scipy, sym=True
                        ),
                        dtype=torch.float64,
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-10, atol=1e-10
                    )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        for n in [0, 1, 5, 100]:
            result = wf.generalized_normal_window(n, p, sigma)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        p = torch.tensor(2.0, dtype=dtype)
        sigma = torch.tensor(3.0, dtype=dtype)
        result = wf.generalized_normal_window(64, p, sigma, dtype=dtype)
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        result = wf.generalized_normal_window(0, p, sigma)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result = wf.generalized_normal_window(1, p, sigma, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that window is symmetric."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_center_value(self):
        """Test that center of odd-length window is 1.0."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            center_idx = n // 2
            torch.testing.assert_close(
                result[center_idx],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_gradient_flow_p(self):
        """Test that gradients flow through p parameter."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result = wf.generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad)

    def test_gradient_flow_sigma(self):
        """Test that gradients flow through sigma parameter."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert sigma.grad is not None
        assert not torch.isnan(sigma.grad)

    def test_gradient_flow_both(self):
        """Test that gradients flow through both parameters."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert p.grad is not None
        assert sigma.grad is not None
        assert not torch.isnan(p.grad)
        assert not torch.isnan(sigma.grad)

    def test_gradcheck_sigma(self):
        """Test gradient correctness for sigma with torch.autograd.gradcheck."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.generalized_normal_window(
                32, p, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_gradcheck_p(self):
        """Test gradient correctness for p with torch.autograd.gradcheck."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64)

        def func(shape_p):
            return wf.generalized_normal_window(
                32, shape_p, sigma, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (p,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        with pytest.raises(ValueError):
            wf.generalized_normal_window(-1, p, sigma)

    def test_float_input(self):
        """Test that p and sigma can be passed as floats."""
        result = wf.generalized_normal_window(
            64, 2.0, 3.0, dtype=torch.float64
        )
        assert result.shape == (64,)

    def test_p_affects_shape(self):
        """Test that p parameter affects window shape."""
        n = 64
        sigma = torch.tensor(5.0, dtype=torch.float64)
        p_peaked = torch.tensor(1.0, dtype=torch.float64)  # More peaked
        p_flat = torch.tensor(4.0, dtype=torch.float64)  # Flatter top
        result_peaked = wf.generalized_normal_window(
            n, p_peaked, sigma, dtype=torch.float64
        )
        result_flat = wf.generalized_normal_window(
            n, p_flat, sigma, dtype=torch.float64
        )
        # p=1 (Laplacian) should have heavier tails than p=4
        # At the edges, p=1 should be larger
        assert result_peaked[0] > result_flat[0]
        assert result_peaked[-1] > result_flat[-1]

    def test_sigma_affects_width(self):
        """Test that sigma parameter affects window width."""
        n = 64
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma_narrow = torch.tensor(2.0, dtype=torch.float64)
        sigma_wide = torch.tensor(5.0, dtype=torch.float64)
        result_narrow = wf.generalized_normal_window(
            n, p, sigma_narrow, dtype=torch.float64
        )
        result_wide = wf.generalized_normal_window(
            n, p, sigma_wide, dtype=torch.float64
        )
        # Wider sigma should have larger values at the edges
        assert result_wide[0] > result_narrow[0]
        assert result_wide[-1] > result_narrow[-1]
        # Sum should be larger for wider window
        assert result_wide.sum() > result_narrow.sum()

    def test_values_positive(self):
        """Test that all values are positive."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            assert (result > 0).all()

    def test_values_bounded(self):
        """Test that values are in (0, 1] range."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            assert (result > 0).all()
            assert (result <= 1.0 + 1e-10).all()

    def test_max_value_at_center(self):
        """Test that maximum value is at the center."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # For even n, max could be at center-1 or center
            assert abs(max_idx - expected_center) <= 1

    def test_p_equals_1_laplacian(self):
        """Test that p=1 gives Laplacian (double exponential) decay."""
        n = 64
        p = torch.tensor(1.0, dtype=torch.float64)
        sigma = torch.tensor(5.0, dtype=torch.float64)
        result = wf.generalized_normal_window(n, p, sigma, dtype=torch.float64)
        # For p=1: w[k] = exp(-|k - center| / sigma)
        # This is exponential decay from center
        center = (n - 1) / 2.0
        k = torch.arange(n, dtype=torch.float64)
        expected = torch.exp(-torch.abs(k - center) / sigma)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_p_equals_2_gaussian(self):
        """Test that p=2 gives Gaussian-like shape."""
        n = 64
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(5.0, dtype=torch.float64)
        result = wf.generalized_normal_window(n, p, sigma, dtype=torch.float64)
        # For p=2: w[k] = exp(-((k - center) / sigma)^2)
        center = (n - 1) / 2.0
        k = torch.arange(n, dtype=torch.float64)
        normalized = (k - center) / sigma
        expected = torch.exp(-normalized * normalized)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_various_p_values(self):
        """Test window generation with various p values."""
        n = 32
        sigma = torch.tensor(
            10.0, dtype=torch.float64
        )  # Larger sigma to avoid underflow
        for p_val in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]:
            p = torch.tensor(p_val, dtype=torch.float64)
            result = wf.generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            # All should be valid windows
            assert result.shape == (n,)
            # Values should be non-negative (may underflow to 0 at edges for large p)
            assert (result >= 0).all()
            assert (result <= 1.0 + 1e-10).all()
            # Should be symmetric
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_mixed_dtype_promotion(self):
        """Test that mixed dtypes are handled correctly."""
        p = torch.tensor(2.0, dtype=torch.float32)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result = wf.generalized_normal_window(32, p, sigma)
        assert result.shape == (32,)

    @staticmethod
    def _reference_generalized_normal(
        n: int, p: float, sigma: float
    ) -> torch.Tensor:
        """Reference implementation of generalized normal window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        center = (n - 1) / 2.0
        k = torch.arange(n, dtype=torch.float64)
        normalized = torch.abs(k - center) / sigma
        return torch.exp(-torch.pow(normalized, p))


class TestPeriodicGeneralizedNormalWindow:
    """Tests for periodic_generalized_normal_window."""

    def test_reference(self):
        """Compare against reference implementation."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            expected = self._reference_periodic_generalized_normal(
                n, p.item(), sigma.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_scipy_comparison(self):
        """Compare with scipy.signal.windows.general_gaussian (periodic)."""
        scipy_signal = pytest.importorskip("scipy.signal")
        # scipy formula: exp(-0.5 * |x / sig|^(2*p_scipy))
        # Our formula: exp(-|x / sigma|^p_ours)
        # To match:
        #   - p_scipy = p_ours / 2
        #   - sigma = sig * 2^(1/p_ours)
        for n in [4, 16, 64]:
            for p_ours in [1.0, 2.0, 4.0]:
                for sig_scipy in [2.0, 5.0, 10.0]:
                    p_scipy = p_ours / 2.0
                    sigma_val = sig_scipy * (2.0 ** (1.0 / p_ours))
                    p = torch.tensor(p_ours, dtype=torch.float64)
                    sigma = torch.tensor(sigma_val, dtype=torch.float64)
                    result = wf.periodic_generalized_normal_window(
                        n, p, sigma, dtype=torch.float64
                    )
                    expected = torch.tensor(
                        scipy_signal.windows.general_gaussian(
                            n, p=p_scipy, sig=sig_scipy, sym=False
                        ),
                        dtype=torch.float64,
                    )
                    torch.testing.assert_close(
                        result, expected, rtol=1e-10, atol=1e-10
                    )

    def test_output_shape(self):
        """Test output shape is (n,)."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        for n in [0, 1, 5, 100]:
            result = wf.periodic_generalized_normal_window(n, p, sigma)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        p = torch.tensor(2.0, dtype=dtype)
        sigma = torch.tensor(3.0, dtype=dtype)
        result = wf.periodic_generalized_normal_window(
            64, p, sigma, dtype=dtype
        )
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        result = wf.periodic_generalized_normal_window(0, p, sigma)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result = wf.periodic_generalized_normal_window(
            1, p, sigma, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_center_differs_from_symmetric(self):
        """Test that periodic uses n/2 center vs symmetric (n-1)/2."""
        n = 10
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result_periodic = wf.periodic_generalized_normal_window(
            n, p, sigma, dtype=torch.float64
        )
        result_symmetric = wf.generalized_normal_window(
            n, p, sigma, dtype=torch.float64
        )
        # They should be different
        assert not torch.allclose(result_periodic, result_symmetric)
        # Periodic window has center at n/2 = 5, so index 5 should be max
        # Symmetric has center at (n-1)/2 = 4.5
        periodic_max_idx = result_periodic.argmax().item()
        assert periodic_max_idx == 5  # n/2 for even n

    def test_gradient_flow_p(self):
        """Test that gradients flow through p parameter."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        result = wf.periodic_generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert p.grad is not None
        assert not torch.isnan(p.grad)

    def test_gradient_flow_sigma(self):
        """Test that gradients flow through sigma parameter."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert sigma.grad is not None
        assert not torch.isnan(sigma.grad)

    def test_gradient_flow_both(self):
        """Test that gradients flow through both parameters."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_generalized_normal_window(
            32, p, sigma, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert p.grad is not None
        assert sigma.grad is not None
        assert not torch.isnan(p.grad)
        assert not torch.isnan(sigma.grad)

    def test_gradcheck_sigma(self):
        """Test gradient correctness for sigma with torch.autograd.gradcheck."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

        def func(s):
            return wf.periodic_generalized_normal_window(
                32, p, s, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (sigma,), raise_exception=True)

    def test_gradcheck_p(self):
        """Test gradient correctness for p with torch.autograd.gradcheck."""
        p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(3.0, dtype=torch.float64)

        def func(shape_p):
            return wf.periodic_generalized_normal_window(
                32, shape_p, sigma, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (p,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        p = torch.tensor(2.0)
        sigma = torch.tensor(3.0)
        with pytest.raises(ValueError):
            wf.periodic_generalized_normal_window(-1, p, sigma)

    def test_float_input(self):
        """Test that p and sigma can be passed as floats."""
        result = wf.periodic_generalized_normal_window(
            64, 2.0, 3.0, dtype=torch.float64
        )
        assert result.shape == (64,)

    def test_values_positive(self):
        """Test that all values are positive."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            assert (result > 0).all()

    def test_values_bounded(self):
        """Test that values are in (0, 1] range."""
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(3.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            assert (result > 0).all()
            assert (result <= 1.0 + 1e-10).all()

    def test_p_equals_1_laplacian(self):
        """Test that p=1 gives Laplacian (double exponential) decay."""
        n = 64
        p = torch.tensor(1.0, dtype=torch.float64)
        sigma = torch.tensor(5.0, dtype=torch.float64)
        result = wf.periodic_generalized_normal_window(
            n, p, sigma, dtype=torch.float64
        )
        # For p=1: w[k] = exp(-|k - center| / sigma)
        # Periodic center = n/2
        center = n / 2.0
        k = torch.arange(n, dtype=torch.float64)
        expected = torch.exp(-torch.abs(k - center) / sigma)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_p_equals_2_gaussian(self):
        """Test that p=2 gives Gaussian-like shape."""
        n = 64
        p = torch.tensor(2.0, dtype=torch.float64)
        sigma = torch.tensor(5.0, dtype=torch.float64)
        result = wf.periodic_generalized_normal_window(
            n, p, sigma, dtype=torch.float64
        )
        # For p=2: w[k] = exp(-((k - center) / sigma)^2)
        # Periodic center = n/2
        center = n / 2.0
        k = torch.arange(n, dtype=torch.float64)
        normalized = (k - center) / sigma
        expected = torch.exp(-normalized * normalized)
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_various_p_values(self):
        """Test window generation with various p values."""
        n = 32
        sigma = torch.tensor(10.0, dtype=torch.float64)
        for p_val in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]:
            p = torch.tensor(p_val, dtype=torch.float64)
            result = wf.periodic_generalized_normal_window(
                n, p, sigma, dtype=torch.float64
            )
            # All should be valid windows
            assert result.shape == (n,)
            # Values should be non-negative
            assert (result >= 0).all()
            assert (result <= 1.0 + 1e-10).all()

    @staticmethod
    def _reference_periodic_generalized_normal(
        n: int, p: float, sigma: float
    ) -> torch.Tensor:
        """Reference implementation of periodic generalized normal window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        # Periodic uses center = n/2
        center = n / 2.0
        k = torch.arange(n, dtype=torch.float64)
        normalized = torch.abs(k - center) / sigma
        return torch.exp(-torch.pow(normalized, p))
