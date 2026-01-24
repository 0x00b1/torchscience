import pytest
import torch
import torch.testing

import torchscience.window_function as wf


class TestPlanckBesselWindow:
    """Tests for planck_bessel_window."""

    def test_symmetric_reference(self):
        """Compare against reference implementation."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            expected = self._reference_planck_bessel(
                n, epsilon.item(), beta.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        for n in [0, 1, 5, 100]:
            result = wf.planck_bessel_window(n, epsilon, beta)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        epsilon = torch.tensor(0.1, dtype=dtype)
        beta = torch.tensor(8.0, dtype=dtype)
        result = wf.planck_bessel_window(64, epsilon, beta, dtype=dtype)
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        result = wf.planck_bessel_window(0, epsilon, beta)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        result = wf.planck_bessel_window(1, epsilon, beta, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that symmetric Planck-Bessel window is symmetric."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that endpoints are zero (from Planck taper component)."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
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

    def test_gradient_flow_epsilon(self):
        """Test that gradients flow through epsilon parameter."""
        epsilon = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(8.0, dtype=torch.float64)
        result = wf.planck_bessel_window(
            32, epsilon, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert epsilon.grad is not None
        assert not torch.isnan(epsilon.grad)

    def test_gradient_flow_beta(self):
        """Test that gradients flow through beta parameter."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        result = wf.planck_bessel_window(
            32, epsilon, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert beta.grad is not None
        assert not torch.isnan(beta.grad)

    def test_gradient_flow_both(self):
        """Test that gradients flow through both parameters."""
        epsilon = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        result = wf.planck_bessel_window(
            32, epsilon, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert epsilon.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(epsilon.grad)
        assert not torch.isnan(beta.grad)

    def test_gradcheck_beta(self):
        """Test gradient correctness for beta with torch.autograd.gradcheck."""
        beta = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)

        def func(b):
            return wf.planck_bessel_window(
                32, 0.1, b, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (beta,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        with pytest.raises(ValueError):
            wf.planck_bessel_window(-1, epsilon, beta)

    def test_float_parameter_inputs(self):
        """Test that epsilon and beta can be passed as floats."""
        result = wf.planck_bessel_window(64, 0.1, 8.0, dtype=torch.float64)
        assert result.shape == (64,)

    def test_default_values(self):
        """Test default epsilon and beta values."""
        result = wf.planck_bessel_window(64, dtype=torch.float64)
        expected = wf.planck_bessel_window(64, 0.1, 8.0, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_epsilon_zero_approaches_kaiser(self):
        """Test that epsilon=0 approaches Kaiser window behavior."""
        n = 64
        beta_val = 6.0
        epsilon = torch.tensor(0.0, dtype=torch.float64)
        beta = torch.tensor(beta_val, dtype=torch.float64)
        # With epsilon=0, Planck taper is all ones except endpoints
        # But endpoints are forced to 0, so it's Kaiser with zero endpoints
        result = wf.planck_bessel_window(n, epsilon, beta, dtype=torch.float64)
        kaiser = wf.kaiser_window(n, beta, dtype=torch.float64)
        # Interior values should be close to Kaiser (Planck taper = 1)
        # Only endpoints differ (forced to 0 by Planck taper)
        torch.testing.assert_close(
            result[1:-1], kaiser[1:-1], rtol=1e-6, atol=1e-6
        )

    def test_beta_zero_is_planck_taper(self):
        """Test that beta=0 gives Planck taper (Kaiser becomes rectangular)."""
        n = 64
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(0.0, dtype=torch.float64)
        result = wf.planck_bessel_window(n, epsilon, beta, dtype=torch.float64)
        # With beta=0, Kaiser window is all ones
        # So result should be pure Planck taper
        expected = self._reference_planck_taper(n, epsilon.item())
        torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_epsilon_affects_taper_width(self):
        """Test that larger epsilon produces wider taper regions."""
        n = 64
        beta = torch.tensor(8.0, dtype=torch.float64)
        epsilon_narrow = torch.tensor(0.05, dtype=torch.float64)
        epsilon_wide = torch.tensor(0.2, dtype=torch.float64)
        result_narrow = wf.planck_bessel_window(
            n, epsilon_narrow, beta, dtype=torch.float64
        )
        result_wide = wf.planck_bessel_window(
            n, epsilon_wide, beta, dtype=torch.float64
        )
        # Wider epsilon should have smaller sum (more tapering)
        assert result_narrow.sum() > result_wide.sum()

    def test_beta_affects_shape(self):
        """Test that larger beta produces narrower Kaiser component."""
        n = 64
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta_small = torch.tensor(4.0, dtype=torch.float64)
        beta_large = torch.tensor(12.0, dtype=torch.float64)
        result_small = wf.planck_bessel_window(
            n, epsilon, beta_small, dtype=torch.float64
        )
        result_large = wf.planck_bessel_window(
            n, epsilon, beta_large, dtype=torch.float64
        )
        # Larger beta should generally result in smaller sum
        # (narrower Kaiser component)
        assert result_small.sum() > result_large.sum()

    def test_values_non_negative(self):
        """Test that all values are non-negative."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert (result >= 0).all()

    def test_values_bounded_by_one(self):
        """Test that all values are <= 1."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert (result <= 1.0 + 1e-10).all()

    def test_max_value_near_center(self):
        """Test that maximum value is near the center."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # Max should be within 1 of center
            assert abs(max_idx - expected_center) <= 1

    def test_center_value_close_to_one(self):
        """Test that center value is close to 1.0 for small epsilon."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [65, 129]:
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            center_idx = n // 2
            # Center should be close to 1.0 (Planck taper = 1, Kaiser = 1)
            assert result[center_idx] > 0.99

    def test_monotonic_increase_left_half(self):
        """Test that window increases monotonically in left half."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        n = 65
        result = wf.planck_bessel_window(n, epsilon, beta, dtype=torch.float64)
        center = n // 2
        # Check monotonic increase from start to center
        for i in range(center):
            assert result[i] <= result[i + 1] + 1e-10

    def test_various_epsilon_values(self):
        """Test with various epsilon values."""
        beta = torch.tensor(8.0, dtype=torch.float64)
        n = 64
        for eps_val in [0.01, 0.05, 0.1, 0.2, 0.3]:
            epsilon = torch.tensor(eps_val, dtype=torch.float64)
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert (result >= 0).all()
            assert result[0] == 0.0
            assert result[-1] == 0.0

    def test_various_beta_values(self):
        """Test with various beta values."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        n = 64
        for beta_val in [0.0, 2.0, 5.0, 8.0, 12.0]:
            beta = torch.tensor(beta_val, dtype=torch.float64)
            result = wf.planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert (result >= 0).all()
            assert result[0] == 0.0
            assert result[-1] == 0.0

    @staticmethod
    def _reference_planck_bessel(
        n: int, epsilon: float, beta: float
    ) -> torch.Tensor:
        """Reference implementation of Planck-Bessel window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        N = float(n - 1)
        k = torch.arange(n, dtype=torch.float64)

        # Planck taper component
        taper_width = epsilon * N

        planck_taper = torch.ones_like(k)

        # Left taper region
        left_mask = (k > 0) & (k < taper_width)
        if left_mask.any():
            k_left = k[left_mask]
            z_left = taper_width * (
                1.0 / k_left + 1.0 / (k_left - taper_width)
            )
            planck_taper[left_mask] = 1.0 / (1.0 + torch.exp(z_left))

        # Right taper region (mirrored for symmetry)
        right_taper_start = N - taper_width
        right_mask = (k > right_taper_start) & (k < N)
        if right_mask.any():
            k_right = k[right_mask]
            k_mirrored = N - k_right
            z_right = taper_width * (
                1.0 / k_mirrored + 1.0 / (k_mirrored - taper_width)
            )
            planck_taper[right_mask] = 1.0 / (1.0 + torch.exp(z_right))

        planck_taper[0] = 0.0
        if n > 1:
            planck_taper[-1] = 0.0

        # Kaiser-Bessel component
        center = N / 2.0
        x = (k - center) / center
        arg = beta * torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
        kaiser = torch.i0(arg) / torch.i0(
            torch.tensor(beta, dtype=torch.float64)
        )

        return planck_taper * kaiser

    @staticmethod
    def _reference_planck_taper(n: int, epsilon: float) -> torch.Tensor:
        """Reference implementation of Planck taper window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        N = float(n - 1)
        k = torch.arange(n, dtype=torch.float64)

        taper_width = epsilon * N

        planck_taper = torch.ones_like(k)

        left_mask = (k > 0) & (k < taper_width)
        if left_mask.any():
            k_left = k[left_mask]
            z_left = taper_width * (
                1.0 / k_left + 1.0 / (k_left - taper_width)
            )
            planck_taper[left_mask] = 1.0 / (1.0 + torch.exp(z_left))

        right_taper_start = N - taper_width
        right_mask = (k > right_taper_start) & (k < N)
        if right_mask.any():
            k_right = k[right_mask]
            k_mirrored = N - k_right
            z_right = taper_width * (
                1.0 / k_mirrored + 1.0 / (k_mirrored - taper_width)
            )
            planck_taper[right_mask] = 1.0 / (1.0 + torch.exp(z_right))

        planck_taper[0] = 0.0
        if n > 1:
            planck_taper[-1] = 0.0

        return planck_taper


class TestPeriodicPlanckBesselWindow:
    """Tests for periodic_planck_bessel_window."""

    def test_periodic_reference(self):
        """Compare against reference implementation."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [1, 5, 64, 128]:
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            expected = self._reference_periodic_planck_bessel(
                n, epsilon.item(), beta.item()
            )
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        for n in [0, 1, 5, 100]:
            result = wf.periodic_planck_bessel_window(n, epsilon, beta)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        epsilon = torch.tensor(0.1, dtype=dtype)
        beta = torch.tensor(8.0, dtype=dtype)
        result = wf.periodic_planck_bessel_window(
            64, epsilon, beta, dtype=dtype
        )
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        result = wf.periodic_planck_bessel_window(0, epsilon, beta)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        result = wf.periodic_planck_bessel_window(
            1, epsilon, beta, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_first_endpoint_zero(self):
        """Test that first endpoint is zero (from Planck taper component)."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_gradient_flow_epsilon(self):
        """Test that gradients flow through epsilon parameter."""
        epsilon = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(8.0, dtype=torch.float64)
        result = wf.periodic_planck_bessel_window(
            32, epsilon, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert epsilon.grad is not None
        assert not torch.isnan(epsilon.grad)

    def test_gradient_flow_beta(self):
        """Test that gradients flow through beta parameter."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_planck_bessel_window(
            32, epsilon, beta, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert beta.grad is not None
        assert not torch.isnan(beta.grad)

    def test_gradcheck_beta(self):
        """Test gradient correctness for beta with torch.autograd.gradcheck."""
        beta = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)

        def func(b):
            return wf.periodic_planck_bessel_window(
                32, 0.1, b, dtype=torch.float64
            ).sum()

        torch.autograd.gradcheck(func, (beta,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        epsilon = torch.tensor(0.1)
        beta = torch.tensor(8.0)
        with pytest.raises(ValueError):
            wf.periodic_planck_bessel_window(-1, epsilon, beta)

    def test_float_parameter_inputs(self):
        """Test that epsilon and beta can be passed as floats."""
        result = wf.periodic_planck_bessel_window(
            64, 0.1, 8.0, dtype=torch.float64
        )
        assert result.shape == (64,)

    def test_default_values(self):
        """Test default epsilon and beta values."""
        result = wf.periodic_planck_bessel_window(64, dtype=torch.float64)
        expected = wf.periodic_planck_bessel_window(
            64, 0.1, 8.0, dtype=torch.float64
        )
        torch.testing.assert_close(result, expected)

    def test_values_non_negative(self):
        """Test that all values are non-negative."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert (result >= 0).all()

    def test_values_bounded_by_one(self):
        """Test that all values are <= 1."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert (result <= 1.0 + 1e-10).all()

    def test_differs_from_symmetric(self):
        """Test that periodic version differs from symmetric version."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(8.0, dtype=torch.float64)
        n = 64
        symmetric = wf.planck_bessel_window(
            n, epsilon, beta, dtype=torch.float64
        )
        periodic = wf.periodic_planck_bessel_window(
            n, epsilon, beta, dtype=torch.float64
        )
        # They should be different (different denominators)
        assert not torch.allclose(symmetric, periodic, rtol=1e-6, atol=1e-6)

    def test_various_epsilon_values(self):
        """Test with various epsilon values."""
        beta = torch.tensor(8.0, dtype=torch.float64)
        n = 64
        for eps_val in [0.01, 0.05, 0.1, 0.2, 0.3]:
            epsilon = torch.tensor(eps_val, dtype=torch.float64)
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert (result >= 0).all()
            assert result[0] == 0.0

    def test_various_beta_values(self):
        """Test with various beta values."""
        epsilon = torch.tensor(0.1, dtype=torch.float64)
        n = 64
        for beta_val in [0.0, 2.0, 5.0, 8.0, 12.0]:
            beta = torch.tensor(beta_val, dtype=torch.float64)
            result = wf.periodic_planck_bessel_window(
                n, epsilon, beta, dtype=torch.float64
            )
            assert result.shape == (n,)
            assert (result >= 0).all()
            assert result[0] == 0.0

    @staticmethod
    def _reference_periodic_planck_bessel(
        n: int, epsilon: float, beta: float
    ) -> torch.Tensor:
        """Reference implementation of periodic Planck-Bessel window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        N = float(n)  # Periodic uses n, not n-1
        k = torch.arange(n, dtype=torch.float64)

        # Planck taper component
        taper_width = epsilon * N

        planck_taper = torch.ones_like(k)

        # Left taper region
        left_mask = (k > 0) & (k < taper_width)
        if left_mask.any():
            k_left = k[left_mask]
            z_left = taper_width * (
                1.0 / k_left + 1.0 / (k_left - taper_width)
            )
            planck_taper[left_mask] = 1.0 / (1.0 + torch.exp(z_left))

        # Right taper region (mirrored for symmetry)
        right_taper_start = N - taper_width
        right_mask = (k > right_taper_start) & (k < N)
        if right_mask.any():
            k_right = k[right_mask]
            k_mirrored = N - k_right
            z_right = taper_width * (
                1.0 / k_mirrored + 1.0 / (k_mirrored - taper_width)
            )
            planck_taper[right_mask] = 1.0 / (1.0 + torch.exp(z_right))

        planck_taper[0] = 0.0

        # Kaiser-Bessel component (periodic uses n as denominator)
        center = N / 2.0
        x = (k - center) / center
        arg = beta * torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
        kaiser = torch.i0(arg) / torch.i0(
            torch.tensor(beta, dtype=torch.float64)
        )

        return planck_taper * kaiser
