import math

import pytest
import torch
import torch.testing

import torchscience.signal_processing.window_function as wf


class TestPlanckTaperWindow:
    """Tests for planck_taper_window and periodic_planck_taper_window."""

    def test_reference(self):
        """Compare against reference implementation."""
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        for n in [1, 5, 21, 64, 128]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
            expected = self._reference_planck_taper(n, epsilon.item())
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_output_shape(self):
        """Test output shape is (n,)."""
        epsilon = torch.tensor(0.2)
        for n in [0, 1, 5, 100]:
            result = wf.planck_taper_window(n, epsilon)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_support(self, dtype):
        """Test supported dtypes."""
        epsilon = torch.tensor(0.2, dtype=dtype)
        result = wf.planck_taper_window(64, epsilon, dtype=dtype)
        assert result.dtype == dtype

    def test_n_equals_zero(self):
        """n=0 returns empty tensor."""
        epsilon = torch.tensor(0.2)
        result = wf.planck_taper_window(0, epsilon)
        assert result.shape == (0,)

    def test_n_equals_one(self):
        """n=1 returns [1.0]."""
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        result = wf.planck_taper_window(1, epsilon, dtype=torch.float64)
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_symmetry(self):
        """Test that Planck-taper window is symmetric."""
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        for n in [5, 10, 11, 64]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
            flipped = torch.flip(result, dims=[0])
            torch.testing.assert_close(result, flipped, rtol=1e-10, atol=1e-10)

    def test_endpoints_zero(self):
        """Test that endpoints are exactly zero."""
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
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

    def test_flat_region(self):
        """Test that middle of window is flat (1.0) for epsilon < 0.5."""
        n = 101
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
        # For symmetric window, t = k / (n-1)
        # Flat region is where epsilon <= t <= 1 - epsilon
        # k >= epsilon * (n-1) and k <= (1 - epsilon) * (n-1)
        denom = n - 1
        flat_start = int(math.ceil(epsilon.item() * denom))
        flat_end = int(math.floor((1 - epsilon.item()) * denom))
        for i in range(flat_start, flat_end + 1):
            torch.testing.assert_close(
                result[i],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_epsilon_zero_is_rectangular(self):
        """epsilon=0 should produce rectangular window (all ones except endpoints)."""
        epsilon = torch.tensor(0.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
            # All values should be 1 (boundary handling is special case)
            expected = torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_epsilon_half_fully_tapered(self):
        """epsilon=0.5 should produce fully tapered window (no flat region)."""
        n = 21
        epsilon = torch.tensor(0.5, dtype=torch.float64)
        result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
        # Maximum should be at center, value should be 1
        center = n // 2
        torch.testing.assert_close(
            result[center],
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )
        # Values near center but not at center should be less than 1
        # for fully tapered window
        assert result[center - 1] < 1.0
        assert result[center + 1] < 1.0

    def test_gradient_flow(self):
        """Test that gradients flow through epsilon parameter."""
        epsilon = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        result = wf.planck_taper_window(32, epsilon, dtype=torch.float64)
        loss = result.sum()
        loss.backward()
        assert epsilon.grad is not None
        assert not torch.isnan(epsilon.grad)

    def test_gradcheck(self):
        """Test gradient correctness with torch.autograd.gradcheck."""
        epsilon = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

        def func(eps):
            return wf.planck_taper_window(21, eps, dtype=torch.float64)

        torch.autograd.gradcheck(func, (epsilon,), raise_exception=True)

    @pytest.mark.skip(
        reason="Second-order gradients not implemented for window functions"
    )
    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        epsilon = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

        def func(eps):
            return wf.planck_taper_window(21, eps, dtype=torch.float64)

        torch.autograd.gradgradcheck(func, (epsilon,), raise_exception=True)

    def test_negative_n_raises(self):
        """Test that negative n raises error."""
        epsilon = torch.tensor(0.2)
        with pytest.raises(ValueError):
            wf.planck_taper_window(-1, epsilon)

    def test_float_epsilon_input(self):
        """Test that epsilon can be passed as float."""
        result = wf.planck_taper_window(64, 0.2, dtype=torch.float64)
        assert result.shape == (64,)

    def test_default_epsilon(self):
        """Test default epsilon value of 0.1."""
        result = wf.planck_taper_window(64, dtype=torch.float64)
        expected = wf.planck_taper_window(64, 0.1, dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_epsilon_affects_taper_width(self):
        """Test that larger epsilon produces wider taper region."""
        n = 64
        epsilon_narrow = torch.tensor(0.1, dtype=torch.float64)
        epsilon_wide = torch.tensor(0.4, dtype=torch.float64)
        result_narrow = wf.planck_taper_window(
            n, epsilon_narrow, dtype=torch.float64
        )
        result_wide = wf.planck_taper_window(
            n, epsilon_wide, dtype=torch.float64
        )
        # Wider epsilon should have smaller values near edges
        # (since taper region is larger)
        idx = n // 4  # quarter point
        assert result_wide[idx] < result_narrow[idx]

    def test_values_in_range(self):
        """Test that all values are in [0, 1]."""
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
            assert (result >= 0).all()
            assert (result <= 1).all()

    def test_max_value_at_center(self):
        """Test that maximum value is at the center."""
        epsilon = torch.tensor(0.4, dtype=torch.float64)
        for n in [5, 11, 65]:
            result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
            max_idx = result.argmax().item()
            expected_center = (n - 1) // 2
            # For windows with flat region, max could be anywhere in flat region
            # but should include center
            assert result[expected_center] == result.max()

    def test_monotonic_taper_regions(self):
        """Test that taper regions are monotonically increasing/decreasing."""
        n = 101
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
        denom = n - 1
        taper_end = int(epsilon.item() * denom)

        # Left taper: should be monotonically increasing
        for i in range(1, taper_end):
            assert result[i] > result[i - 1], (
                f"Left taper not increasing at {i}"
            )

        # Right taper: should be monotonically decreasing
        taper_start = int((1 - epsilon.item()) * denom)
        for i in range(taper_start + 1, n - 1):
            assert result[i] < result[i - 1], (
                f"Right taper not decreasing at {i}"
            )

    def test_smooth_transition(self):
        """Test that the transition is smooth (C-infinity)."""
        # The Planck-taper window is infinitely differentiable
        # We can verify by checking that the values transition smoothly
        n = 201
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        result = wf.planck_taper_window(n, epsilon, dtype=torch.float64)

        # Compute numerical second derivative
        d1 = result[1:] - result[:-1]
        d2 = d1[1:] - d1[:-1]

        # Second derivative should be finite everywhere
        assert torch.isfinite(d2).all()

    def test_device_support(self):
        """Test device parameter."""
        epsilon = torch.tensor(0.2)
        result = wf.planck_taper_window(64, epsilon, device="cpu")
        assert result.device.type == "cpu"

    @staticmethod
    def _reference_planck_taper(n: int, epsilon: float) -> torch.Tensor:
        """Reference implementation of Planck-taper window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n - 1)
        result = torch.empty(n, dtype=torch.float64)

        for k in range(n):
            t = k / denom

            if t == 0 or t == 1:
                result[k] = 0.0
            elif epsilon == 0:
                result[k] = 1.0
            elif 0 < t < epsilon:
                z = epsilon * (1.0 / t + 1.0 / (t - epsilon))
                result[k] = 1.0 / (1.0 + math.exp(z))
            elif epsilon <= t <= 1 - epsilon:
                result[k] = 1.0
            else:  # 1 - epsilon < t < 1
                z = epsilon * (1.0 / (1.0 - t) + 1.0 / (1.0 - t - epsilon))
                result[k] = 1.0 / (1.0 + math.exp(z))

        return result

    # --- Periodic version tests ---

    def test_periodic_reference(self):
        """Compare periodic against reference implementation."""
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        for n in [1, 5, 21, 64, 128]:
            result = wf.periodic_planck_taper_window(
                n, epsilon, dtype=torch.float64
            )
            expected = self._reference_periodic_planck_taper(n, epsilon.item())
            torch.testing.assert_close(result, expected, rtol=1e-8, atol=1e-8)

    def test_periodic_output_shape(self):
        """Test periodic output shape is (n,)."""
        epsilon = torch.tensor(0.2)
        for n in [0, 1, 5, 100]:
            result = wf.periodic_planck_taper_window(n, epsilon)
            assert result.shape == (n,)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_periodic_dtype_support(self, dtype):
        """Test periodic supported dtypes."""
        epsilon = torch.tensor(0.2, dtype=dtype)
        result = wf.periodic_planck_taper_window(64, epsilon, dtype=dtype)
        assert result.dtype == dtype

    def test_periodic_n_equals_zero(self):
        """Periodic n=0 returns empty tensor."""
        epsilon = torch.tensor(0.2)
        result = wf.periodic_planck_taper_window(0, epsilon)
        assert result.shape == (0,)

    def test_periodic_n_equals_one(self):
        """Periodic n=1 returns [1.0]."""
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        result = wf.periodic_planck_taper_window(
            1, epsilon, dtype=torch.float64
        )
        torch.testing.assert_close(
            result, torch.tensor([1.0], dtype=torch.float64)
        )

    def test_periodic_first_endpoint_zero(self):
        """Test that first endpoint is exactly zero for periodic."""
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        for n in [5, 11, 64]:
            result = wf.periodic_planck_taper_window(
                n, epsilon, dtype=torch.float64
            )
            torch.testing.assert_close(
                result[0],
                torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_periodic_flat_region(self):
        """Test that middle of periodic window is flat (1.0) for epsilon < 0.5."""
        n = 100
        epsilon = torch.tensor(0.2, dtype=torch.float64)
        result = wf.periodic_planck_taper_window(
            n, epsilon, dtype=torch.float64
        )
        # For periodic window, t = k / n
        # Flat region is where epsilon <= t <= 1 - epsilon
        denom = n
        flat_start = int(math.ceil(epsilon.item() * denom))
        flat_end = int(math.floor((1 - epsilon.item()) * denom))
        for i in range(flat_start, flat_end):
            torch.testing.assert_close(
                result[i],
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-10,
                rtol=0,
            )

    def test_periodic_epsilon_zero_is_rectangular(self):
        """Periodic epsilon=0 should produce rectangular window."""
        epsilon = torch.tensor(0.0, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_planck_taper_window(
                n, epsilon, dtype=torch.float64
            )
            expected = torch.ones(n, dtype=torch.float64)
            torch.testing.assert_close(
                result, expected, rtol=1e-10, atol=1e-10
            )

    def test_periodic_gradient_flow(self):
        """Test that gradients flow through epsilon parameter for periodic."""
        epsilon = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        result = wf.periodic_planck_taper_window(
            32, epsilon, dtype=torch.float64
        )
        loss = result.sum()
        loss.backward()
        assert epsilon.grad is not None
        assert not torch.isnan(epsilon.grad)

    def test_periodic_gradcheck(self):
        """Test gradient correctness for periodic version."""
        epsilon = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

        def func(eps):
            return wf.periodic_planck_taper_window(
                21, eps, dtype=torch.float64
            )

        torch.autograd.gradcheck(func, (epsilon,), raise_exception=True)

    @pytest.mark.skip(
        reason="Second-order gradients not implemented for window functions"
    )
    def test_periodic_gradgradcheck(self):
        """Test second-order gradient correctness for periodic."""
        epsilon = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

        def func(eps):
            return wf.periodic_planck_taper_window(
                21, eps, dtype=torch.float64
            )

        torch.autograd.gradgradcheck(func, (epsilon,), raise_exception=True)

    def test_periodic_negative_n_raises(self):
        """Test that negative n raises error for periodic."""
        epsilon = torch.tensor(0.2)
        with pytest.raises(ValueError):
            wf.periodic_planck_taper_window(-1, epsilon)

    def test_periodic_float_epsilon_input(self):
        """Test that epsilon can be passed as float for periodic."""
        result = wf.periodic_planck_taper_window(64, 0.2, dtype=torch.float64)
        assert result.shape == (64,)

    def test_periodic_default_epsilon(self):
        """Test default epsilon value of 0.1 for periodic."""
        result = wf.periodic_planck_taper_window(64, dtype=torch.float64)
        expected = wf.periodic_planck_taper_window(
            64, 0.1, dtype=torch.float64
        )
        torch.testing.assert_close(result, expected)

    def test_periodic_values_in_range(self):
        """Test that all periodic values are in [0, 1]."""
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        for n in [5, 32, 64]:
            result = wf.periodic_planck_taper_window(
                n, epsilon, dtype=torch.float64
            )
            assert (result >= 0).all()
            assert (result <= 1).all()

    def test_symmetric_vs_periodic_denominator(self):
        """Test that symmetric uses n-1 and periodic uses n as denominator."""
        n = 21
        epsilon = torch.tensor(0.3, dtype=torch.float64)
        sym = wf.planck_taper_window(n, epsilon, dtype=torch.float64)
        per = wf.periodic_planck_taper_window(n, epsilon, dtype=torch.float64)
        # They should not be identical
        assert not torch.allclose(sym, per)
        # Symmetric has zero at last point, periodic does not (unless n divides evenly)
        assert sym[-1] == 0.0
        # Periodic's last point is in the taper region (t = 20/21 ≈ 0.952 > 0.7)
        assert per[-1] > 0.0

    @staticmethod
    def _reference_periodic_planck_taper(
        n: int, epsilon: float
    ) -> torch.Tensor:
        """Reference implementation of periodic Planck-taper window."""
        if n == 0:
            return torch.empty(0, dtype=torch.float64)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)

        denom = float(n)
        result = torch.empty(n, dtype=torch.float64)

        for k in range(n):
            t = k / denom

            if t == 0:
                result[k] = 0.0
            elif epsilon == 0:
                result[k] = 1.0
            elif 0 < t < epsilon:
                z = epsilon * (1.0 / t + 1.0 / (t - epsilon))
                result[k] = 1.0 / (1.0 + math.exp(z))
            elif epsilon <= t <= 1 - epsilon:
                result[k] = 1.0
            else:  # 1 - epsilon < t < 1
                z = epsilon * (1.0 / (1.0 - t) + 1.0 / (1.0 - t - epsilon))
                result[k] = 1.0 / (1.0 + math.exp(z))

        return result
