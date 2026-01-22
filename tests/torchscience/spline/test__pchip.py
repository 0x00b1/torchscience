"""Tests for PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline."""

import math

import pytest
import torch


class TestPCHIPFit:
    def test_fit_returns_pchip_spline(self):
        """Test that pchip_fit returns a PCHIPSpline tensorclass."""
        from torchscience.spline import PCHIPSpline, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = pchip_fit(x, y)

        assert isinstance(spline, PCHIPSpline)
        assert spline.knots.shape == (5,)
        assert spline.coefficients.shape == (4, 4)  # 4 segments, 4 coeffs each

    def test_fit_interpolates_data(self):
        """Test that PCHIP passes through all data points."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x * 2 * math.pi)

        spline = pchip_fit(x, y)
        y_eval = pchip_evaluate(spline, x)

        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_fit_validates_monotonic_knots(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, pchip_fit

        x = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            pchip_fit(x, y)

    def test_fit_validates_minimum_points(self):
        """Test that too few points raises KnotError."""
        from torchscience.spline import KnotError, pchip_fit

        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            pchip_fit(x, y)

    def test_fit_two_points(self):
        """Test fitting with exactly two points (linear interpolation)."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = torch.tensor([0.0, 2.0], dtype=torch.float64)

        spline = pchip_fit(x, y)

        # Evaluate at midpoint - should be linear
        t = torch.tensor([0.5], dtype=torch.float64)
        y_eval = pchip_evaluate(spline, t)

        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(y_eval, expected, atol=1e-10, rtol=1e-10)

    def test_fit_multidimensional_values(self):
        """Test fitting with multi-dimensional y values."""
        from torchscience.spline import pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x), x], dim=-1)  # (5, 3)

        spline = pchip_fit(x, y)

        # 4 segments, 4 coefficients, 3 value dimensions
        assert spline.coefficients.shape == (4, 4, 3)


class TestPCHIPMonotonicity:
    """Tests for PCHIP monotonicity preservation."""

    def test_preserves_monotonic_increasing(self):
        """Test that monotonically increasing data stays monotonic."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        # Strictly increasing data
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = x**2 + x  # Monotonically increasing

        spline = pchip_fit(x, y)

        # Evaluate at many points
        t = torch.linspace(0, 1, 100, dtype=torch.float64)
        y_eval = pchip_evaluate(spline, t)

        # Check monotonicity: all differences should be >= 0
        diffs = y_eval[1:] - y_eval[:-1]
        assert torch.all(diffs >= -1e-10), "PCHIP should preserve monotonicity"

    def test_preserves_monotonic_decreasing(self):
        """Test that monotonically decreasing data stays monotonic."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        # Strictly decreasing data
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = 1 - x**2  # Monotonically decreasing

        spline = pchip_fit(x, y)

        # Evaluate at many points
        t = torch.linspace(0, 1, 100, dtype=torch.float64)
        y_eval = pchip_evaluate(spline, t)

        # Check monotonicity: all differences should be <= 0
        diffs = y_eval[1:] - y_eval[:-1]
        assert torch.all(diffs <= 1e-10), "PCHIP should preserve monotonicity"

    def test_no_overshoot_monotonic_region(self):
        """Test that PCHIP doesn't overshoot in monotonic regions."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        # Monotonically increasing data
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

        spline = pchip_fit(x, y)

        # Evaluate at many points
        t = torch.linspace(0, 4, 100, dtype=torch.float64)
        y_eval = pchip_evaluate(spline, t)

        # All values should be between 0 and 1 (no overshoot in monotonic data)
        assert torch.all(y_eval >= -1e-10), "PCHIP should not undershoot"
        assert torch.all(y_eval <= 1.0 + 1e-10), "PCHIP should not overshoot"


class TestPCHIPEvaluate:
    def test_evaluate_at_knots(self):
        """Test that evaluating at knot points returns original y values."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = pchip_fit(x, y)
        y_eval = pchip_evaluate(spline, x)

        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_evaluate_scalar_query(self):
        """Test that a single scalar query point works."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = pchip_fit(x, y)

        x_query = torch.tensor(0.5, dtype=torch.float64)
        y_eval = pchip_evaluate(spline, x_query)

        assert y_eval.shape == ()
        assert y_eval >= y.min() and y_eval <= y.max()

    def test_evaluate_extrapolate_error(self):
        """Test that query outside domain with extrapolate='error' raises."""
        from torchscience.spline import (
            ExtrapolationError,
            pchip_evaluate,
            pchip_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = pchip_fit(x, y, extrapolate="error")

        with pytest.raises(ExtrapolationError):
            pchip_evaluate(spline, torch.tensor([-0.1], dtype=torch.float64))

        with pytest.raises(ExtrapolationError):
            pchip_evaluate(spline, torch.tensor([1.1], dtype=torch.float64))

    def test_evaluate_extrapolate_clamp(self):
        """Test that query outside domain with extrapolate='clamp' clamps."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0], dtype=torch.float64)

        spline = pchip_fit(x, y, extrapolate="clamp")

        x_below = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        y_below = pchip_evaluate(spline, x_below)
        y_at_0 = pchip_evaluate(
            spline, torch.tensor([0.0], dtype=torch.float64)
        )

        torch.testing.assert_close(
            y_below, y_at_0.expand(2), atol=1e-12, rtol=1e-12
        )

    def test_evaluate_extrapolate_extend(self):
        """Test that extrapolate='extend' extrapolates using boundary polynomial."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        # Linear function - should extrapolate exactly
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 2 * x + 1

        spline = pchip_fit(x, y, extrapolate="extend")

        x_query = torch.tensor([-0.5, 1.5], dtype=torch.float64)
        y_eval = pchip_evaluate(spline, x_query)

        y_expected = 2 * x_query + 1
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_multidimensional(self):
        """Test evaluation with multi-dimensional y values."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi), x], dim=-1
        )  # (5, 3)

        spline = pchip_fit(x, y)

        y_eval = pchip_evaluate(spline, x)
        assert y_eval.shape == (5, 3)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

        x_mid = torch.tensor([0.5], dtype=torch.float64)
        y_mid = pchip_evaluate(spline, x_mid)
        assert y_mid.shape == (1, 3)

    def test_gradcheck(self):
        """Test that gradients flow through evaluation."""
        from torch.autograd import gradcheck

        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = pchip_fit(x, y)

        x_query = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return pchip_evaluate(spline, xq)

        assert gradcheck(eval_fn, (x_query,), eps=1e-6, atol=1e-4)

    def test_scipy_comparison(self):
        """Test that results match scipy.interpolate.PchipInterpolator."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import PchipInterpolator

        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        # Fit with torchscience
        spline = pchip_fit(x, y)

        # Fit with scipy
        scipy_pchip = PchipInterpolator(x.numpy(), y.numpy())

        # Evaluate at many points
        x_query = torch.linspace(0, 2 * math.pi, 100, dtype=torch.float64)
        y_torch = pchip_evaluate(spline, x_query)
        y_scipy = torch.from_numpy(scipy_pchip(x_query.numpy()))

        torch.testing.assert_close(y_torch, y_scipy, atol=1e-10, rtol=1e-10)


class TestPCHIPDerivative:
    def test_derivative_of_linear(self):
        """Test that derivative of linear function is constant."""
        from torchscience.spline import (
            pchip_derivative,
            pchip_evaluate,
            pchip_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 3 * x + 2  # y = 3x + 2, dy/dx = 3

        spline = pchip_fit(x, y)
        deriv = pchip_derivative(spline, order=1)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = pchip_evaluate(deriv, t)

        expected = torch.full_like(t, 3.0)
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_of_quadratic(self):
        """Test that derivative of x^2 is approximately 2x."""
        from torchscience.spline import (
            pchip_derivative,
            pchip_evaluate,
            pchip_fit,
        )

        # Use more points for better quadratic approximation
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        spline = pchip_fit(x, y)
        deriv = pchip_derivative(spline, order=1)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = pchip_evaluate(deriv, t)

        expected = 2 * t
        # PCHIP is not exact for polynomials, so allow looser tolerance
        torch.testing.assert_close(deriv_eval, expected, atol=1e-2, rtol=1e-2)

    def test_derivative_invalid_order(self):
        """Test that invalid derivative order raises ValueError."""
        from torchscience.spline import pchip_derivative, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        spline = pchip_fit(x, y)

        with pytest.raises(ValueError):
            pchip_derivative(spline, order=0)

        with pytest.raises(ValueError):
            pchip_derivative(spline, order=4)

    def test_derivative_preserves_knots(self):
        """Test that derivative preserves the knot vector."""
        from torchscience.spline import pchip_derivative, pchip_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3
        spline = pchip_fit(x, y)

        deriv = pchip_derivative(spline, order=1)

        torch.testing.assert_close(deriv.knots, spline.knots)


class TestPCHIPIntegral:
    def test_integral_of_constant(self):
        """Test that integral of constant = constant * (b - a)."""
        from torchscience.spline import pchip_fit, pchip_integral

        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = torch.full_like(x, 5.0)

        spline = pchip_fit(x, y)

        integral = pchip_integral(spline, 0.0, 2.0)
        expected = torch.tensor(10.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_of_linear(self):
        """Test that integral of x from 0 to 1 = 0.5."""
        from torchscience.spline import pchip_fit, pchip_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x.clone()

        spline = pchip_fit(x, y)

        integral = pchip_integral(spline, 0.0, 1.0)
        expected = torch.tensor(0.5, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_negative_bounds(self):
        """Test that integral from b to a = -integral from a to b."""
        from torchscience.spline import pchip_fit, pchip_integral

        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = x**2

        spline = pchip_fit(x, y)

        integral_forward = pchip_integral(spline, 0.0, 1.0)
        integral_backward = pchip_integral(spline, 1.0, 0.0)

        torch.testing.assert_close(
            integral_backward, -integral_forward, atol=1e-10, rtol=1e-10
        )

    def test_integral_same_bounds(self):
        """Test that integral with a == b returns zero."""
        from torchscience.spline import pchip_fit, pchip_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        spline = pchip_fit(x, y)

        integral = pchip_integral(spline, 0.5, 0.5)
        expected = torch.tensor(0.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-12, rtol=1e-12)

    def test_integral_multidimensional(self):
        """Test integral with multi-dimensional y values."""
        from torchscience.spline import pchip_fit, pchip_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x, x**2], dim=-1)  # (5, 2)

        spline = pchip_fit(x, y)

        integral = pchip_integral(spline, 0.0, 1.0)

        # Expected: integral of x = 0.5, integral of x^2 ~= 1/3
        assert integral.shape == (2,)
        torch.testing.assert_close(
            integral[0],
            torch.tensor(0.5, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            integral[1],
            torch.tensor(1.0 / 3.0, dtype=torch.float64),
            atol=1e-3,
            rtol=1e-3,
        )


class TestPCHIPConvenience:
    def test_pchip_convenience(self):
        """Test basic usage of pchip convenience function."""
        from torchscience.spline import pchip

        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        f = pchip(x, y)

        y_eval = f(x)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_pchip_convenience_returns_callable(self):
        """Test that pchip returns a callable."""
        from torchscience.spline import pchip

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        f = pchip(x, y)

        assert callable(f)
        result = f(torch.tensor([0.5], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)

    def test_pchip_convenience_extrapolate_error(self):
        """Test that extrapolate='error' raises ExtrapolationError."""
        from torchscience.spline import ExtrapolationError, pchip

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x)

        f = pchip(x, y, extrapolate="error")

        with pytest.raises(ExtrapolationError):
            f(torch.tensor([-0.1], dtype=torch.float64))


class TestPCHIPBatching:
    """Tests for batched PCHIP operations."""

    def test_fit_batched_y(self):
        """Test fitting with batched y values (multiple curves)."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.stack(
            [
                torch.sin(x * 2 * math.pi),
                torch.cos(x * 2 * math.pi),
                x**2,
            ],
            dim=-1,
        )

        spline = pchip_fit(x, y)

        assert spline.coefficients.shape == (9, 4, 3)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_eval = pchip_evaluate(spline, t)

        assert y_eval.shape == (3, 3)

    def test_evaluate_batched_query(self):
        """Test evaluation with batched query points."""
        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x * 2 * math.pi)

        spline = pchip_fit(x, y)

        t = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(4, 5)
        y_eval = pchip_evaluate(spline, t)

        assert y_eval.shape == (4, 5)

        t_flat = t.flatten()
        y_flat = pchip_evaluate(spline, t_flat)
        torch.testing.assert_close(
            y_eval, y_flat.reshape(4, 5), atol=1e-10, rtol=1e-10
        )

    def test_gradcheck_batched(self):
        """Test that gradients flow correctly through batched operations."""
        from torch.autograd import gradcheck

        from torchscience.spline import pchip_evaluate, pchip_fit

        x = torch.linspace(0, 1, 8, dtype=torch.float64)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=-1
        )

        spline = pchip_fit(x, y)

        t = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return pchip_evaluate(spline, xq)

        assert gradcheck(eval_fn, (t,), eps=1e-6, atol=1e-4)
