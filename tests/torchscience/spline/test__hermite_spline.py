"""Tests for cubic Hermite spline interpolation with user-specified derivatives."""

import math

import pytest
import torch


class TestHermiteSplineFit:
    def test_fit_returns_hermite_spline(self):
        """Test that hermite_spline_fit returns a HermiteSpline tensorclass."""
        from torchscience.spline import HermiteSpline, hermite_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)
        dydx = math.pi * torch.cos(x * math.pi)

        spline = hermite_spline_fit(x, y, dydx)

        assert isinstance(spline, HermiteSpline)
        assert spline.knots.shape == (5,)
        assert spline.y.shape == (5,)
        assert spline.dydx.shape == (5,)

    def test_fit_validates_monotonic_knots(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, hermite_spline_fit

        x = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)
        dydx = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            hermite_spline_fit(x, y, dydx)

    def test_fit_validates_minimum_points(self):
        """Test that too few points raises KnotError."""
        from torchscience.spline import KnotError, hermite_spline_fit

        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)
        dydx = torch.tensor([0.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            hermite_spline_fit(x, y, dydx)

    def test_fit_validates_shape_match(self):
        """Test that mismatched y and dydx shapes raise ValueError."""
        from torchscience.spline import hermite_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)
        dydx = torch.cos(x[:4])  # Wrong shape

        with pytest.raises(ValueError):
            hermite_spline_fit(x, y, dydx)

    def test_fit_multidimensional_values(self):
        """Test fitting with multi-dimensional y values."""
        from torchscience.spline import hermite_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x), x], dim=-1)  # (5, 3)
        dydx = torch.stack(
            [torch.cos(x), -torch.sin(x), torch.ones_like(x)], dim=-1
        )

        spline = hermite_spline_fit(x, y, dydx)

        assert spline.y.shape == (5, 3)
        assert spline.dydx.shape == (5, 3)


class TestHermiteSplineEvaluate:
    def test_evaluate_at_knots(self):
        """Test that evaluating at knot points returns original y values."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)
        dydx = math.pi * torch.cos(x * math.pi)

        spline = hermite_spline_fit(x, y, dydx)
        y_eval = hermite_spline_evaluate(spline, x)

        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_evaluate_with_exact_derivatives(self):
        """Test that Hermite with exact derivatives reproduces function."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        # sin(pi*x) with exact derivatives
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x * math.pi)
        dydx = math.pi * torch.cos(x * math.pi)

        spline = hermite_spline_fit(x, y, dydx)

        # Evaluate at intermediate points
        t = torch.linspace(0, 1, 100, dtype=torch.float64)
        y_eval = hermite_spline_evaluate(spline, t)
        y_expected = torch.sin(t * math.pi)

        # With exact derivatives, should be very accurate
        torch.testing.assert_close(y_eval, y_expected, atol=1e-4, rtol=1e-4)

    def test_evaluate_polynomial_exact(self):
        """Test that Hermite exactly represents cubic polynomials."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        # f(x) = x^3, f'(x) = 3x^2
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x**3
        dydx = 3 * x**2

        spline = hermite_spline_fit(x, y, dydx)

        # Evaluate at many points
        t = torch.linspace(0, 1, 50, dtype=torch.float64)
        y_eval = hermite_spline_evaluate(spline, t)
        y_expected = t**3

        # Should be exact for cubic
        torch.testing.assert_close(y_eval, y_expected, atol=1e-10, rtol=1e-10)

    def test_evaluate_scalar_query(self):
        """Test that a single scalar query point works."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)
        dydx = torch.cos(x)

        spline = hermite_spline_fit(x, y, dydx)

        x_query = torch.tensor(0.5, dtype=torch.float64)
        y_eval = hermite_spline_evaluate(spline, x_query)

        assert y_eval.shape == ()

    def test_evaluate_extrapolate_error(self):
        """Test that query outside domain with extrapolate='error' raises."""
        from torchscience.spline import (
            ExtrapolationError,
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)
        dydx = torch.cos(x)

        spline = hermite_spline_fit(x, y, dydx, extrapolate="error")

        with pytest.raises(ExtrapolationError):
            hermite_spline_evaluate(
                spline, torch.tensor([-0.1], dtype=torch.float64)
            )

    def test_evaluate_extrapolate_clamp(self):
        """Test that extrapolate='clamp' clamps to boundary."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0], dtype=torch.float64)
        dydx = torch.tensor([1.0, 1.0, 0.0, -1.0, -1.0], dtype=torch.float64)

        spline = hermite_spline_fit(x, y, dydx, extrapolate="clamp")

        x_below = torch.tensor([-1.0], dtype=torch.float64)
        y_below = hermite_spline_evaluate(spline, x_below)
        y_at_0 = hermite_spline_evaluate(
            spline, torch.tensor([0.0], dtype=torch.float64)
        )

        torch.testing.assert_close(y_below, y_at_0, atol=1e-12, rtol=1e-12)

    def test_evaluate_multidimensional(self):
        """Test evaluation with multi-dimensional y values."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=-1
        )
        dydx = torch.stack(
            [
                math.pi * torch.cos(x * math.pi),
                -math.pi * torch.sin(x * math.pi),
            ],
            dim=-1,
        )

        spline = hermite_spline_fit(x, y, dydx)

        y_eval = hermite_spline_evaluate(spline, x)
        assert y_eval.shape == (5, 2)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_gradcheck(self):
        """Test that gradients flow through evaluation."""
        from torch.autograd import gradcheck

        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)
        dydx = math.pi * torch.cos(x * math.pi)

        spline = hermite_spline_fit(x, y, dydx)

        x_query = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return hermite_spline_evaluate(spline, xq)

        assert gradcheck(eval_fn, (x_query,), eps=1e-6, atol=1e-4)


class TestHermiteSplineDerivative:
    def test_derivative_matches_specified(self):
        """Test that first derivative matches specified derivatives at knots."""
        from torchscience.spline import (
            hermite_spline_derivative_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)
        dydx = math.pi * torch.cos(x * math.pi)

        spline = hermite_spline_fit(x, y, dydx)

        # Evaluate derivative at knots
        deriv_at_knots = hermite_spline_derivative_evaluate(spline, x, order=1)

        # Should match specified derivatives
        torch.testing.assert_close(
            deriv_at_knots, dydx, atol=1e-10, rtol=1e-10
        )

    def test_derivative_of_cubic(self):
        """Test derivative of x^3 is 3x^2."""
        from torchscience.spline import (
            hermite_spline_derivative_evaluate,
            hermite_spline_fit,
        )

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x**3
        dydx = 3 * x**2

        spline = hermite_spline_fit(x, y, dydx)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = hermite_spline_derivative_evaluate(spline, t, order=1)
        expected = 3 * t**2

        torch.testing.assert_close(
            deriv_eval, expected, atol=1e-10, rtol=1e-10
        )

    def test_second_derivative_of_cubic(self):
        """Test second derivative of x^3 is 6x."""
        from torchscience.spline import (
            hermite_spline_derivative_evaluate,
            hermite_spline_fit,
        )

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x**3
        dydx = 3 * x**2

        spline = hermite_spline_fit(x, y, dydx)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = hermite_spline_derivative_evaluate(spline, t, order=2)
        expected = 6 * t

        torch.testing.assert_close(
            deriv_eval, expected, atol=1e-10, rtol=1e-10
        )

    def test_third_derivative_of_cubic(self):
        """Test third derivative of x^3 is 6."""
        from torchscience.spline import (
            hermite_spline_derivative_evaluate,
            hermite_spline_fit,
        )

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x**3
        dydx = 3 * x**2

        spline = hermite_spline_fit(x, y, dydx)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = hermite_spline_derivative_evaluate(spline, t, order=3)
        expected = torch.full_like(t, 6.0)

        torch.testing.assert_close(
            deriv_eval, expected, atol=1e-10, rtol=1e-10
        )

    def test_derivative_invalid_order(self):
        """Test that invalid derivative order raises ValueError."""
        from torchscience.spline import (
            hermite_spline_derivative_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        dydx = 2 * x
        spline = hermite_spline_fit(x, y, dydx)

        t = torch.tensor([0.5], dtype=torch.float64)

        with pytest.raises(ValueError):
            hermite_spline_derivative_evaluate(spline, t, order=0)

        with pytest.raises(ValueError):
            hermite_spline_derivative_evaluate(spline, t, order=4)


class TestHermiteSplineIntegral:
    def test_integral_of_constant(self):
        """Test that integral of constant = constant * (b - a)."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = torch.full_like(x, 5.0)
        dydx = torch.zeros_like(x)

        spline = hermite_spline_fit(x, y, dydx)

        integral = hermite_spline_integral(spline, 0.0, 2.0)
        expected = torch.tensor(10.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_of_linear(self):
        """Test that integral of x from 0 to 1 = 0.5."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x.clone()
        dydx = torch.ones_like(x)

        spline = hermite_spline_fit(x, y, dydx)

        integral = hermite_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(0.5, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_of_cubic(self):
        """Test that integral of x^3 from 0 to 1 = 1/4."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x**3
        dydx = 3 * x**2

        spline = hermite_spline_fit(x, y, dydx)

        integral = hermite_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(0.25, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_negative_bounds(self):
        """Test that integral from b to a = -integral from a to b."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = x**2
        dydx = 2 * x

        spline = hermite_spline_fit(x, y, dydx)

        integral_forward = hermite_spline_integral(spline, 0.0, 1.0)
        integral_backward = hermite_spline_integral(spline, 1.0, 0.0)

        torch.testing.assert_close(
            integral_backward, -integral_forward, atol=1e-10, rtol=1e-10
        )

    def test_integral_same_bounds(self):
        """Test that integral with a == b returns zero."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        dydx = 2 * x

        spline = hermite_spline_fit(x, y, dydx)

        integral = hermite_spline_integral(spline, 0.5, 0.5)
        expected = torch.tensor(0.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-12, rtol=1e-12)

    def test_integral_multidimensional(self):
        """Test integral with multi-dimensional y values."""
        from torchscience.spline import (
            hermite_spline_fit,
            hermite_spline_integral,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x, x**2], dim=-1)
        dydx = torch.stack([torch.ones_like(x), 2 * x], dim=-1)

        spline = hermite_spline_fit(x, y, dydx)

        integral = hermite_spline_integral(spline, 0.0, 1.0)

        # Expected: integral of x = 0.5, integral of x^2 = 1/3
        expected = torch.tensor([0.5, 1.0 / 3.0], dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)


class TestHermiteSplineConvenience:
    def test_hermite_spline_convenience(self):
        """Test basic usage of hermite_spline convenience function."""
        from torchscience.spline import hermite_spline

        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)
        dydx = torch.cos(x)

        f = hermite_spline(x, y, dydx)

        y_eval = f(x)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_hermite_spline_convenience_returns_callable(self):
        """Test that hermite_spline returns a callable."""
        from torchscience.spline import hermite_spline

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        dydx = 2 * x

        f = hermite_spline(x, y, dydx)

        assert callable(f)
        result = f(torch.tensor([0.5], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)


class TestHermiteSplineBatching:
    """Tests for batched Hermite spline operations."""

    def test_fit_batched_y(self):
        """Test fitting with batched y values."""
        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.stack(
            [torch.sin(x * 2 * math.pi), torch.cos(x * 2 * math.pi), x**2],
            dim=-1,
        )
        dydx = torch.stack(
            [
                2 * math.pi * torch.cos(x * 2 * math.pi),
                -2 * math.pi * torch.sin(x * 2 * math.pi),
                2 * x,
            ],
            dim=-1,
        )

        spline = hermite_spline_fit(x, y, dydx)

        assert spline.y.shape == (10, 3)
        assert spline.dydx.shape == (10, 3)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_eval = hermite_spline_evaluate(spline, t)

        assert y_eval.shape == (3, 3)

    def test_gradcheck_batched(self):
        """Test that gradients flow correctly through batched operations."""
        from torch.autograd import gradcheck

        from torchscience.spline import (
            hermite_spline_evaluate,
            hermite_spline_fit,
        )

        x = torch.linspace(0, 1, 8, dtype=torch.float64)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=-1
        )
        dydx = torch.stack(
            [
                math.pi * torch.cos(x * math.pi),
                -math.pi * torch.sin(x * math.pi),
            ],
            dim=-1,
        )

        spline = hermite_spline_fit(x, y, dydx)

        t = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return hermite_spline_evaluate(spline, xq)

        assert gradcheck(eval_fn, (t,), eps=1e-6, atol=1e-4)
