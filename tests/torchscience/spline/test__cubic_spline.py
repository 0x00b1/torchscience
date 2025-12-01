"""Tests for cubic spline interpolation."""

import math

import pytest
import torch


class TestCubicSplineFit:
    def test_fit_returns_cubic_spline(self):
        """Test that cubic_spline_fit returns a CubicSpline tensorclass."""
        from torchscience.spline import CubicSpline, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        assert isinstance(spline, CubicSpline)
        assert spline.knots.shape == (5,)
        assert spline.coefficients.shape == (4, 4)  # 4 segments, 4 coeffs each

    def test_fit_natural_boundary(self):
        """Test natural boundary conditions (zero second derivative at ends)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3  # Known cubic

        spline = cubic_spline_fit(x, y, boundary="natural")

        # For natural splines, second derivative at endpoints should be ~0
        # Coefficient layout: [a, b, c, d] for a + b*(t-ti) + c*(t-ti)^2 + d*(t-ti)^3
        # Second derivative at ti is 2*c
        # Check first segment at x[0]
        assert abs(spline.coefficients[0, 2].item()) < 1e-6

    def test_fit_not_a_knot_boundary(self):
        """Test not-a-knot boundary (default)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="not_a_knot")

        assert spline.boundary == "not_a_knot"

    def test_fit_clamped_boundary(self):
        """Test clamped boundary conditions (known first derivatives)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # dy/dx = 2x, so at x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)

        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        assert spline.boundary == "clamped"

    def test_fit_periodic_boundary(self):
        """Test periodic boundary conditions."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="periodic")

        assert spline.boundary == "periodic"

    def test_fit_validates_monotonic_knots(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor(
            [0.0, 0.5, 0.3, 1.0], dtype=torch.float64
        )  # Not monotonic
        y = torch.tensor([0.0, 0.5, 0.3, 1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_validates_minimum_points(self):
        """Test that too few points raises KnotError."""
        from torchscience.spline import KnotError, cubic_spline_fit

        x = torch.tensor([0.0], dtype=torch.float64)
        y = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            cubic_spline_fit(x, y)

    def test_fit_multidimensional_values(self):
        """Test fitting with multi-dimensional y values (e.g., 3D curve)."""
        from torchscience.spline import cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x), x], dim=-1)  # (5, 3)

        spline = cubic_spline_fit(x, y)

        # 4 segments, 4 coefficients, 3 value dimensions
        assert spline.coefficients.shape == (4, 4, 3)


class TestCubicSplineEvaluate:
    def test_evaluate_at_knots(self):
        """Test that evaluating at knot points returns original y values."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = cubic_spline_fit(x, y)
        y_eval = cubic_spline_evaluate(spline, x)

        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

    def test_evaluate_between_knots(self):
        """Test that evaluating between knots interpolates smoothly."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        # Use a polynomial that a cubic spline can represent exactly
        # For y = x^3, dy/dx = 3x^2, so at x=0: 0, at x=1: 3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3  # Cubic polynomial

        # Use clamped boundary with correct derivatives for exact representation
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Query at midpoints
        x_query = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64
        )
        y_eval = cubic_spline_evaluate(spline, x_query)

        # For a cubic polynomial with matching boundary derivatives, spline is exact
        y_expected = x_query**3
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_scalar_query(self):
        """Test that a single scalar query point works."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)

        # Single query point (0-d tensor)
        x_query = torch.tensor(0.5, dtype=torch.float64)
        y_eval = cubic_spline_evaluate(spline, x_query)

        # Result should also be a scalar
        assert y_eval.shape == ()
        # Value should be reasonable (between min and max of y)
        assert y_eval >= y.min() and y_eval <= y.max()

    def test_evaluate_extrapolate_error(self):
        """Test that query outside domain with extrapolate='error' raises ExtrapolationError."""
        from torchscience.spline import (
            ExtrapolationError,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, extrapolate="error")

        # Query below domain
        x_query_below = torch.tensor([-0.1], dtype=torch.float64)
        with pytest.raises(ExtrapolationError):
            cubic_spline_evaluate(spline, x_query_below)

        # Query above domain
        x_query_above = torch.tensor([1.1], dtype=torch.float64)
        with pytest.raises(ExtrapolationError):
            cubic_spline_evaluate(spline, x_query_above)

    def test_evaluate_extrapolate_clamp(self):
        """Test that query outside domain with extrapolate='clamp' clamps to boundary."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0], dtype=torch.float64)

        spline = cubic_spline_fit(x, y, extrapolate="clamp")

        # Query below domain should return value at x[0]
        x_query_below = torch.tensor([-1.0, -0.5], dtype=torch.float64)
        y_eval_below = cubic_spline_evaluate(spline, x_query_below)
        y_at_start = cubic_spline_evaluate(
            spline, torch.tensor([0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            y_eval_below, y_at_start.expand(2), atol=1e-12, rtol=1e-12
        )

        # Query above domain should return value at x[-1]
        x_query_above = torch.tensor([1.5, 2.0], dtype=torch.float64)
        y_eval_above = cubic_spline_evaluate(spline, x_query_above)
        y_at_end = cubic_spline_evaluate(
            spline, torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            y_eval_above, y_at_end.expand(2), atol=1e-12, rtol=1e-12
        )

    def test_evaluate_extrapolate_extend(self):
        """Test that query outside domain with extrapolate='extend' extrapolates using boundary polynomial."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        # Use a linear function - extrapolation should follow the line
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = 2 * x + 1  # y = 2x + 1

        spline = cubic_spline_fit(x, y, extrapolate="extend")

        # Query outside domain
        x_query = torch.tensor([-0.5, 1.5], dtype=torch.float64)
        y_eval = cubic_spline_evaluate(spline, x_query)

        # For a linear function, extrapolation should be exact
        y_expected = 2 * x_query + 1
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_evaluate_multidimensional(self):
        """Test evaluation with multi-dimensional y values (e.g., 3D curve)."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        # 3D curve: (sin(t), cos(t), t)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi), x], dim=-1
        )  # (5, 3)

        spline = cubic_spline_fit(x, y)

        # Evaluate at knots
        y_eval = cubic_spline_evaluate(spline, x)

        assert y_eval.shape == (5, 3)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

        # Evaluate at midpoint
        x_mid = torch.tensor([0.5], dtype=torch.float64)
        y_mid = cubic_spline_evaluate(spline, x_mid)

        assert y_mid.shape == (1, 3)

    def test_gradcheck(self):
        """Test that gradients flow through evaluation."""
        from torch.autograd import gradcheck

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.sin(x * math.pi)

        spline = cubic_spline_fit(x, y)

        # Query points need gradients
        x_query = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return cubic_spline_evaluate(spline, xq)

        assert gradcheck(eval_fn, (x_query,), eps=1e-6, atol=1e-4)

    def test_scipy_comparison(self):
        """Test that results match scipy.interpolate.CubicSpline."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import CubicSpline as ScipyCubicSpline

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 2 * math.pi, 10, dtype=torch.float64)
        y = torch.sin(x)

        # Fit with torchscience
        spline = cubic_spline_fit(x, y, boundary="natural")

        # Fit with scipy
        scipy_spline = ScipyCubicSpline(
            x.numpy(), y.numpy(), bc_type="natural"
        )

        # Evaluate at many points
        x_query = torch.linspace(0, 2 * math.pi, 100, dtype=torch.float64)
        y_torch = cubic_spline_evaluate(spline, x_query)
        y_scipy = torch.from_numpy(scipy_spline(x_query.numpy()))

        torch.testing.assert_close(y_torch, y_scipy, atol=1e-10, rtol=1e-10)


class TestCubicSplineDerivative:
    """Tests for cubic_spline_derivative function."""

    def test_derivative_of_cubic(self):
        """Test that derivative of x^3 should be 3x^2."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3 with appropriate boundary conditions
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives for exact representation
        # dy/dx = 3x^2, so at x=0: 0, at x=1: 3
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 3x^2
        expected = 3 * x_test**2
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_of_quadratic(self):
        """Test that derivative of x^2 should be 2x."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^2
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # Use clamped boundary with correct derivatives
        # dy/dx = 2x, so at x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 2x
        expected = 2 * x_test
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_second_derivative(self):
        """Test that second derivative of x^3 should be 6x."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get second derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=2)

        # Evaluate second derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: 6x
        expected = 6 * x_test
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_third_derivative(self):
        """Test that third derivative of x^3 should be 6 (constant)."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit x^3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get third derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=3)

        # Evaluate third derivative at test points
        x_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: constant 6
        expected = torch.full_like(x_test, 6.0)
        torch.testing.assert_close(deriv_eval, expected, atol=1e-6, rtol=1e-6)

    def test_derivative_at_points(self):
        """Test derivative evaluation against analytical derivative of sin(x)."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit sin(x)
        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="natural")

        # Get derivative spline
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at interior test points (avoid boundary effects)
        x_test = torch.linspace(
            0.5, 2 * math.pi - 0.5, 10, dtype=torch.float64
        )
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: cos(x)
        expected = torch.cos(x_test)

        # Use looser tolerance since spline approximation isn't exact
        torch.testing.assert_close(deriv_eval, expected, atol=1e-3, rtol=1e-3)

    def test_derivative_invalid_order(self):
        """Test that invalid derivative order raises ValueError."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2
        spline = cubic_spline_fit(x, y)

        # Order 0 is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=0)

        # Order > 3 is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=4)

        # Negative order is invalid
        with pytest.raises(ValueError):
            cubic_spline_derivative(spline, order=-1)

    def test_derivative_preserves_knots(self):
        """Test that derivative preserves the knot vector."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3
        spline = cubic_spline_fit(x, y)

        deriv_spline = cubic_spline_derivative(spline, order=1)

        torch.testing.assert_close(deriv_spline.knots, spline.knots)

    def test_derivative_multidimensional(self):
        """Test derivative with multi-dimensional y values."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        # Fit a 2D curve: (t^2, t^3)
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x**2, x**3], dim=-1)  # (5, 2)

        spline = cubic_spline_fit(x, y)
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate derivative at test points
        x_test = torch.tensor([0.5], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, x_test)

        # Expected: (2t, 3t^2) at t=0.5: (1.0, 0.75)
        expected = torch.tensor([[1.0, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(deriv_eval, expected, atol=1e-4, rtol=1e-4)


class TestCubicSplineIntegral:
    """Tests for cubic_spline_integral function."""

    def test_integral_of_constant(self):
        """Test that integral of constant = constant * (b - a)."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline that represents a constant function y = 5
        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = torch.full_like(x, 5.0)

        spline = cubic_spline_fit(x, y, boundary="natural")

        # Integral of constant 5 from 0 to 2 should be 5 * 2 = 10
        integral = cubic_spline_integral(spline, 0.0, 2.0)
        expected = torch.tensor(10.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

        # Integral from 0.5 to 1.5 should be 5 * 1 = 5
        integral_partial = cubic_spline_integral(spline, 0.5, 1.5)
        expected_partial = torch.tensor(5.0, dtype=torch.float64)
        torch.testing.assert_close(
            integral_partial, expected_partial, atol=1e-10, rtol=1e-10
        )

    def test_integral_of_linear(self):
        """Test that integral of x from 0 to 1 = 0.5."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline that represents y = x
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x.clone()

        # Use clamped boundary with correct derivatives (dy/dx = 1)
        boundary_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Integral of x from 0 to 1 = x^2/2 |_0^1 = 0.5
        integral = cubic_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(0.5, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_of_quadratic(self):
        """Test that integral of x^2 from 0 to 1 = 1/3."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline that represents y = x^2
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        # Use clamped boundary with correct derivatives (dy/dx = 2x)
        # At x=0: 0, at x=1: 2
        boundary_values = torch.tensor([0.0, 2.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Integral of x^2 from 0 to 1 = x^3/3 |_0^1 = 1/3
        integral = cubic_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(1.0 / 3.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_of_cubic(self):
        """Test that integral of x^3 from 0 to 1 = 1/4."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline that represents y = x^3
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**3

        # Use clamped boundary with correct derivatives (dy/dx = 3x^2)
        # At x=0: 0, at x=1: 3
        boundary_values = torch.tensor([0.0, 3.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Integral of x^3 from 0 to 1 = x^4/4 |_0^1 = 1/4
        integral = cubic_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(0.25, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_negative_bounds(self):
        """Test that integral from b to a = -integral from a to b."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline for y = x^2
        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = x**2

        spline = cubic_spline_fit(x, y, boundary="natural")

        # Integral from 0 to 1
        integral_forward = cubic_spline_integral(spline, 0.0, 1.0)

        # Integral from 1 to 0 should be the negative
        integral_backward = cubic_spline_integral(spline, 1.0, 0.0)

        torch.testing.assert_close(
            integral_backward, -integral_forward, atol=1e-10, rtol=1e-10
        )

    def test_integral_partial_domain(self):
        """Test integral over a subset of the spline domain."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline for y = x (linear)
        x = torch.linspace(0, 4, 9, dtype=torch.float64)
        y = x.clone()

        # Use clamped boundary for exact linear representation
        boundary_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Integral of x from 1 to 3 = x^2/2 |_1^3 = 9/2 - 1/2 = 4
        integral = cubic_spline_integral(spline, 1.0, 3.0)
        expected = torch.tensor(4.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_scipy_comparison(self):
        """Compare integral to scipy.integrate.quad."""
        scipy = pytest.importorskip("scipy")
        from scipy.integrate import quad
        from scipy.interpolate import CubicSpline as ScipyCubicSpline

        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline for sin(x)
        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y, boundary="natural")

        # Create scipy spline for comparison
        scipy_spline = ScipyCubicSpline(
            x.numpy(), y.numpy(), bc_type="natural"
        )

        # Compute integrals
        a, b = 0.5, 2.0
        torch_integral = cubic_spline_integral(spline, a, b)
        scipy_integral, _ = quad(scipy_spline, a, b)

        torch.testing.assert_close(
            torch_integral,
            torch.tensor(scipy_integral, dtype=torch.float64),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_integral_tensor_bounds(self):
        """Test that tensor bounds work correctly."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a spline for y = x
        x = torch.linspace(0, 2, 5, dtype=torch.float64)
        y = x.clone()

        boundary_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Use tensor bounds
        a = torch.tensor(0.0, dtype=torch.float64)
        b = torch.tensor(1.0, dtype=torch.float64)

        integral = cubic_spline_integral(spline, a, b)
        expected = torch.tensor(0.5, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_multidimensional(self):
        """Test integral with multi-dimensional y values."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        # Create a 2D spline: (t, t^2)
        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.stack([x, x**2], dim=-1)  # (5, 2)

        # Use clamped boundary with correct derivatives for exact representation
        # For (t, t^2): derivatives are (1, 2t)
        # At t=0: (1, 0), at t=1: (1, 2)
        boundary_values = torch.tensor(
            [[1.0, 0.0], [1.0, 2.0]], dtype=torch.float64
        )
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Integral from 0 to 1
        integral = cubic_spline_integral(spline, 0.0, 1.0)

        # Expected: integral of (t, t^2) = (t^2/2, t^3/3) |_0^1 = (0.5, 1/3)
        expected = torch.tensor([0.5, 1.0 / 3.0], dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_integral_same_bounds(self):
        """Test that integral with a == b returns zero."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        spline = cubic_spline_fit(x, y)

        integral = cubic_spline_integral(spline, 0.5, 0.5)
        expected = torch.tensor(0.0, dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-12, rtol=1e-12)


class TestCubicSplineConvenience:
    """Tests for cubic_spline convenience function."""

    def test_cubic_spline_convenience(self):
        """Test basic usage of cubic_spline convenience function."""
        from torchscience.spline import cubic_spline

        # Create data from sin(x)
        x = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y = torch.sin(x)

        # Create callable using convenience function
        f = cubic_spline(x, y)

        # Evaluate at original points - should match original y values
        y_eval = f(x)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

        # Evaluate at intermediate points
        x_mid = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        y_mid = f(x_mid)

        # Results should be reasonable (close to sin values)
        y_expected = torch.sin(x_mid)
        torch.testing.assert_close(y_mid, y_expected, atol=1e-2, rtol=1e-2)

    def test_cubic_spline_convenience_boundary(self):
        """Test that boundary conditions are passed correctly."""
        from torchscience.spline import cubic_spline

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = x**2

        # Test with natural boundary
        f_natural = cubic_spline(x, y, boundary="natural")
        result = f_natural(torch.tensor([0.5], dtype=torch.float64))
        assert result.shape == (1,)

        # Test with periodic boundary
        x_periodic = torch.linspace(0, 2 * math.pi, 20, dtype=torch.float64)
        y_periodic = torch.sin(x_periodic)
        f_periodic = cubic_spline(x_periodic, y_periodic, boundary="periodic")
        result_periodic = f_periodic(
            torch.tensor([math.pi], dtype=torch.float64)
        )
        assert result_periodic.shape == (1,)

    def test_cubic_spline_convenience_extrapolate_error(self):
        """Test that extrapolate='error' raises ExtrapolationError."""
        from torchscience.spline import ExtrapolationError, cubic_spline

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x)

        # Default extrapolate='error' should raise on out-of-domain query
        f = cubic_spline(x, y, extrapolate="error")

        # Query below domain
        with pytest.raises(ExtrapolationError):
            f(torch.tensor([-0.1], dtype=torch.float64))

        # Query above domain
        with pytest.raises(ExtrapolationError):
            f(torch.tensor([1.1], dtype=torch.float64))

    def test_cubic_spline_convenience_extrapolate_clamp(self):
        """Test that extrapolate='clamp' clamps to boundary values."""
        from torchscience.spline import cubic_spline

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.linspace(0, 2, 10, dtype=torch.float64)  # y = 2*x

        f = cubic_spline(x, y, extrapolate="clamp")

        # Query outside domain should return boundary values
        y_outside = f(torch.tensor([-1.0, 2.0], dtype=torch.float64))
        y_at_0 = f(torch.tensor([0.0], dtype=torch.float64))
        y_at_1 = f(torch.tensor([1.0], dtype=torch.float64))

        torch.testing.assert_close(
            y_outside[0], y_at_0.squeeze(), atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            y_outside[1], y_at_1.squeeze(), atol=1e-12, rtol=1e-12
        )

    def test_cubic_spline_convenience_extrapolate_extend(self):
        """Test that extrapolate='extend' extrapolates using boundary polynomial."""
        from torchscience.spline import cubic_spline

        # Use a linear function - extrapolation should follow the line
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = 2 * x + 1  # y = 2x + 1

        f = cubic_spline(x, y, extrapolate="extend")

        # Query outside domain
        x_query = torch.tensor([-0.5, 1.5], dtype=torch.float64)
        y_eval = f(x_query)

        # For a linear function, extrapolation should be exact
        y_expected = 2 * x_query + 1
        torch.testing.assert_close(y_eval, y_expected, atol=1e-6, rtol=1e-6)

    def test_cubic_spline_convenience_returns_callable(self):
        """Test that cubic_spline returns a callable."""
        from torchscience.spline import cubic_spline

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = x**2

        f = cubic_spline(x, y)

        # Should be callable
        assert callable(f)

        # Should accept tensor input
        result = f(torch.tensor([0.5], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)

    def test_cubic_spline_convenience_multidimensional(self):
        """Test cubic_spline with multi-dimensional y values."""
        from torchscience.spline import cubic_spline

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        # 2D curve: (sin(t), cos(t))
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=-1
        )  # (10, 2)

        f = cubic_spline(x, y)

        # Evaluate at original points
        y_eval = f(x)
        assert y_eval.shape == (10, 2)
        torch.testing.assert_close(y_eval, y, atol=1e-12, rtol=1e-12)

        # Evaluate at single point
        y_mid = f(torch.tensor([0.5], dtype=torch.float64))
        assert y_mid.shape == (1, 2)


class TestCubicSplineBatching:
    """Tests for batched cubic spline operations."""

    def test_fit_batched_y(self):
        """Test fitting with batched y values (e.g., multiple curves)."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        # y has shape (10, 3) - 3 curves
        y = torch.stack(
            [
                torch.sin(x * 2 * math.pi),
                torch.cos(x * 2 * math.pi),
                x**2,
            ],
            dim=-1,
        )

        spline = cubic_spline_fit(x, y)

        # Coefficients should have shape (n_segments, 4, 3)
        assert spline.coefficients.shape == (9, 4, 3)

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_eval = cubic_spline_evaluate(spline, t)

        # Should have shape (3, 3) - 3 query points, 3 curves
        assert y_eval.shape == (3, 3)

        # Verify each curve matches individual fit
        for i in range(3):
            y_col = y[:, i]
            spline_single = cubic_spline_fit(x, y_col)
            y_single = cubic_spline_evaluate(spline_single, t)
            torch.testing.assert_close(
                y_eval[:, i], y_single, atol=1e-10, rtol=1e-10
            )

    def test_evaluate_batched_query(self):
        """Test evaluation with batched query points (2D query shape)."""
        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.sin(x * 2 * math.pi)

        spline = cubic_spline_fit(x, y)

        # Query with shape (batch, n_queries) = (4, 5)
        t = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(4, 5)
        y_eval = cubic_spline_evaluate(spline, t)

        # Output shape should match query shape
        assert y_eval.shape == (4, 5)

        # Verify values match flat evaluation
        t_flat = t.flatten()
        y_flat = cubic_spline_evaluate(spline, t_flat)
        torch.testing.assert_close(
            y_eval, y_flat.reshape(4, 5), atol=1e-10, rtol=1e-10
        )

    def test_derivative_batched(self):
        """Test derivative of batched spline (multiple curves)."""
        from torchscience.spline import (
            cubic_spline_derivative,
            cubic_spline_evaluate,
            cubic_spline_fit,
        )

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        # Two curves: x^2 and x^3
        y = torch.stack([x**2, x**3], dim=-1)

        # Use clamped boundary with correct derivatives for exact representation
        # dy/dx for x^2 at x=0: 0, at x=1: 2
        # dy/dx for x^3 at x=0: 0, at x=1: 3
        boundary_values = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0]], dtype=torch.float64
        )
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Get derivative
        deriv_spline = cubic_spline_derivative(spline, order=1)

        # Evaluate at test points
        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        deriv_eval = cubic_spline_evaluate(deriv_spline, t)

        # Should have shape (3, 2) - 3 query points, 2 curves
        assert deriv_eval.shape == (3, 2)

        # Expected: derivative of x^2 is 2x, derivative of x^3 is 3x^2
        expected_col0 = 2 * t  # d/dx(x^2) = 2x
        expected_col1 = 3 * t**2  # d/dx(x^3) = 3x^2

        torch.testing.assert_close(
            deriv_eval[:, 0], expected_col0, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            deriv_eval[:, 1], expected_col1, atol=1e-6, rtol=1e-6
        )

    def test_integral_batched(self):
        """Test integral of batched spline (multiple curves)."""
        from torchscience.spline import cubic_spline_fit, cubic_spline_integral

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        # Two curves: x and x^2
        y = torch.stack([x, x**2], dim=-1)

        # Use clamped boundary with correct derivatives for exact representation
        # dy/dx for x at x=0: 1, at x=1: 1
        # dy/dx for x^2 at x=0: 0, at x=1: 2
        boundary_values = torch.tensor(
            [[1.0, 0.0], [1.0, 2.0]], dtype=torch.float64
        )
        spline = cubic_spline_fit(
            x, y, boundary="clamped", boundary_values=boundary_values
        )

        # Compute integral from 0 to 1
        integral = cubic_spline_integral(spline, 0.0, 1.0)

        # Should have shape (2,) - one integral per curve
        assert integral.shape == (2,)

        # Expected: integral of x from 0 to 1 = 0.5
        #           integral of x^2 from 0 to 1 = 1/3
        expected = torch.tensor([0.5, 1.0 / 3.0], dtype=torch.float64)
        torch.testing.assert_close(integral, expected, atol=1e-10, rtol=1e-10)

    def test_gradcheck_batched(self):
        """Test that gradients flow correctly through batched operations."""
        from torch.autograd import gradcheck

        from torchscience.spline import cubic_spline_evaluate, cubic_spline_fit

        x = torch.linspace(0, 1, 8, dtype=torch.float64)
        # Batched y values with shape (8, 2)
        y = torch.stack(
            [torch.sin(x * math.pi), torch.cos(x * math.pi)], dim=-1
        )

        spline = cubic_spline_fit(x, y)

        # Query points need gradients
        t = torch.tensor(
            [0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )

        def eval_fn(xq):
            return cubic_spline_evaluate(spline, xq)

        assert gradcheck(eval_fn, (t,), eps=1e-6, atol=1e-4)
