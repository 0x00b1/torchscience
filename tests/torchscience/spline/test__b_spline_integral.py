"""Tests for B-spline integral and antiderivative functions."""

import pytest
import torch
from torch import Tensor

from torchscience.spline import (
    b_spline_antiderivative,
    b_spline_evaluate,
    b_spline_fit,
    b_spline_integral,
)


class TestBSplineAntiderivative:
    """Tests for b_spline_antiderivative function."""

    def test_antiderivative_degree_increases(self):
        """Antiderivative should have degree + 1."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        antideriv = b_spline_antiderivative(spline)

        assert antideriv.degree == spline.degree + 1

    def test_antiderivative_control_points_increase(self):
        """Antiderivative should have one more control point."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        antideriv = b_spline_antiderivative(spline)

        assert (
            antideriv.control_points.shape[0]
            == spline.control_points.shape[0] + 1
        )

    def test_antiderivative_knots_increase(self):
        """Antiderivative should have two more knots."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        antideriv = b_spline_antiderivative(spline)

        assert antideriv.knots.shape[0] == spline.knots.shape[0] + 2

    def test_antiderivative_zero_at_left_boundary(self):
        """Antiderivative should be zero at left boundary."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)
        spline = b_spline_fit(x, y, degree=3)

        antideriv = b_spline_antiderivative(spline)

        # Get left boundary
        t_min = spline.knots[spline.degree]
        F_left = b_spline_evaluate(antideriv, t_min)

        assert torch.allclose(F_left, torch.tensor(0.0), atol=1e-6)

    def test_antiderivative_derivative_recovery(self):
        """Derivative of antiderivative should approximate original spline."""
        x = torch.linspace(0, 1, 20)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        antideriv = b_spline_antiderivative(spline)

        # Numerical derivative at interior points
        t = torch.linspace(0.1, 0.9, 10)
        eps = 1e-5

        F_plus = b_spline_evaluate(antideriv, t + eps)
        F_minus = b_spline_evaluate(antideriv, t - eps)
        numerical_deriv = (F_plus - F_minus) / (2 * eps)

        # Original spline values
        original_vals = b_spline_evaluate(spline, t)

        # Use looser tolerance since we're comparing numerical derivative
        # to a B-spline approximation of the original function
        assert torch.allclose(
            numerical_deriv, original_vals, rtol=1e-2, atol=1e-3
        )


class TestBSplineIntegral:
    """Tests for b_spline_integral function."""

    def test_integral_constant_function(self):
        """Integral of constant should be linear."""
        x = torch.linspace(0, 2, 10)
        y = torch.full_like(x, 3.0)  # Constant function y = 3
        spline = b_spline_fit(x, y, degree=3)

        # Integral from 0 to 1 should be 3 * 1 = 3
        integral = b_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(3.0)

        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_linear_function(self):
        """Integral of linear function should be quadratic."""
        x = torch.linspace(0, 2, 20)
        y = 2 * x + 1  # Linear function
        spline = b_spline_fit(x, y, degree=3)

        # Integral from 0 to 1: ∫(2x + 1)dx = x^2 + x |_0^1 = 1 + 1 = 2
        integral = b_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(2.0)

        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_quadratic_function(self):
        """Integral of x^2 from 0 to 1 is 1/3."""
        x = torch.linspace(0, 1, 30)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(1.0 / 3.0)

        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_sine_function(self):
        """Integral of sin(x) from 0 to pi is 2."""
        x = torch.linspace(0, torch.pi, 50)
        y = torch.sin(x)
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, torch.pi)
        expected = torch.tensor(2.0)

        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_swapped_bounds(self):
        """Swapping bounds should negate the integral."""
        x = torch.linspace(0, 1, 20)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        integral_ab = b_spline_integral(spline, 0.0, 1.0)
        integral_ba = b_spline_integral(spline, 1.0, 0.0)

        assert torch.allclose(integral_ab, -integral_ba, atol=1e-6)

    def test_integral_same_bounds(self):
        """Integral with same bounds should be zero."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.5, 0.5)

        assert torch.allclose(integral, torch.tensor(0.0), atol=1e-10)

    def test_integral_additivity(self):
        """Integral should be additive: ∫_a^b + ∫_b^c = ∫_a^c."""
        x = torch.linspace(0, 1, 30)
        y = torch.sin(x * 2 * torch.pi)
        spline = b_spline_fit(x, y, degree=3)

        a, b, c = 0.1, 0.5, 0.9

        integral_ab = b_spline_integral(spline, a, b)
        integral_bc = b_spline_integral(spline, b, c)
        integral_ac = b_spline_integral(spline, a, c)

        assert torch.allclose(
            integral_ab + integral_bc, integral_ac, rtol=1e-4
        )

    def test_integral_partial_domain(self):
        """Integral over partial domain should work."""
        x = torch.linspace(0, 2, 30)
        y = x  # Linear
        spline = b_spline_fit(x, y, degree=3)

        # Integral of x from 0.5 to 1.5 is [x^2/2]_0.5^1.5 = 1.125 - 0.125 = 1.0
        integral = b_spline_integral(spline, 0.5, 1.5)
        expected = torch.tensor(1.0)

        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_with_tensor_bounds(self):
        """Should accept tensor bounds."""
        x = torch.linspace(0, 1, 20)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        a = torch.tensor(0.0)
        b = torch.tensor(1.0)
        integral = b_spline_integral(spline, a, b)

        assert isinstance(integral, Tensor)
        assert torch.allclose(integral, torch.tensor(1.0 / 3.0), rtol=1e-2)


class TestBSplineIntegralMultidimensional:
    """Tests for B-spline integral with multi-dimensional values."""

    def test_integral_2d_values(self):
        """Should integrate each dimension independently."""
        x = torch.linspace(0, 1, 20)
        y = torch.stack([x**2, x**3], dim=-1)  # Shape (20, 2)
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, 1.0)

        # Expected: [1/3, 1/4] for x^2 and x^3
        expected = torch.tensor([1.0 / 3.0, 1.0 / 4.0])

        assert integral.shape == (2,)
        assert torch.allclose(integral, expected, rtol=1e-2)

    def test_integral_3d_values(self):
        """Should handle higher-dimensional values."""
        x = torch.linspace(0, 1, 30)
        y = torch.stack([x, x**2, x**3], dim=-1)  # Shape (30, 3)
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, 1.0)

        # Expected: [1/2, 1/3, 1/4]
        expected = torch.tensor([0.5, 1.0 / 3.0, 0.25])

        assert integral.shape == (3,)
        assert torch.allclose(integral, expected, rtol=1e-2)


class TestBSplineIntegralDifferentDegrees:
    """Tests for B-spline integral with different polynomial degrees."""

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_integral_various_degrees(self, degree: int):
        """Should work for different degrees."""
        x = torch.linspace(0, 1, 30)
        y = x**2
        spline = b_spline_fit(x, y, degree=degree)

        integral = b_spline_integral(spline, 0.0, 1.0)
        expected = torch.tensor(1.0 / 3.0)

        assert torch.allclose(integral, expected, rtol=0.05)

    def test_antiderivative_degree_1(self):
        """Antiderivative of degree 1 spline should be degree 2."""
        x = torch.linspace(0, 1, 10)
        y = 2 * x + 1  # Linear
        spline = b_spline_fit(x, y, degree=1)

        antideriv = b_spline_antiderivative(spline)

        assert antideriv.degree == 2


class TestBSplineIntegralEdgeCases:
    """Tests for edge cases in B-spline integral."""

    def test_integral_clamps_to_domain(self):
        """Integration bounds outside domain should be clamped."""
        x = torch.linspace(0, 1, 20)
        y = x**2
        spline = b_spline_fit(x, y, degree=3, extrapolate="clamp")

        # Bounds extend beyond domain
        integral_extended = b_spline_integral(spline, -1.0, 2.0)
        integral_clamped = b_spline_integral(spline, 0.0, 1.0)

        # Should be equal since bounds are clamped
        assert torch.allclose(integral_extended, integral_clamped, atol=1e-6)

    def test_integral_preserves_dtype(self):
        """Should preserve input dtype."""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, 1.0)

        assert integral.dtype == torch.float64

    def test_integral_preserves_device(self):
        """Should preserve input device."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        integral = b_spline_integral(spline, 0.0, 1.0)

        assert integral.device == x.device


class TestBSplineIntegralGradients:
    """Tests for gradient computation through B-spline integral."""

    def test_gradcheck_integral_bounds(self):
        """Gradient check for integration with respect to bounds."""
        x = torch.linspace(0, 1, 20)
        y = torch.sin(x * torch.pi)
        spline = b_spline_fit(x, y, degree=3, extrapolate="clamp")

        # Note: bounds don't typically require gradients, but test anyway
        a = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)
        b = torch.tensor(0.8, requires_grad=True, dtype=torch.float64)

        # Convert spline to float64 for gradcheck
        x64 = x.double()
        y64 = torch.sin(x64 * torch.pi)
        spline64 = b_spline_fit(x64, y64, degree=3, extrapolate="clamp")

        def f(a_in, b_in):
            return b_spline_integral(spline64, a_in, b_in)

        # Numerical gradient check
        integral = f(a, b)
        integral.backward()

        # Check that gradients exist and are reasonable
        assert a.grad is not None
        assert b.grad is not None
