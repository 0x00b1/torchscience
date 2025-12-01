"""Tests for LegendrePolynomialP fitting and interpolation."""

import math

import numpy as np
import pytest
import torch
from numpy.polynomial import legendre as np_leg

from torchscience.polynomial import (
    DomainError,
    legendre_polynomial_p,
    legendre_polynomial_p_evaluate,
    legendre_polynomial_p_fit,
    legendre_polynomial_p_interpolate,
    legendre_polynomial_p_linspace,
    legendre_polynomial_p_points,
    legendre_polynomial_p_vandermonde,
    legendre_polynomial_p_weight,
)


class TestLegendrePolynomialPPoints:
    """Tests for legendre_polynomial_p_points (Gauss-Legendre nodes)."""

    def test_points_n1(self):
        """Single Gauss-Legendre point."""
        x = legendre_polynomial_p_points(1)
        # x_0 = 0 (root of P_1(x) = x)
        torch.testing.assert_close(x, torch.tensor([0.0]))

    def test_points_n2(self):
        """Two Gauss-Legendre points."""
        x = legendre_polynomial_p_points(2)
        # Roots of P_2(x) = (3x^2 - 1)/2: x = +/- 1/sqrt(3)
        expected = torch.tensor(
            [1.0 / np.sqrt(3), -1.0 / np.sqrt(3)], dtype=torch.float32
        )
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_points_n5(self):
        """Five Gauss-Legendre points."""
        x = legendre_polynomial_p_points(5)
        assert x.shape == (5,)
        # Points should be in [-1, 1]
        assert x.min() >= -1.0
        assert x.max() <= 1.0

    def test_points_symmetric(self):
        """Gauss-Legendre points are symmetric about 0."""
        x = legendre_polynomial_p_points(5)
        # Sorted ascending + sorted descending should sum to zero
        torch.testing.assert_close(
            x.sort().values + x.sort(descending=True).values,
            torch.zeros(5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_points_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.leggauss."""
        n = 10
        x_torch = legendre_polynomial_p_points(n)
        x_np, _ = np_leg.leggauss(n)
        # Our implementation returns descending order
        np.testing.assert_allclose(x_torch.numpy(), x_np[::-1], rtol=1e-6)

    def test_points_dtype(self):
        """Preserve dtype."""
        x = legendre_polynomial_p_points(5, dtype=torch.float64)
        assert x.dtype == torch.float64


class TestLegendrePolynomialPVandermonde:
    """Tests for legendre_polynomial_p_vandermonde."""

    def test_vandermonde_shape(self):
        """Vandermonde matrix shape."""
        x = torch.tensor([0.0, 0.5, 1.0])
        V = legendre_polynomial_p_vandermonde(x, degree=3)
        assert V.shape == (3, 4)  # (n_points, degree+1)

    def test_vandermonde_first_column(self):
        """First column is all ones (P_0 = 1)."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = legendre_polynomial_p_vandermonde(x, degree=3)
        torch.testing.assert_close(V[:, 0], torch.ones(4))

    def test_vandermonde_second_column(self):
        """Second column is x (P_1 = x)."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = legendre_polynomial_p_vandermonde(x, degree=3)
        torch.testing.assert_close(V[:, 1], x)

    def test_vandermonde_third_column(self):
        """Third column is P_2 = (3x^2 - 1)/2."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = legendre_polynomial_p_vandermonde(x, degree=3)
        expected = (3 * x**2 - 1) / 2
        torch.testing.assert_close(V[:, 2], expected)

    def test_vandermonde_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legvander."""
        x = np.linspace(-1, 1, 10)
        deg = 5

        V_torch = legendre_polynomial_p_vandermonde(
            torch.tensor(x), degree=deg
        )
        V_np = np_leg.legvander(x, deg)

        np.testing.assert_allclose(V_torch.numpy(), V_np, rtol=1e-6)

    def test_vandermonde_evaluation_consistency(self):
        """V @ coeffs == evaluate(series, x)."""
        x = torch.linspace(-1, 1, 10)
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])

        V = legendre_polynomial_p_vandermonde(x, degree=3)
        y_vander = V @ coeffs

        c = legendre_polynomial_p(coeffs)
        y_eval = legendre_polynomial_p_evaluate(c, x)

        torch.testing.assert_close(y_vander, y_eval, atol=1e-5, rtol=1e-5)


class TestLegendrePolynomialPFit:
    """Tests for legendre_polynomial_p_fit."""

    def test_fit_exact_linear(self):
        """Fit exactly recovers linear function."""
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = 2.0 * x + 3.0  # 3 + 2*P_1 (since P_1 = x)

        c = legendre_polynomial_p_fit(x, y, degree=1)
        y_fit = legendre_polynomial_p_evaluate(c, x)

        torch.testing.assert_close(y_fit, y, atol=1e-5, rtol=1e-5)

    def test_fit_exact_quadratic(self):
        """Fit exactly recovers quadratic function."""
        x = legendre_polynomial_p_points(5)
        y = x**2 - 0.5 * x + 1.0

        c = legendre_polynomial_p_fit(x, y, degree=2)
        y_fit = legendre_polynomial_p_evaluate(c, x)

        torch.testing.assert_close(y_fit, y, atol=1e-4, rtol=1e-4)

    def test_fit_overdetermined(self):
        """Overdetermined system (more points than degree)."""
        x = torch.linspace(-1, 1, 20)
        # True function: 1 + 2*P_1 + 3*P_2
        coeffs_true = torch.tensor([1.0, 2.0, 3.0])
        y = legendre_polynomial_p_evaluate(
            legendre_polynomial_p(coeffs_true), x
        )

        c = legendre_polynomial_p_fit(x, y, degree=2)

        torch.testing.assert_close(c.coeffs, coeffs_true, atol=1e-5, rtol=1e-5)

    def test_fit_noisy_data(self):
        """Fit with noisy data."""
        torch.manual_seed(42)
        x = torch.linspace(-1, 1, 50)
        y_true = x**2
        y = y_true + 0.01 * torch.randn_like(y_true)

        c = legendre_polynomial_p_fit(x, y, degree=4)
        y_fit = legendre_polynomial_p_evaluate(c, x)

        # Residual should be small
        residual = (y_fit - y_true).abs().mean()
        assert residual < 0.05

    def test_fit_vs_numpy(self):
        """Compare with numpy.polynomial.legendre.legfit."""
        x = np.linspace(-1, 1, 20)
        y = np.sin(np.pi * x)
        deg = 5

        c_torch = legendre_polynomial_p_fit(
            torch.tensor(x), torch.tensor(y), degree=deg
        )
        c_np = np_leg.legfit(x, y, deg)

        np.testing.assert_allclose(
            c_torch.coeffs.numpy(), c_np, rtol=1e-5, atol=1e-5
        )

    def test_fit_outside_domain_raises(self):
        """Fitting outside [-1, 1] raises DomainError."""
        x = torch.tensor([0.0, 1.5, 2.0])
        y = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(DomainError):
            legendre_polynomial_p_fit(x, y, degree=2)


class TestLegendrePolynomialPInterpolate:
    """Tests for legendre_polynomial_p_interpolate."""

    def test_interpolate_linear(self):
        """Interpolate linear function."""

        def f(x):
            return 2 * x + 3

        c = legendre_polynomial_p_interpolate(f, n=2)

        # Verify at many points
        x = legendre_polynomial_p_linspace(10)
        y_interp = legendre_polynomial_p_evaluate(c, x)
        y_true = f(x)

        torch.testing.assert_close(y_interp, y_true, atol=1e-5, rtol=1e-5)

    def test_interpolate_polynomial(self):
        """Interpolate exactly recovers polynomial."""

        def f(x):
            return x**3 - 2 * x + 1

        c = legendre_polynomial_p_interpolate(f, n=4)

        x = torch.linspace(-1, 1, 20)
        y_interp = legendre_polynomial_p_evaluate(c, x)
        y_true = f(x)

        torch.testing.assert_close(y_interp, y_true, atol=1e-4, rtol=1e-4)

    def test_interpolate_sin(self):
        """Interpolate sin function."""

        def f(x):
            return torch.sin(math.pi * x)

        c = legendre_polynomial_p_interpolate(f, n=20)

        x = torch.linspace(-1, 1, 50)
        y_interp = legendre_polynomial_p_evaluate(c, x)
        y_true = f(x)

        # Should be very accurate for smooth function
        torch.testing.assert_close(y_interp, y_true, atol=1e-3, rtol=1e-3)

    def test_interpolate_at_nodes(self):
        """Interpolation is exact at Gauss-Legendre nodes."""

        def f(x):
            return x**2 - x + 1

        n = 5
        c = legendre_polynomial_p_interpolate(f, n=n)

        x_nodes = legendre_polynomial_p_points(n)
        y_interp = legendre_polynomial_p_evaluate(c, x_nodes)
        y_true = f(x_nodes)

        torch.testing.assert_close(y_interp, y_true, atol=1e-6, rtol=1e-6)


class TestLegendrePolynomialPWeight:
    """Tests for legendre_polynomial_p_weight."""

    def test_weight_all_ones(self):
        """Legendre weight is w(x) = 1."""
        x = torch.linspace(-1, 1, 10)
        w = legendre_polynomial_p_weight(x)
        torch.testing.assert_close(w, torch.ones(10))

    def test_weight_preserves_shape(self):
        """Weight function preserves input shape."""
        x = torch.randn(3, 4).clamp(-1, 1)
        w = legendre_polynomial_p_weight(x)
        assert w.shape == x.shape

    def test_weight_warns_outside_domain(self):
        """Warning when evaluating outside [-1, 1]."""
        x = torch.tensor([2.0])
        with pytest.warns(UserWarning, match="outside natural domain"):
            legendre_polynomial_p_weight(x)


class TestLegendrePolynomialPLinspace:
    """Tests for legendre_polynomial_p_linspace."""

    def test_linspace_default(self):
        """Default range is [-1, 1]."""
        pts = legendre_polynomial_p_linspace(5)
        torch.testing.assert_close(
            pts, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        )

    def test_linspace_custom_range(self):
        """Custom start and end."""
        pts = legendre_polynomial_p_linspace(3, start=0.0, end=1.0)
        torch.testing.assert_close(pts, torch.tensor([0.0, 0.5, 1.0]))

    def test_linspace_dtype(self):
        """Preserve dtype."""
        pts = legendre_polynomial_p_linspace(5, dtype=torch.float64)
        assert pts.dtype == torch.float64

    def test_linspace_partial_custom(self):
        """Only one of start/end specified."""
        pts = legendre_polynomial_p_linspace(3, end=0.5)
        torch.testing.assert_close(pts, torch.tensor([-1.0, -0.25, 0.5]))


class TestLegendrePolynomialPFittingAutograd:
    """Tests for autograd support in fitting operations."""

    def test_vandermonde_gradcheck(self):
        """Gradcheck for Vandermonde."""
        x = torch.tensor(
            [0.0, 0.3, 0.7, 1.0], dtype=torch.float64, requires_grad=True
        )

        def fn(x_):
            return legendre_polynomial_p_vandermonde(x_, degree=3)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_fit_gradcheck(self):
        """Gradcheck for fit w.r.t. y."""
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(y_):
            return legendre_polynomial_p_fit(x, y_, degree=2).coeffs

        assert torch.autograd.gradcheck(fn, (y,), raise_exception=True)

    def test_fit_gradgradcheck(self):
        """Second-order gradients for fit."""
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(y_):
            return legendre_polynomial_p_fit(x, y_, degree=2).coeffs.sum()

        assert torch.autograd.gradgradcheck(fn, (y,), raise_exception=True)
