"""Tests for ChebyshevPolynomialT fitting and interpolation."""

import math

import numpy as np
import torch
from numpy.polynomial import chebyshev as np_cheb

from torchscience.polynomial import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_t_evaluate,
    chebyshev_polynomial_t_fit,
    chebyshev_polynomial_t_interpolate,
    chebyshev_polynomial_t_points,
    chebyshev_polynomial_t_vandermonde,
)


class TestChebyshevPolynomialTPoints:
    """Tests for chebyshev_polynomial_t_points (Chebyshev nodes)."""

    def test_points_n1(self):
        """Single Chebyshev point."""
        x = chebyshev_polynomial_t_points(1)
        # x_0 = cos(pi/2) = 0
        torch.testing.assert_close(x, torch.tensor([0.0]))

    def test_points_n2(self):
        """Two Chebyshev points."""
        x = chebyshev_polynomial_t_points(2)
        # x_k = cos((2k+1)*pi/(2n)) for k=0,1
        # x_0 = cos(pi/4) = sqrt(2)/2
        # x_1 = cos(3*pi/4) = -sqrt(2)/2
        expected = torch.tensor(
            [np.sqrt(2) / 2, -np.sqrt(2) / 2], dtype=torch.float32
        )
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_points_n5(self):
        """Five Chebyshev points."""
        x = chebyshev_polynomial_t_points(5)
        assert x.shape == (5,)
        # Points should be in [-1, 1]
        assert x.min() >= -1.0
        assert x.max() <= 1.0
        # Points should be symmetric around 0
        torch.testing.assert_close(
            x.sort().values + x.sort(descending=True).values,
            torch.zeros(5),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_points_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebpts1."""
        n = 10
        x_torch = chebyshev_polynomial_t_points(n)
        x_np = np_cheb.chebpts1(n)
        # NumPy uses ascending order, ours is descending, so flip for comparison
        np.testing.assert_allclose(x_torch.numpy()[::-1], x_np, rtol=1e-6)

    def test_points_dtype(self):
        """Preserve dtype."""
        x = chebyshev_polynomial_t_points(5, dtype=torch.float64)
        assert x.dtype == torch.float64


class TestChebyshevPolynomialTVandermonde:
    """Tests for chebyshev_polynomial_t_vandermonde."""

    def test_vandermonde_shape(self):
        """Vandermonde matrix shape."""
        x = torch.tensor([0.0, 0.5, 1.0])
        V = chebyshev_polynomial_t_vandermonde(x, degree=3)
        assert V.shape == (3, 4)  # (n_points, degree+1)

    def test_vandermonde_first_column(self):
        """First column is all ones (T_0 = 1)."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = chebyshev_polynomial_t_vandermonde(x, degree=3)
        torch.testing.assert_close(V[:, 0], torch.ones(4))

    def test_vandermonde_second_column(self):
        """Second column is x (T_1 = x)."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = chebyshev_polynomial_t_vandermonde(x, degree=3)
        torch.testing.assert_close(V[:, 1], x)

    def test_vandermonde_third_column(self):
        """Third column is T_2 = 2x^2 - 1."""
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        V = chebyshev_polynomial_t_vandermonde(x, degree=3)
        expected = 2 * x**2 - 1
        torch.testing.assert_close(V[:, 2], expected)

    def test_vandermonde_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebvander."""
        x = np.linspace(-1, 1, 10)
        deg = 5

        V_torch = chebyshev_polynomial_t_vandermonde(
            torch.tensor(x), degree=deg
        )
        V_np = np_cheb.chebvander(x, deg)

        np.testing.assert_allclose(V_torch.numpy(), V_np, rtol=1e-6)

    def test_vandermonde_evaluation_consistency(self):
        """V @ coeffs == evaluate(series, x)."""
        x = torch.linspace(-1, 1, 10)
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])

        V = chebyshev_polynomial_t_vandermonde(x, degree=3)
        y_vander = V @ coeffs

        c = chebyshev_polynomial_t(coeffs)
        y_eval = chebyshev_polynomial_t_evaluate(c, x)

        torch.testing.assert_close(y_vander, y_eval, atol=1e-5, rtol=1e-5)


class TestChebyshevPolynomialTFit:
    """Tests for chebyshev_polynomial_t_fit."""

    def test_fit_exact_linear(self):
        """Fit exactly recovers linear function."""
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = 2.0 * x + 3.0  # 3 + 2*T_1

        c = chebyshev_polynomial_t_fit(x, y, degree=1)
        y_fit = chebyshev_polynomial_t_evaluate(c, x)

        torch.testing.assert_close(y_fit, y, atol=1e-5, rtol=1e-5)

    def test_fit_exact_quadratic(self):
        """Fit exactly recovers quadratic function."""
        x = chebyshev_polynomial_t_points(5)
        y = x**2 - 0.5 * x + 1.0

        c = chebyshev_polynomial_t_fit(x, y, degree=2)
        y_fit = chebyshev_polynomial_t_evaluate(c, x)

        torch.testing.assert_close(y_fit, y, atol=1e-4, rtol=1e-4)

    def test_fit_overdetermined(self):
        """Overdetermined system (more points than degree)."""
        x = torch.linspace(-1, 1, 20)
        # True function: 1 + 2*T_1 + 3*T_2
        coeffs_true = torch.tensor([1.0, 2.0, 3.0])
        y = chebyshev_polynomial_t_evaluate(
            chebyshev_polynomial_t(coeffs_true), x
        )

        c = chebyshev_polynomial_t_fit(x, y, degree=2)

        torch.testing.assert_close(c, coeffs_true, atol=1e-5, rtol=1e-5)

    def test_fit_noisy_data(self):
        """Fit with noisy data."""
        torch.manual_seed(42)
        x = torch.linspace(-1, 1, 50)
        y_true = x**2
        y = y_true + 0.01 * torch.randn_like(y_true)

        c = chebyshev_polynomial_t_fit(x, y, degree=4)
        y_fit = chebyshev_polynomial_t_evaluate(c, x)

        # Residual should be small
        residual = (y_fit - y_true).abs().mean()
        assert residual < 0.05

    def test_fit_vs_numpy(self):
        """Compare with numpy.polynomial.chebyshev.chebfit."""
        x = np.linspace(-1, 1, 20)
        y = np.sin(np.pi * x)
        deg = 5

        c_torch = chebyshev_polynomial_t_fit(
            torch.tensor(x), torch.tensor(y), degree=deg
        )
        c_np = np_cheb.chebfit(x, y, deg)

        np.testing.assert_allclose(c_torch.numpy(), c_np, rtol=1e-5, atol=1e-5)


class TestChebyshevPolynomialTInterpolate:
    """Tests for chebyshev_polynomial_t_interpolate."""

    def test_interpolate_linear(self):
        """Interpolate linear function."""

        def f(x):
            return 2 * x + 3

        c = chebyshev_polynomial_t_interpolate(f, n=2)

        # Verify at Chebyshev points
        x = chebyshev_polynomial_t_points(10)
        y_interp = chebyshev_polynomial_t_evaluate(c, x)
        y_true = f(x)

        torch.testing.assert_close(y_interp, y_true, atol=1e-5, rtol=1e-5)

    def test_interpolate_polynomial(self):
        """Interpolate exactly recovers polynomial."""

        def f(x):
            return x**3 - 2 * x + 1

        c = chebyshev_polynomial_t_interpolate(f, n=4)

        x = torch.linspace(-1, 1, 20)
        y_interp = chebyshev_polynomial_t_evaluate(c, x)
        y_true = f(x)

        torch.testing.assert_close(y_interp, y_true, atol=1e-4, rtol=1e-4)

    def test_interpolate_sin(self):
        """Interpolate sin function."""

        def f(x):
            return torch.sin(math.pi * x)

        c = chebyshev_polynomial_t_interpolate(f, n=20)

        x = torch.linspace(-1, 1, 50)
        y_interp = chebyshev_polynomial_t_evaluate(c, x)
        y_true = f(x)

        # Should be very accurate for smooth function
        torch.testing.assert_close(y_interp, y_true, atol=1e-3, rtol=1e-3)

    def test_interpolate_at_nodes(self):
        """Interpolation is exact at Chebyshev nodes."""

        def f(x):
            return x**2 - x + 1

        n = 5
        c = chebyshev_polynomial_t_interpolate(f, n=n)

        x_nodes = chebyshev_polynomial_t_points(n)
        y_interp = chebyshev_polynomial_t_evaluate(c, x_nodes)
        y_true = f(x_nodes)

        torch.testing.assert_close(y_interp, y_true, atol=1e-6, rtol=1e-6)


class TestChebyshevPolynomialTFittingAutograd:
    """Tests for autograd support in fitting operations."""

    def test_vandermonde_gradcheck(self):
        """Gradcheck for Vandermonde."""
        x = torch.tensor(
            [0.0, 0.3, 0.7, 1.0], dtype=torch.float64, requires_grad=True
        )

        def fn(x_):
            return chebyshev_polynomial_t_vandermonde(x_, degree=3)

        assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)

    def test_fit_gradcheck(self):
        """Gradcheck for fit w.r.t. y."""
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(y_):
            return chebyshev_polynomial_t_fit(x, y_, degree=2)

        assert torch.autograd.gradcheck(fn, (y,), raise_exception=True)

    def test_fit_gradgradcheck(self):
        """Second-order gradients for fit."""
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.tensor(
            [1.0, 2.0, 1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True
        )

        def fn(y_):
            return chebyshev_polynomial_t_fit(x, y_, degree=2).sum()

        assert torch.autograd.gradgradcheck(fn, (y,), raise_exception=True)
