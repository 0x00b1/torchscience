"""Tests for Chebyshev derivative operator."""

import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import chebyshev_derivative, chebyshev_points


class TestChebyshevPoints:
    """Tests for chebyshev_points function."""

    def test_chebyshev_points_n5(self):
        """Chebyshev-Gauss-Lobatto points for N=5."""
        pts = chebyshev_points(5)
        # x_j = cos(pi*j/N) for j = 0, ..., N
        expected = torch.tensor(
            [
                1.0,
                math.cos(math.pi / 5),
                math.cos(2 * math.pi / 5),
                math.cos(3 * math.pi / 5),
                math.cos(4 * math.pi / 5),
                -1.0,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(pts, expected, atol=1e-6, rtol=1e-6)

    def test_chebyshev_points_endpoints(self):
        """Chebyshev points include -1 and 1."""
        for n in [4, 8, 16]:
            pts = chebyshev_points(n)
            assert pts[0].item() == pytest.approx(1.0)
            assert pts[-1].item() == pytest.approx(-1.0)

    def test_chebyshev_points_symmetric(self):
        """Chebyshev points are symmetric about 0."""
        pts = chebyshev_points(10)
        torch.testing.assert_close(pts, -pts.flip(0), atol=1e-6, rtol=1e-6)


class TestChebyshevDerivative:
    """Tests for chebyshev_derivative function."""

    def test_derivative_of_polynomial(self):
        """Derivative of x^2 is 2x."""
        n = 16
        x = chebyshev_points(n, dtype=torch.float64)
        f = x**2

        df = chebyshev_derivative(f, dim=0)
        expected = 2 * x

        torch.testing.assert_close(df, expected, atol=1e-10, rtol=1e-10)

    def test_derivative_of_sine(self):
        """Derivative of sin(pi*x/2) on [-1, 1]."""
        n = 32
        x = chebyshev_points(n, dtype=torch.float64)
        f = torch.sin(math.pi * x / 2)

        df = chebyshev_derivative(f, dim=0)
        expected = (math.pi / 2) * torch.cos(math.pi * x / 2)

        torch.testing.assert_close(df, expected, atol=1e-8, rtol=1e-8)

    def test_second_derivative(self):
        """Second derivative of sin(x) is -sin(x)."""
        n = 32
        x = chebyshev_points(n, dtype=torch.float64)
        f = torch.sin(x)

        d2f = chebyshev_derivative(f, dim=0, order=2)
        expected = -torch.sin(x)

        torch.testing.assert_close(d2f, expected, atol=1e-6, rtol=1e-6)

    def test_2d_derivative(self):
        """Chebyshev derivative along one dimension of 2D field."""
        n = 16
        x = chebyshev_points(n, dtype=torch.float64)
        y = chebyshev_points(n, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # f = x^2 * y
        f = X**2 * Y

        # df/dx = 2xy
        df_dx = chebyshev_derivative(f, dim=0)
        expected = 2 * X * Y

        torch.testing.assert_close(df_dx, expected, atol=1e-8, rtol=1e-8)

    def test_preserves_dtype(self):
        """Chebyshev derivative preserves float64."""
        n = 16
        x = chebyshev_points(n, dtype=torch.float64)
        f = x**3

        df = chebyshev_derivative(f, dim=0)

        assert df.dtype == torch.float64


class TestChebyshevDerivativeAutograd:
    """Autograd tests for chebyshev_derivative."""

    def test_gradcheck(self):
        """Chebyshev derivative passes gradcheck."""
        n = 8
        f = torch.randn(n + 1, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda x: chebyshev_derivative(x, dim=0),
            (f,),
        )

    def test_gradgradcheck(self):
        """Chebyshev derivative passes gradgradcheck."""
        n = 8
        f = torch.randn(n + 1, dtype=torch.float64, requires_grad=True)

        assert gradgradcheck(
            lambda x: chebyshev_derivative(x, dim=0),
            (f,),
        )
