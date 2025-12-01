"""Tests for scalar field differential operators."""

import pytest
import torch

from torchscience.differentiation import (
    derivative,
    gradient,
    hessian,
    laplacian,
)


class TestDerivative:
    """Tests for derivative function."""

    def test_first_derivative(self):
        """First derivative of x^2 is 2x."""
        x = torch.linspace(0, 1, 21)
        f = x**2
        df = derivative(f, dim=0, order=1, dx=0.05)
        expected = 2 * x
        torch.testing.assert_close(
            df[2:-2], expected[2:-2], rtol=1e-2, atol=1e-4
        )

    def test_second_derivative(self):
        """Second derivative of x^2 is 2."""
        x = torch.linspace(0, 1, 21)
        f = x**2
        d2f = derivative(f, dim=0, order=2, dx=0.05)
        torch.testing.assert_close(
            d2f[2:-2], torch.full_like(d2f[2:-2], 2.0), rtol=1e-2, atol=1e-4
        )

    def test_multidimensional(self):
        """Derivative along specific dimension."""
        f = torch.randn(10, 20, 30)
        df = derivative(f, dim=1, order=1, dx=0.1)
        assert df.shape == f.shape


class TestGradient:
    """Tests for gradient function."""

    def test_gradient_of_linear(self):
        """Gradient of a*x + b*y is (a, b)."""
        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = 3 * X + 2 * Y  # grad = (3, 2)

        grad = gradient(f, dx=(0.05, 0.05))

        # Shape should be (2, n, n)
        assert grad.shape == (2, n, n)

        # Interior values
        torch.testing.assert_close(
            grad[0, 2:-2, 2:-2],
            torch.full((n - 4, n - 4), 3.0),
            rtol=1e-2,
            atol=1e-4,
        )
        torch.testing.assert_close(
            grad[1, 2:-2, 2:-2],
            torch.full((n - 4, n - 4), 2.0),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_gradient_shape(self):
        """Gradient adds dimension for components."""
        f = torch.randn(10, 20, 30)
        grad = gradient(f, dim=(-2, -1))  # 2D gradient of last two dims
        assert grad.shape == (10, 2, 20, 30)


class TestLaplacian:
    """Tests for Laplacian function."""

    def test_laplacian_of_quadratic(self):
        """Laplacian of x^2 + y^2 is 4."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = X**2 + Y**2

        lap = laplacian(f, dx=dx)

        assert lap.shape == f.shape
        torch.testing.assert_close(
            lap[2:-2, 2:-2],
            torch.full((n - 4, n - 4), 4.0),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_laplacian_3d(self):
        """3D Laplacian works."""
        f = torch.randn(10, 10, 10)
        lap = laplacian(f, dx=0.1)
        assert lap.shape == f.shape


class TestHessian:
    """Tests for Hessian function."""

    def test_hessian_of_quadratic(self):
        """Hessian of ax^2 + bxy + cy^2 is [[2a, b], [b, 2c]]."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = 2 * X**2 + 3 * X * Y + 4 * Y**2

        H = hessian(f, dx=dx)

        # Shape should be (2, 2, n, n)
        assert H.shape == (2, 2, n, n)

        # Interior values
        torch.testing.assert_close(
            H[0, 0, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 4.0),
            rtol=0.1,
            atol=0.1,
        )  # d^2f/dx^2 = 4
        torch.testing.assert_close(
            H[1, 1, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 8.0),
            rtol=0.1,
            atol=0.1,
        )  # d^2f/dy^2 = 8
        torch.testing.assert_close(
            H[0, 1, 3:-3, 3:-3],
            torch.full((n - 6, n - 6), 3.0),
            rtol=0.1,
            atol=0.1,
        )  # d^2f/dxdy = 3

    def test_hessian_symmetric(self):
        """Hessian is symmetric."""
        f = torch.randn(20, 20)
        H = hessian(f, dx=0.1)
        torch.testing.assert_close(H[0, 1], H[1, 0])


class TestNumpyComparison:
    """Tests comparing against numpy.gradient."""

    @pytest.fixture
    def numpy_available(self):
        """Check if numpy is available."""
        pytest.importorskip("numpy")
        return True

    def test_gradient_matches_numpy_1d(self, numpy_available):
        """1D gradient matches numpy.gradient."""
        import numpy as np

        x = torch.linspace(0, 1, 21)
        f = torch.sin(2 * 3.14159 * x)
        dx = (x[1] - x[0]).item()

        our_grad = derivative(f, dim=0, order=1, dx=dx, accuracy=2)
        np_grad = np.gradient(f.numpy(), dx)

        # Compare interior (boundary handling may differ)
        torch.testing.assert_close(
            our_grad[2:-2],
            torch.tensor(np_grad[2:-2], dtype=f.dtype),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_gradient_matches_numpy_2d(self, numpy_available):
        """2D gradient matches numpy.gradient."""
        import numpy as np

        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = X**2 + Y**2
        dx = (x[1] - x[0]).item()

        our_grad = gradient(f, dx=dx, accuracy=2)
        np_grad = np.gradient(f.numpy(), dx)

        # Compare x-gradient
        torch.testing.assert_close(
            our_grad[0, 2:-2, 2:-2],
            torch.tensor(np_grad[0][2:-2, 2:-2], dtype=f.dtype),
            rtol=1e-2,
            atol=1e-4,
        )

        # Compare y-gradient
        torch.testing.assert_close(
            our_grad[1, 2:-2, 2:-2],
            torch.tensor(np_grad[1][2:-2, 2:-2], dtype=f.dtype),
            rtol=1e-2,
            atol=1e-4,
        )
