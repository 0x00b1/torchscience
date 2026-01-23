"""Tests for spectral differentiation operators."""

import math

import torch

from torchscience.differentiation import spectral_derivative, spectral_gradient


class TestSpectralDerivative:
    """Tests for spectral_derivative function."""

    def test_derivative_of_sine(self):
        """Spectral derivative of sin(2*pi*x) is 2*pi*cos(2*pi*x)."""
        n = 64
        x = torch.linspace(0, 1, n + 1)[
            :-1
        ]  # Exclude endpoint for periodicity
        dx = 1.0 / n

        f = torch.sin(2 * math.pi * x)
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        df = spectral_derivative(f, dim=0, dx=dx)

        torch.testing.assert_close(df, expected, rtol=1e-5, atol=1e-5)

    def test_derivative_of_cosine(self):
        """Spectral derivative of cos(2*pi*x) is -2*pi*sin(2*pi*x)."""
        n = 64
        x = torch.linspace(0, 1, n + 1)[:-1]
        dx = 1.0 / n

        f = torch.cos(2 * math.pi * x)
        expected = -2 * math.pi * torch.sin(2 * math.pi * x)

        df = spectral_derivative(f, dim=0, dx=dx)

        torch.testing.assert_close(df, expected, rtol=1e-5, atol=1e-5)

    def test_second_derivative_of_sine(self):
        """Second spectral derivative of sin(2*pi*x) is -(2*pi)^2*sin(2*pi*x)."""
        n = 64
        # Use float64 for second derivative to achieve machine precision
        x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
        dx = 1.0 / n

        f = torch.sin(2 * math.pi * x)
        expected = -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * x)

        d2f = spectral_derivative(f, dim=0, order=2, dx=dx)

        torch.testing.assert_close(d2f, expected, rtol=1e-10, atol=1e-10)

    def test_2d_derivative(self):
        """Spectral derivative in 2D."""
        n = 32
        x = torch.linspace(0, 1, n + 1)[:-1]
        y = torch.linspace(0, 1, n + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / n

        # f = sin(2*pi*x) * cos(2*pi*y)
        f = torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * Y)

        # df/dx = 2*pi*cos(2*pi*x)*cos(2*pi*y)
        expected_dx = (
            2
            * math.pi
            * torch.cos(2 * math.pi * X)
            * torch.cos(2 * math.pi * Y)
        )

        df_dx = spectral_derivative(f, dim=0, dx=dx)

        torch.testing.assert_close(df_dx, expected_dx, rtol=1e-4, atol=1e-4)

    def test_preserves_dtype(self):
        """Spectral derivative preserves float64 precision."""
        n = 64
        x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
        dx = 1.0 / n
        f = torch.sin(2 * math.pi * x)

        df = spectral_derivative(f, dim=0, dx=dx)

        assert df.dtype == torch.float64


class TestSpectralGradient:
    """Tests for spectral_gradient function."""

    def test_gradient_of_product(self):
        """Gradient of sin(2*pi*x)*cos(2*pi*y)."""
        n = 32
        x = torch.linspace(0, 1, n + 1)[:-1]
        y = torch.linspace(0, 1, n + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / n

        f = torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * Y)

        grad = spectral_gradient(f, dx=dx)

        # df/dx = 2*pi*cos(2*pi*x)*cos(2*pi*y)
        expected_dx = (
            2
            * math.pi
            * torch.cos(2 * math.pi * X)
            * torch.cos(2 * math.pi * Y)
        )
        # df/dy = -2*pi*sin(2*pi*x)*sin(2*pi*y)
        expected_dy = (
            -2
            * math.pi
            * torch.sin(2 * math.pi * X)
            * torch.sin(2 * math.pi * Y)
        )

        assert grad.shape == (2, n, n)
        torch.testing.assert_close(grad[0], expected_dx, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(grad[1], expected_dy, rtol=1e-4, atol=1e-4)

    def test_gradient_3d(self):
        """Gradient in 3D returns 3 components."""
        n = 16
        field = torch.randn(n, n, n)
        dx = 1.0 / n

        grad = spectral_gradient(field, dx=dx)

        assert grad.shape == (3, n, n, n)

    def test_gradient_1d(self):
        """1D gradient returns 1 component."""
        n = 64
        x = torch.linspace(0, 1, n + 1)[:-1]
        dx = 1.0 / n
        f = torch.sin(2 * math.pi * x)

        grad = spectral_gradient(f, dx=dx)

        assert grad.shape == (1, n)
        # Should equal spectral_derivative
        expected = spectral_derivative(f, dim=0, dx=dx)
        torch.testing.assert_close(grad[0], expected, rtol=1e-5, atol=1e-5)
