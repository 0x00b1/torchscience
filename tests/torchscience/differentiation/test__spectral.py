"""Tests for spectral differentiation operators."""

import math

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import (
    spectral_derivative,
    spectral_gradient,
    spectral_laplacian,
)


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


class TestSpectralLaplacian:
    """Tests for spectral_laplacian function."""

    def test_sin_2d(self):
        """Laplacian of sin(x)*sin(y) is -2*sin(x)*sin(y)."""
        n = 64
        x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]  # Periodic domain
        y = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")

        f = torch.sin(X) * torch.sin(Y)
        dx = 2 * math.pi / n

        lap = spectral_laplacian(f, spacing=dx)
        expected = -2 * torch.sin(X) * torch.sin(Y)

        torch.testing.assert_close(lap, expected, rtol=1e-3, atol=1e-3)

    def test_cos_3d(self):
        """Laplacian of cos(x)*cos(y)*cos(z) is -3*cos(x)*cos(y)*cos(z)."""
        n = 32
        x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        y = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        z = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        f = torch.cos(X) * torch.cos(Y) * torch.cos(Z)
        dx = 2 * math.pi / n

        lap = spectral_laplacian(f, spacing=dx)
        expected = -3 * torch.cos(X) * torch.cos(Y) * torch.cos(Z)

        torch.testing.assert_close(lap, expected, rtol=1e-3, atol=1e-3)

    def test_partial_dims(self):
        """Laplacian over subset of dimensions."""
        n = 64
        x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        y = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")

        f = torch.sin(X) * torch.sin(Y)
        dx = 2 * math.pi / n

        # Laplacian only in x direction: d^2/dx^2[sin(x)sin(y)] = -sin(x)sin(y)
        lap_x = spectral_laplacian(f, dims=[0], spacing=dx)
        expected_x = -torch.sin(X) * torch.sin(Y)

        torch.testing.assert_close(lap_x, expected_x, rtol=1e-3, atol=1e-3)

    def test_1d_second_derivative(self):
        """1D Laplacian equals second derivative."""
        n = 64
        x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        dx = 2 * math.pi / n

        # f = sin(x), d^2f/dx^2 = -sin(x)
        f = torch.sin(x)
        lap = spectral_laplacian(f, spacing=dx)
        expected = -torch.sin(x)

        torch.testing.assert_close(lap, expected, rtol=1e-3, atol=1e-3)

    def test_different_spacings(self):
        """Laplacian with different spacing per dimension."""
        n = 64
        # Domain [0, 2*pi] x [0, 4*pi]
        x = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        y = torch.linspace(0, 4 * math.pi, n + 1)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # f = sin(x) * sin(y/2), so Laplacian = -sin(x)*sin(y/2) - (1/4)*sin(x)*sin(y/2)
        #                                     = -(5/4)*sin(x)*sin(y/2)
        f = torch.sin(X) * torch.sin(Y / 2)
        dx = 2 * math.pi / n
        dy = 4 * math.pi / n

        lap = spectral_laplacian(f, spacing=[dx, dy])
        expected = -(1 + 0.25) * torch.sin(X) * torch.sin(Y / 2)

        torch.testing.assert_close(lap, expected, rtol=1e-3, atol=1e-3)

    def test_preserves_shape(self):
        """Output has same shape as input."""
        field = torch.randn(16, 32, 24)
        lap = spectral_laplacian(field)
        assert lap.shape == field.shape

    def test_preserves_dtype(self):
        """Output has same dtype as input."""
        field = torch.randn(32, 32, dtype=torch.float64)
        lap = spectral_laplacian(field)
        assert lap.dtype == torch.float64


class TestSpectralVsFiniteDifference:
    """Compare spectral and finite difference accuracy."""

    def test_derivative_accuracy_comparison(self):
        """Spectral derivative should be more accurate than FD for smooth periodic functions."""

        n = 64
        # Use float64 for high-precision comparison
        x = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        dx = 2 * math.pi / n

        # sin(x) -> cos(x)
        f = torch.sin(x)
        expected = torch.cos(x)

        # Spectral derivative (using impl directly to avoid autocast)
        from torchscience.differentiation._spectral_derivative import (
            _spectral_derivative_impl,
        )

        spectral_result = _spectral_derivative_impl(f, dim=0, dx=dx)
        spectral_error = (spectral_result - expected).abs().max().item()

        # Finite difference derivative with circular boundary for fair comparison
        from torchscience.differentiation._derivative import _derivative_impl

        fd_result = _derivative_impl(
            f, dim=0, dx=dx, accuracy=2, boundary="circular"
        )
        fd_error = (fd_result - expected).abs().max().item()

        # Spectral should be significantly more accurate
        assert spectral_error < fd_error, (
            f"Spectral error {spectral_error} should be less than FD error {fd_error}"
        )
        # Spectral should achieve near-machine precision for this simple case
        assert spectral_error < 1e-10, (
            f"Spectral error {spectral_error} should be near machine precision"
        )

    def test_gradient_accuracy_comparison(self):
        """Spectral gradient should be more accurate than FD for smooth periodic functions."""
        n = 32
        # Use float64 for high-precision comparison
        x = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        y = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2 * math.pi / n

        # sin(x)*sin(y)
        f = torch.sin(X) * torch.sin(Y)
        expected_dx = torch.cos(X) * torch.sin(Y)
        expected_dy = torch.sin(X) * torch.cos(Y)

        # Spectral gradient (using impl directly to avoid autocast)
        from torchscience.differentiation._spectral_gradient import (
            _spectral_gradient_impl,
        )

        spectral_grad = _spectral_gradient_impl(f, dx=dx)
        spectral_gx, spectral_gy = spectral_grad[0], spectral_grad[1]
        spectral_error_x = (spectral_gx - expected_dx).abs().max().item()
        spectral_error_y = (spectral_gy - expected_dy).abs().max().item()
        spectral_error = max(spectral_error_x, spectral_error_y)

        # Finite difference gradient with circular boundary for fair comparison
        from torchscience.differentiation._gradient import _gradient_impl

        fd_grad = _gradient_impl(f, dx=dx, accuracy=2, boundary="circular")
        fd_gx = fd_grad[0]
        fd_gy = fd_grad[1]
        fd_error_x = (fd_gx - expected_dx).abs().max().item()
        fd_error_y = (fd_gy - expected_dy).abs().max().item()
        fd_error = max(fd_error_x, fd_error_y)

        # Spectral should be more accurate
        assert spectral_error < fd_error, (
            f"Spectral error {spectral_error} should be less than FD error {fd_error}"
        )

    def test_laplacian_accuracy_comparison(self):
        """Spectral Laplacian should be more accurate than FD for smooth periodic functions."""
        n = 32
        # Use float64 for high-precision comparison
        x = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        y = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2 * math.pi / n

        # sin(x)*sin(y) -> Laplacian = -2*sin(x)*sin(y)
        f = torch.sin(X) * torch.sin(Y)
        expected = -2 * torch.sin(X) * torch.sin(Y)

        # Spectral Laplacian (using impl directly to avoid autocast)
        from torchscience.differentiation._spectral_laplacian import (
            _spectral_laplacian_impl,
        )

        spectral_result = _spectral_laplacian_impl(f, spacing=dx)
        spectral_error = (spectral_result - expected).abs().max().item()

        # Finite difference Laplacian with circular boundary for fair comparison
        from torchscience.differentiation._laplacian import _laplacian_impl

        fd_result = _laplacian_impl(f, dx=dx, accuracy=2, boundary="circular")
        fd_error = (fd_result - expected).abs().max().item()

        # Spectral should be more accurate
        assert spectral_error < fd_error, (
            f"Spectral error {spectral_error} should be less than FD error {fd_error}"
        )
        # Spectral should achieve high accuracy
        assert spectral_error < 1e-8, (
            f"Spectral error {spectral_error} should be very small"
        )


class TestSpectralAutograd:
    """Autograd tests for spectral operators."""

    def test_spectral_derivative_gradcheck(self):
        """Verify gradients for spectral_derivative."""
        n = 16
        x = torch.randn(n, dtype=torch.float64, requires_grad=True)

        def func(field):
            return spectral_derivative(field, dim=0)

        assert gradcheck(func, (x,), raise_exception=True)

    def test_spectral_derivative_gradgradcheck(self):
        """Verify second-order gradients for spectral_derivative."""
        n = 16
        x = torch.randn(n, dtype=torch.float64, requires_grad=True)

        def func(field):
            return spectral_derivative(field, dim=0)

        assert gradgradcheck(func, (x,), raise_exception=True)

    def test_spectral_gradient_gradcheck(self):
        """Verify gradients for spectral_gradient."""
        n = 8
        field = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

        def func(f):
            # spectral_gradient returns stacked tensor, sum for scalar output
            grad = spectral_gradient(f)
            return grad.sum(dim=0)

        assert gradcheck(func, (field,), raise_exception=True)

    def test_spectral_gradient_gradgradcheck(self):
        """Verify second-order gradients for spectral_gradient."""
        n = 8
        field = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

        def func(f):
            grad = spectral_gradient(f)
            return grad.sum(dim=0)

        assert gradgradcheck(func, (field,), raise_exception=True)

    def test_spectral_laplacian_gradcheck(self):
        """Verify gradients for spectral_laplacian."""
        n = 8
        field = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

        def func(f):
            return spectral_laplacian(f)

        assert gradcheck(func, (field,), raise_exception=True)

    def test_spectral_laplacian_gradgradcheck(self):
        """Verify second-order gradients for spectral_laplacian."""
        n = 8
        field = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

        def func(f):
            return spectral_laplacian(f)

        assert gradgradcheck(func, (field,), raise_exception=True)
