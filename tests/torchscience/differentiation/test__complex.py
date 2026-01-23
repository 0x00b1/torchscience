"""Tests for complex tensor support in differentiation operators."""

import torch

from torchscience.differentiation import derivative, gradient, laplacian


class TestComplexGradient:
    """Tests for gradient with complex tensors."""

    def test_gradient_complex_linear(self):
        """Gradient of complex linear function."""
        # f(z) = (3+2i)*x + (1+4i)*y
        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        field = (3 + 2j) * X + (1 + 4j) * Y
        field = field.to(torch.complex64)

        grad = gradient(field, dx=0.05)

        # Gradient should preserve complex coefficients
        assert grad.dtype == torch.complex64
        assert grad.shape == (2, n, n)

        # df/dx = 3+2i, df/dy = 1+4i
        torch.testing.assert_close(
            grad[0, 2:-2, 2:-2].real,
            torch.full((n - 4, n - 4), 3.0),
            rtol=0.05,
            atol=0.05,
        )
        torch.testing.assert_close(
            grad[0, 2:-2, 2:-2].imag,
            torch.full((n - 4, n - 4), 2.0),
            rtol=0.05,
            atol=0.05,
        )
        torch.testing.assert_close(
            grad[1, 2:-2, 2:-2].real,
            torch.full((n - 4, n - 4), 1.0),
            rtol=0.05,
            atol=0.05,
        )
        torch.testing.assert_close(
            grad[1, 2:-2, 2:-2].imag,
            torch.full((n - 4, n - 4), 4.0),
            rtol=0.05,
            atol=0.05,
        )


class TestComplexLaplacian:
    """Tests for laplacian with complex tensors."""

    def test_laplacian_complex_quadratic(self):
        """Laplacian of complex quadratic function."""
        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # f = (1+i)*x^2 + (2-i)*y^2
        # d^2f/dx^2 = 2*(1+i), d^2f/dy^2 = 2*(2-i)
        # laplacian f = 2*(1+i) + 2*(2-i) = 2 + 2i + 4 - 2i = 6
        field = (1 + 1j) * X**2 + (2 - 1j) * Y**2
        field = field.to(torch.complex64)

        lap = laplacian(field, dx=0.05)

        assert lap.dtype == torch.complex64
        # Interior should be approximately 6+0i
        torch.testing.assert_close(
            lap[4:-4, 4:-4].real,
            torch.full((n - 8, n - 8), 6.0),
            rtol=0.1,
            atol=0.2,
        )
        torch.testing.assert_close(
            lap[4:-4, 4:-4].imag,
            torch.full((n - 8, n - 8), 0.0),
            rtol=0.1,
            atol=0.2,
        )


class TestComplexDerivative:
    """Tests for derivative with complex tensors."""

    def test_derivative_complex_1d(self):
        """Derivative of complex 1D function."""
        n = 51
        x = torch.linspace(0, 1, n)
        # f(x) = (2+3i)*x
        field = (2 + 3j) * x
        field = field.to(torch.complex64)

        deriv = derivative(field, dim=0, dx=0.02)

        assert deriv.dtype == torch.complex64
        # df/dx = 2+3i
        torch.testing.assert_close(
            deriv[2:-2].real,
            torch.full((n - 4,), 2.0),
            rtol=0.05,
            atol=0.05,
        )
        torch.testing.assert_close(
            deriv[2:-2].imag,
            torch.full((n - 4,), 3.0),
            rtol=0.05,
            atol=0.05,
        )


class TestComplexAutograd:
    """Tests for autograd with complex tensors."""

    def test_complex_gradient_backward(self):
        """Complex gradient supports backward pass."""
        field = torch.randn(8, 8, dtype=torch.complex64, requires_grad=True)

        grad = gradient(field, dx=0.1)
        loss = grad.abs().sum()
        loss.backward()

        assert field.grad is not None
        assert field.grad.shape == field.shape
