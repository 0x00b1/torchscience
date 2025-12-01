"""Tests for stencil application."""

import math

import torch

from torchscience.differentiation import (
    apply_stencil,
    finite_difference_stencil,
)


class TestApplyStencil1D:
    """Tests for 1D stencil application."""

    def test_second_derivative_of_quadratic(self):
        """Second derivative of x^2 should be 2."""
        x = torch.linspace(0, 1, 11)  # dx = 0.1
        f = x**2

        stencil = finite_difference_stencil(derivative=2, accuracy=2)
        result = apply_stencil(stencil, f, dx=0.1)

        # Interior points should be close to 2
        torch.testing.assert_close(
            result[1:-1],
            torch.full_like(result[1:-1], 2.0),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_first_derivative_of_sine(self):
        """First derivative of sin(x) should be cos(x)."""
        x = torch.linspace(0, 2 * math.pi, 101)
        dx = x[1] - x[0]
        f = torch.sin(x)

        stencil = finite_difference_stencil(derivative=1, accuracy=4)
        result = apply_stencil(stencil, f, dx=dx.item())

        expected = torch.cos(x)
        torch.testing.assert_close(
            result[2:-2], expected[2:-2], rtol=1e-3, atol=1e-5
        )

    def test_batched_input(self):
        """Handles batched input."""
        x = torch.linspace(0, 1, 11)
        f = torch.stack([x**2, x**3], dim=0)  # (2, 11)

        stencil = finite_difference_stencil(derivative=1, accuracy=2)
        result = apply_stencil(stencil, f, dx=0.1)

        assert result.shape == (2, 11)

    def test_different_boundary_modes(self):
        """Different boundary modes produce different shapes."""
        f = torch.randn(20)
        stencil = finite_difference_stencil(derivative=2, accuracy=2)

        result_replicate = apply_stencil(stencil, f, boundary="replicate")
        result_valid = apply_stencil(stencil, f, boundary="valid")

        assert result_replicate.shape == f.shape  # Same size
        assert result_valid.shape[0] < f.shape[0]  # Smaller


class TestApplyStencil2D:
    """Tests for 2D stencil application."""

    def test_laplacian_of_quadratic(self):
        """Laplacian of x^2 + y^2 should be 4."""
        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = X**2 + Y**2  # Laplacian = 2 + 2 = 4

        stencil_xx = finite_difference_stencil(derivative=(2, 0), accuracy=2)
        stencil_yy = finite_difference_stencil(derivative=(0, 2), accuracy=2)

        dx = 1.0 / (n - 1)
        laplacian = apply_stencil(stencil_xx, f, dx=dx) + apply_stencil(
            stencil_yy, f, dx=dx
        )

        torch.testing.assert_close(
            laplacian[2:-2, 2:-2],
            torch.full_like(laplacian[2:-2, 2:-2], 4.0),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_mixed_partial(self):
        """Mixed partial d^2/dxdy of x*y should be 1."""
        n = 21
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        f = X * Y

        stencil = finite_difference_stencil(derivative=(1, 1), accuracy=2)
        dx = 1.0 / (n - 1)
        result = apply_stencil(stencil, f, dx=dx)

        torch.testing.assert_close(
            result[2:-2, 2:-2],
            torch.full_like(result[2:-2, 2:-2], 1.0),
            rtol=1e-2,
            atol=1e-4,
        )


class TestApplyStencilDx:
    """Tests for grid spacing handling."""

    def test_scalar_dx(self):
        """Scalar dx works for n-D."""
        f = torch.randn(10, 10)
        stencil = finite_difference_stencil(derivative=(2, 0), accuracy=2)
        result = apply_stencil(stencil, f, dx=0.1)
        assert result.shape == f.shape

    def test_tuple_dx(self):
        """Per-dimension dx tuple."""
        f = torch.randn(10, 20)
        stencil = finite_difference_stencil(derivative=(1, 1), accuracy=2)
        result = apply_stencil(stencil, f, dx=(0.1, 0.05))
        assert result.shape == f.shape


class TestApplyStencilAutograd:
    """Tests for autograd compatibility."""

    def test_gradients_flow(self):
        """Gradients flow through stencil application."""
        f = torch.randn(20, requires_grad=True)
        stencil = finite_difference_stencil(derivative=2, accuracy=2)
        result = apply_stencil(stencil, f, dx=0.1)

        loss = result.sum()
        loss.backward()

        assert f.grad is not None
        assert f.grad.shape == f.shape
