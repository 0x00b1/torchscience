"""Integration tests for differentiation module."""

import math

import pytest
import torch

from torchscience.differentiation import (
    apply_stencil,
    curl,
    derivative,
    divergence,
    finite_difference_stencil,
    gradient,
    laplacian,
)


class TestVectorCalculusIdentities:
    """Tests for vector calculus identities."""

    def test_curl_of_gradient_is_zero(self):
        """curl(grad(f)) = 0."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        f = (
            torch.sin(math.pi * X)
            * torch.sin(math.pi * Y)
            * torch.sin(math.pi * Z)
        )
        grad_f = gradient(f, dx=dx)
        curl_grad = curl(grad_f, dx=dx)

        torch.testing.assert_close(
            curl_grad[:, 3:-3, 3:-3, 3:-3],
            torch.zeros_like(curl_grad[:, 3:-3, 3:-3, 3:-3]),
            rtol=1e-1,
            atol=1e-2,
        )

    def test_divergence_of_curl_is_zero(self):
        """div(curl(V)) = 0."""
        n = 21
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        V = torch.stack(
            [
                torch.sin(math.pi * X) * Y,
                torch.cos(math.pi * Y) * Z,
                torch.sin(math.pi * Z) * X,
            ],
            dim=0,
        )
        curl_V = curl(V, dx=dx)
        div_curl = divergence(curl_V, dx=dx)

        torch.testing.assert_close(
            div_curl[3:-3, 3:-3, 3:-3],
            torch.zeros_like(div_curl[3:-3, 3:-3, 3:-3]),
            rtol=1e-1,
            atol=1e-2,
        )


class TestPDEEquations:
    """Tests using PDE solutions."""

    def test_laplacian_of_harmonic(self):
        """Laplacian of harmonic function is zero."""
        n = 31
        dx = 2.0 / (n - 1)
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        f = X**2 - Y**2  # harmonic: Re(z^2)
        lap = laplacian(f, dx=dx)

        torch.testing.assert_close(
            lap[2:-2, 2:-2],
            torch.zeros_like(lap[2:-2, 2:-2]),
            rtol=1e-2,
            atol=1e-4,
        )

    def test_poisson_equation(self):
        """Verify Poisson equation for known solution."""
        n = 31
        dx = 1.0 / (n - 1)
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        u = torch.sin(math.pi * X) * torch.sin(math.pi * Y)
        expected_lap = -2 * (math.pi**2) * u
        lap = laplacian(u, dx=dx, accuracy=4)

        torch.testing.assert_close(
            lap[4:-4, 4:-4], expected_lap[4:-4, 4:-4], rtol=5e-2, atol=1e-2
        )


class TestAutogradIntegration:
    """Tests for autograd compatibility."""

    def test_derivative_gradients(self):
        """Gradients flow through derivative computation."""
        x = torch.linspace(0, 1, 21, requires_grad=True)
        f = x**3
        df = derivative(f, dim=0, order=1, dx=0.05)
        loss = df.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_laplacian_gradients(self):
        """Gradients flow through Laplacian computation."""
        f = torch.randn(10, 10, requires_grad=True)
        lap = laplacian(f, dx=0.1)
        loss = lap.sum()
        loss.backward()

        assert f.grad is not None
        assert not torch.isnan(f.grad).any()

    def test_gradient_op_gradients(self):
        """Gradients flow through gradient computation."""
        f = torch.randn(10, 10, requires_grad=True)
        grad_f = gradient(f, dx=0.1)
        loss = grad_f.sum()
        loss.backward()

        assert f.grad is not None
        assert not torch.isnan(f.grad).any()


class TestDtypeSupport:
    """Tests for various dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_derivative_dtypes(self, dtype):
        """Derivative works with different dtypes."""
        f = torch.randn(20, dtype=dtype)
        df = derivative(f, dim=0, order=1, dx=0.1)
        assert df.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_laplacian_dtypes(self, dtype):
        """Laplacian works with different dtypes."""
        f = torch.randn(10, 10, dtype=dtype)
        lap = laplacian(f, dx=0.1)
        assert lap.dtype == dtype


class TestBoundaryModes:
    """Tests for boundary mode behavior."""

    @pytest.mark.parametrize(
        "boundary", ["replicate", "zeros", "reflect", "circular"]
    )
    def test_boundary_modes_1d(self, boundary):
        """All boundary modes work for 1D."""
        f = torch.randn(20)
        df = derivative(f, dim=0, order=1, dx=0.1, boundary=boundary)
        assert df.shape == f.shape

    @pytest.mark.parametrize(
        "boundary", ["replicate", "zeros", "reflect", "circular"]
    )
    def test_boundary_modes_2d(self, boundary):
        """All boundary modes work for 2D."""
        f = torch.randn(10, 10)
        lap = laplacian(f, dx=0.1, boundary=boundary)
        assert lap.shape == f.shape

    def test_valid_mode_reduces_size(self):
        """Valid mode produces smaller output."""
        f = torch.randn(20)
        stencil = finite_difference_stencil(derivative=2, accuracy=2)
        df = apply_stencil(stencil, f, boundary="valid")
        assert df.shape[0] < f.shape[0]
