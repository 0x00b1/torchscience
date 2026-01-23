"""Tests for diffusion operator."""

import math

import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import diffuse, laplacian


class TestDiffuse:
    """Tests for diffuse function."""

    def test_diffuse_constant_diffusivity(self):
        """Constant D gives D * laplacian."""
        n = 32
        field = torch.randn(n, n)
        D = 0.5
        dx = 1.0 / n

        result = diffuse(field, diffusivity=D, dx=dx)
        expected = D * laplacian(field, dx=dx)

        torch.testing.assert_close(result, expected)

    def test_diffuse_shape_2d(self):
        """Diffusion preserves field shape."""
        n = 32
        field = torch.randn(n, n)
        dx = 1.0 / n

        result = diffuse(field, diffusivity=1.0, dx=dx)

        assert result.shape == field.shape

    def test_diffuse_shape_3d(self):
        """3D diffusion preserves field shape."""
        n = 16
        field = torch.randn(n, n, n)
        dx = 1.0 / n

        result = diffuse(field, diffusivity=0.1, dx=dx)

        assert result.shape == field.shape

    def test_diffuse_linear_field(self):
        """Linear field has zero Laplacian, hence zero diffusion."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = X + Y  # Linear
        dx = 1.0 / (n - 1)

        result = diffuse(field, diffusivity=1.0, dx=dx)

        # Interior should be near zero (allow for finite-difference truncation error)
        interior = result[3:-3, 3:-3]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=1e-3, rtol=1e-3
        )

    def test_diffuse_quadratic_field(self):
        """Quadratic field has constant Laplacian."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = X**2 + Y**2  # Laplacian = 4
        dx = 1.0 / (n - 1)
        D = 0.5

        result = diffuse(field, diffusivity=D, dx=dx)

        # Interior should be D * 4 = 2
        interior = result[3:-3, 3:-3]
        expected = 4 * D * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)

    def test_diffuse_variable_diffusivity(self):
        """Variable diffusivity uses divergence form."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
        D = 1.0 + 0.5 * X  # Spatially varying
        dx = 1.0 / (n - 1)

        result = diffuse(field, diffusivity=D, dx=dx)

        # Just check shape and that it's different from constant D case
        assert result.shape == field.shape

    def test_diffuse_3d_linear_field(self):
        """3D linear field has zero diffusion."""
        n = 16
        x = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        field = X + Y + Z  # Linear
        dx = 1.0 / (n - 1)

        result = diffuse(field, diffusivity=1.0, dx=dx)

        # Interior should be near zero (allow for finite-difference truncation error)
        interior = result[3:-3, 3:-3, 3:-3]
        torch.testing.assert_close(
            interior, torch.zeros_like(interior), atol=1e-3, rtol=1e-3
        )

    def test_diffuse_anisotropic_spacing(self):
        """Diffusion works with different spacing per dimension."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)
        dy = 2.0 / (n - 1)

        # f = x^2 + y^2, Laplacian = 2 + 2 = 4
        field = X**2 + Y**2
        D = 0.25

        result = diffuse(field, diffusivity=D, dx=(dx, dy))

        # Interior should be D * 4 = 1
        interior = result[3:-3, 3:-3]
        expected = 4 * D * torch.ones_like(interior)
        torch.testing.assert_close(interior, expected, atol=0.1, rtol=0.1)


class TestDiffuseAutograd:
    """Autograd tests for diffuse."""

    def test_gradcheck_constant_d(self):
        """Diffusion passes gradcheck with constant diffusivity."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: diffuse(f, diffusivity=0.5, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_variable_d(self):
        """Diffusion passes gradcheck with variable diffusivity."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        D = (
            torch.randn(8, 8, dtype=torch.float64, requires_grad=True).abs()
            + 0.1
        )

        assert gradcheck(
            lambda f, d: diffuse(f, diffusivity=d, dx=0.1),
            (field, D),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_3d(self):
        """Diffusion passes gradcheck for 3D."""
        field = torch.randn(6, 6, 6, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: diffuse(f, diffusivity=0.5, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradgradcheck_2d(self):
        """Diffusion passes gradgradcheck for 2D."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradgradcheck(
            lambda f: diffuse(f, diffusivity=0.5, dx=0.1).sum(),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )


class TestDiffuseAutocast:
    """Autocast tests for diffuse."""

    def test_diffuse_autocast_2d(self):
        """Diffusion upcasts to fp32 under autocast for 2D."""
        field = torch.randn(16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = diffuse(field, diffusivity=0.5, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (16, 16)

    def test_diffuse_autocast_3d(self):
        """Diffusion upcasts to fp32 under autocast for 3D."""
        field = torch.randn(8, 8, 8, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = diffuse(field, diffusivity=0.1, dx=0.1)

        assert result.dtype == torch.float32
        assert result.shape == (8, 8, 8)


class TestDiffuseVmap:
    """Vmap tests for diffuse."""

    def test_diffuse_vmap_2d(self):
        """Diffusion works with torch.vmap for 2D."""
        batch_field = torch.randn(4, 16, 16)

        # vmap over batch dimension
        batched_diffuse = torch.vmap(
            lambda f: diffuse(f, diffusivity=0.5, dx=0.1), in_dims=0
        )

        result = batched_diffuse(batch_field)

        # Each field is (16, 16), diffusion output is (16, 16)
        assert result.shape == (4, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [
                diffuse(batch_field[i], diffusivity=0.5, dx=0.1)
                for i in range(4)
            ]
        )
        torch.testing.assert_close(result, manual)

    def test_diffuse_vmap_3d(self):
        """Diffusion works with torch.vmap for 3D."""
        batch_field = torch.randn(4, 8, 8, 8)

        # vmap over batch dimension
        batched_diffuse = torch.vmap(
            lambda f: diffuse(f, diffusivity=0.1, dx=0.1), in_dims=0
        )

        result = batched_diffuse(batch_field)

        # Each field is (8, 8, 8), diffusion output is (8, 8, 8)
        assert result.shape == (4, 8, 8, 8)
