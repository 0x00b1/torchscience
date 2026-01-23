"""Tests for volume integral operator."""

import math

import torch
from torch.autograd import gradcheck

from torchscience.differentiation import volume_integral


class TestVolumeIntegral:
    """Tests for volume_integral function."""

    def test_volume_integral_constant(self):
        """Integral of constant over unit cube equals constant."""
        n = 32
        field = 3.0 * torch.ones(n, n, n)
        # Use n points to cover a unit cube: each cell has volume (1/n)^3
        dx = 1.0 / n

        result = volume_integral(field, spacing=dx)

        # ∭ 3 dV over [0,1]³ = 3
        expected = torch.tensor(3.0)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)

    def test_volume_integral_2d(self):
        """Volume integral works in 2D (area integral)."""
        n = 32
        field = 2.0 * torch.ones(n, n)
        dx = 1.0 / (n - 1)

        result = volume_integral(field, spacing=dx)

        # ∬ 2 dA over [0,1]² = 2
        expected = torch.tensor(2.0)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)

    def test_volume_integral_returns_scalar(self):
        """Volume integral returns a scalar."""
        field = torch.randn(8, 8, 8)

        result = volume_integral(field, spacing=0.1)

        assert result.ndim == 0

    def test_volume_integral_linear_field(self):
        """Integral of linear field."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        dx = 1.0 / (n - 1)

        # f = x + y + z
        # ∭ (x + y + z) dV over [0,1]³ = 3 * (1/2) = 1.5
        field = X + Y + Z

        result = volume_integral(field, spacing=dx)

        expected = torch.tensor(1.5)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)

    def test_volume_integral_with_mask(self):
        """Integral over a subregion using mask."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Constant field
        field = torch.ones(n, n)

        # Mask for circle of radius 0.5 centered at (0.5, 0.5)
        mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2) <= 0.25

        result = volume_integral(field, spacing=dx, region=mask)

        # Area of circle = π * r² = π * 0.25 ≈ 0.785
        expected = torch.tensor(math.pi * 0.25)
        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.1)

    def test_volume_integral_anisotropic_spacing(self):
        """Integral with different spacing per dimension."""
        n = 16
        field = torch.ones(n, n, n)
        spacing = [0.1, 0.2, 0.3]  # Different spacing per dimension

        result = volume_integral(field, spacing=spacing)

        # Volume = n*0.1 * n*0.2 * n*0.3 (n points per dimension)
        expected_volume = n * 0.1 * n * 0.2 * n * 0.3
        torch.testing.assert_close(
            result, torch.tensor(expected_volume), atol=0.1, rtol=0.1
        )


class TestVolumeIntegralAutograd:
    """Autograd tests for volume_integral."""

    def test_gradcheck(self):
        """Volume integral passes gradcheck."""
        field = torch.randn(6, 6, 6, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: volume_integral(f, spacing=0.1),
            (field,),
        )

    def test_gradcheck_2d(self):
        """2D volume integral passes gradcheck."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda f: volume_integral(f, spacing=0.1),
            (field,),
        )


class TestVolumeIntegralAutocast:
    """Autocast tests for volume_integral."""

    def test_volume_integral_autocast(self):
        """Volume integral upcasts to fp32 under autocast."""
        field = torch.randn(8, 8, 8, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = volume_integral(field, spacing=0.1)

        assert result.dtype == torch.float32
        assert result.ndim == 0
