"""Tests for surface integral operators."""

import math

import torch

from torchscience.differentiation import Surface, flux, surface_integral


class TestSurfaceIntegral:
    """Tests for surface_integral function."""

    def test_surface_integral_constant_scalar(self):
        """Integral of constant scalar over flat surface equals area times constant."""
        # Flat surface in xy-plane from (0,0) to (1,1)
        n = 16
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # Constant scalar field
        scalar_field = 2.0 * torch.ones(n, n)

        result = surface_integral(scalar_field, surface)

        # Area = 1, so integral = 2
        expected = torch.tensor(2.0)
        torch.testing.assert_close(result, expected, atol=0.2, rtol=0.1)

    def test_surface_integral_returns_scalar(self):
        """Surface integral returns a scalar."""
        n = 8
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        scalar_field = torch.randn(n, n)
        result = surface_integral(scalar_field, surface)

        assert result.ndim == 0

    def test_surface_integral_linear_field(self):
        """Integral of linear field over unit square."""
        n = 32
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # f = x + y, integral (x + y) dA over [0,1]^2 = 1
        scalar_field = U + V

        result = surface_integral(scalar_field, surface)

        expected = torch.tensor(1.0)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)

    def test_surface_integral_quadratic_field(self):
        """Integral of x*y over unit square."""
        n = 32
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # f = x*y, integral over [0,1]^2 = 1/4
        scalar_field = U * V

        result = surface_integral(scalar_field, surface)

        expected = torch.tensor(0.25)
        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.1)


class TestFlux:
    """Tests for flux function."""

    def test_flux_uniform_field_flat_surface(self):
        """Flux of uniform z-field through flat xy surface."""
        n = 16
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # Uniform field F = (0, 0, 1)
        Fx = torch.zeros(n, n)
        Fy = torch.zeros(n, n)
        Fz = torch.ones(n, n)
        vector_field = torch.stack([Fx, Fy, Fz], dim=0)

        result = flux(vector_field, surface)

        # Flux = integral F.n dA = 1 * area = 1
        expected = torch.tensor(1.0)
        torch.testing.assert_close(result.abs(), expected, atol=0.2, rtol=0.1)

    def test_flux_returns_scalar(self):
        """Flux returns a scalar."""
        n = 8
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        vector_field = torch.randn(3, n, n)
        result = flux(vector_field, surface)

        assert result.ndim == 0

    def test_flux_tangential_field_zero(self):
        """Flux of tangential field through surface is zero."""
        n = 16
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # Field tangent to surface: F = (1, 0, 0)
        Fx = torch.ones(n, n)
        Fy = torch.zeros(n, n)
        Fz = torch.zeros(n, n)
        vector_field = torch.stack([Fx, Fy, Fz], dim=0)

        result = flux(vector_field, surface)

        # Tangential field has zero normal component
        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=0.1, rtol=0.5
        )

    def test_flux_non_uniform_field(self):
        """Flux of position-dependent field."""
        n = 32
        u = torch.linspace(0, 1, n)
        v = torch.linspace(0, 1, n)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # Field F = (0, 0, x) -> flux = integral x dA = 1/2
        Fx = torch.zeros(n, n)
        Fy = torch.zeros(n, n)
        Fz = U
        vector_field = torch.stack([Fx, Fy, Fz], dim=0)

        result = flux(vector_field, surface)

        expected = torch.tensor(0.5)
        torch.testing.assert_close(result.abs(), expected, atol=0.1, rtol=0.1)


class TestSurfaceIntegralAutocast:
    """Autocast tests for surface_integral."""

    def test_surface_integral_autocast(self):
        """Surface integral upcasts to fp32 under autocast."""
        n = 16
        u = torch.linspace(0, 1, n, dtype=torch.float16)
        v = torch.linspace(0, 1, n, dtype=torch.float16)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        scalar_field = torch.randn(n, n, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = surface_integral(scalar_field, surface)

        assert result.dtype == torch.float32
        assert result.ndim == 0


class TestFluxAutocast:
    """Autocast tests for flux."""

    def test_flux_autocast(self):
        """Flux upcasts to fp32 under autocast."""
        n = 16
        u = torch.linspace(0, 1, n, dtype=torch.float16)
        v = torch.linspace(0, 1, n, dtype=torch.float16)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        vector_field = torch.randn(3, n, n, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = flux(vector_field, surface)

        assert result.dtype == torch.float32
        assert result.ndim == 0


class TestSurfaceIntegralCurvedSurface:
    """Tests for surface_integral on curved surfaces."""

    def test_surface_integral_hemisphere(self):
        """Surface integral over a hemisphere."""
        # Hemisphere of radius 1 centered at origin
        n = 32
        u = torch.linspace(0, math.pi / 2, n)  # polar angle from 0 to pi/2
        v = torch.linspace(0, 2 * math.pi, n)  # azimuthal angle
        U, V = torch.meshgrid(u, v, indexing="ij")

        # Parametric form of hemisphere
        r = 1.0
        X = r * torch.sin(U) * torch.cos(V)
        Y = r * torch.sin(U) * torch.sin(V)
        Z = r * torch.cos(U)

        points = torch.stack([X, Y, Z], dim=-1)
        surface = Surface(points=points)

        # Constant scalar field = 1
        scalar_field = torch.ones(n, n)

        result = surface_integral(scalar_field, surface)

        # Surface area of hemisphere = 2*pi*r^2
        expected = torch.tensor(2 * math.pi * r**2)
        torch.testing.assert_close(result, expected, atol=0.5, rtol=0.2)
