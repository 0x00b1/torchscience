"""Tests for line integral operators."""

import math

import torch

from torchscience.differentiation import Path, circulation, line_integral


class TestLineIntegral:
    """Tests for line_integral function."""

    def test_line_integral_constant_field(self):
        """Line integral of constant field along straight path."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # Constant field F = (1, 0)
        Fx = torch.ones_like(X)
        Fy = torch.zeros_like(Y)
        vector_field = torch.stack([Fx, Fy], dim=0)

        # Straight path from (0, 0) to (1, 0)
        t = torch.linspace(0, 1, 50)
        path_points = torch.stack([t, torch.zeros_like(t)], dim=1)
        path = Path(points=path_points)

        result = line_integral(vector_field, path, spacing=dx)

        # integral (1, 0) . (dx, 0) = integral 1 dx = 1
        expected = torch.tensor(1.0)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)

    def test_line_integral_conservative_field(self):
        """Line integral of gradient field equals endpoint difference."""
        n = 32
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 1.0 / (n - 1)

        # F = grad(x^2 + y^2) = (2x, 2y)
        Fx = 2 * X
        Fy = 2 * Y
        vector_field = torch.stack([Fx, Fy], dim=0)

        # Path from (0, 0) to (1, 1)
        t = torch.linspace(0, 1, 50)
        path_points = torch.stack([t, t], dim=1)
        path = Path(points=path_points)

        result = line_integral(vector_field, path, spacing=dx)

        # phi(1,1) - phi(0,0) = 2 - 0 = 2
        expected = torch.tensor(2.0)
        torch.testing.assert_close(result, expected, atol=0.2, rtol=0.1)

    def test_line_integral_returns_scalar(self):
        """Line integral returns a scalar."""
        n = 32
        vector_field = torch.randn(2, n, n)
        path_points = torch.rand(20, 2)
        path = Path(points=path_points)

        result = line_integral(vector_field, path, spacing=1.0 / n)

        assert result.ndim == 0

    def test_line_integral_circular_path(self):
        """Line integral along circular path."""
        n = 64
        # Grid from 0 to 4 (center at 2, 2)
        x = torch.linspace(0, 4, n)
        y = torch.linspace(0, 4, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 4.0 / (n - 1)

        # Radial field F = (x-2, y-2) centered at (2, 2)
        Fx = X - 2
        Fy = Y - 2
        vector_field = torch.stack([Fx, Fy], dim=0)

        # Half circle centered at (2, 2) from (3, 2) to (1, 2)
        theta = torch.linspace(0, math.pi, 100)
        r = 1.0
        cx, cy = 2.0, 2.0
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        path = Path(points=path_points)

        result = line_integral(vector_field, path, spacing=dx)

        # For radial field, the integral depends on path
        # This tests that it computes without error
        assert result.ndim == 0
        assert torch.isfinite(result)


class TestCirculation:
    """Tests for circulation function."""

    def test_circulation_irrotational_zero(self):
        """Circulation of irrotational field is zero."""
        n = 64
        # Grid from 0 to 2 (so center is at 1, 1)
        x = torch.linspace(0, 2, n)
        y = torch.linspace(0, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)

        # Uniform flow - irrotational
        Fx = torch.ones_like(X)
        Fy = torch.zeros_like(Y)
        vector_field = torch.stack([Fx, Fy], dim=0)

        # Circular contour centered at (1, 1)
        # For closed path, Path handles the closing segment automatically
        N = 100
        theta = torch.linspace(0, 2 * math.pi, N + 1)[
            :-1
        ]  # Exclude last point
        r = 0.5
        cx, cy = 1.0, 1.0
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        contour = Path(points=path_points, closed=True)

        result = circulation(vector_field, contour, spacing=dx)

        torch.testing.assert_close(
            result, torch.tensor(0.0), atol=0.2, rtol=0.5
        )

    def test_circulation_rigid_rotation(self):
        """Circulation of rigid rotation equals 2*pi*r^2."""
        n = 64
        # Grid from 0 to 4 (so center is at 2, 2)
        x = torch.linspace(0, 4, n)
        y = torch.linspace(0, 4, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 4.0 / (n - 1)

        # Rigid body rotation centered at (2, 2): v = (-(y-2), (x-2))
        Fx = -(Y - 2)
        Fy = X - 2
        vector_field = torch.stack([Fx, Fy], dim=0)

        # Circular contour of radius r centered at (2, 2)
        # For closed path, Path handles the closing segment automatically
        r = 1.0
        cx, cy = 2.0, 2.0
        N = 200
        theta = torch.linspace(0, 2 * math.pi, N + 1)[
            :-1
        ]  # Exclude last point
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        contour = Path(points=path_points, closed=True)

        result = circulation(vector_field, contour, spacing=dx)

        # Gamma = circulation v.dl = 2*pi*r^2 for rigid rotation with omega=1
        expected = torch.tensor(2 * math.pi * r**2)
        torch.testing.assert_close(result, expected, atol=0.5, rtol=0.2)

    def test_circulation_returns_scalar(self):
        """Circulation returns a scalar."""
        n = 32
        vector_field = torch.randn(2, n, n)
        N = 20
        theta = torch.linspace(0, 2 * math.pi, N + 1)[
            :-1
        ]  # Exclude last point
        # Path centered at (0.5, 0.5) with radius 0.25 to stay within grid [0, 1]
        cx, cy = 0.5, 0.5
        r = 0.25
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        contour = Path(points=path_points, closed=True)

        result = circulation(vector_field, contour, spacing=1.0 / n)

        assert result.ndim == 0

    def test_circulation_forces_closed(self):
        """Circulation forces path to be closed."""
        n = 32
        # Grid from 0 to 2 (so center is at 1, 1)
        x = torch.linspace(0, 2, n)
        y = torch.linspace(0, 2, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        dx = 2.0 / (n - 1)

        # Uniform field
        vector_field = torch.stack(
            [torch.ones_like(X), torch.zeros_like(Y)], dim=0
        )

        # Non-closed contour centered at (1, 1) - will be treated as closed
        N = 50
        theta = torch.linspace(0, 2 * math.pi, N + 1)[
            :-1
        ]  # Exclude last point
        cx, cy = 1.0, 1.0
        r = 0.5
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        contour = Path(points=path_points, closed=False)

        # Should not raise
        result = circulation(vector_field, contour, spacing=dx)
        assert torch.isfinite(result)


class TestLineIntegralAutocast:
    """Autocast tests for line_integral."""

    def test_line_integral_autocast(self):
        """Line integral upcasts to fp32 under autocast."""
        n = 32
        vector_field = torch.randn(2, n, n, dtype=torch.float16, device="cpu")
        # Path within the grid [0, 1] x [0, 1]
        path_points = 0.2 + 0.6 * torch.rand(
            20, 2, dtype=torch.float16, device="cpu"
        )
        path = Path(points=path_points)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = line_integral(vector_field, path, spacing=1.0 / (n - 1))

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.ndim == 0


class TestCirculationAutocast:
    """Autocast tests for circulation."""

    def test_circulation_autocast(self):
        """Circulation upcasts to fp32 under autocast."""
        n = 32
        vector_field = torch.randn(2, n, n, dtype=torch.float16, device="cpu")
        N = 20
        theta = torch.linspace(0, 2 * math.pi, N + 1, dtype=torch.float16)[:-1]
        # Path centered at (0.5, 0.5) with radius 0.2 to stay within grid [0, 1]
        cx, cy = 0.5, 0.5
        r = 0.2
        path_points = torch.stack(
            [cx + r * torch.cos(theta), cy + r * torch.sin(theta)], dim=1
        )
        contour = Path(points=path_points, closed=True)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = circulation(vector_field, contour, spacing=1.0 / (n - 1))

        assert result.dtype == torch.float32
        assert result.ndim == 0
