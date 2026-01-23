"""Tests for Path and Surface classes."""

import math

import pytest
import torch

from torchscience.differentiation import Path, Surface


class TestPath:
    """Tests for Path class."""

    def test_from_points(self):
        """Create path from explicit points."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        path = Path(points=points)
        assert path.n_points == 3
        assert path.ndim == 2
        assert not path.closed

    def test_closed_path(self):
        """Create closed path."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        path = Path(points=points, closed=True)
        assert path.closed

    def test_tangent_vectors(self):
        """Path tangent vectors."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        path = Path(points=points)
        tangents = path.tangents
        assert tangents.shape == (2, 2)  # N-1 tangent vectors
        torch.testing.assert_close(tangents[0], torch.tensor([1.0, 0.0]))

    def test_closed_path_tangents(self):
        """Closed path has N tangent vectors."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        path = Path(points=points, closed=True)
        tangents = path.tangents
        assert tangents.shape == (3, 2)  # N tangent vectors for closed

    def test_segment_lengths(self):
        """Path segment lengths."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        path = Path(points=points)
        lengths = path.segment_lengths
        torch.testing.assert_close(lengths, torch.tensor([1.0, 1.0]))

    def test_total_length(self):
        """Total arc length."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        path = Path(points=points)
        assert path.total_length.item() == pytest.approx(2.0)

    def test_from_parametric(self):
        """Create path from parametric function."""

        # Circle parametrization
        def circle(t: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.cos(t), torch.sin(t)])

        t = torch.linspace(0, 2 * math.pi, 100)
        path = Path.from_parametric(circle, t, closed=True)

        assert path.n_points == 100
        assert path.ndim == 2
        assert path.closed

        # Circle circumference is 2*pi
        assert path.total_length.item() == pytest.approx(2 * math.pi, rel=0.01)

    def test_3d_path(self):
        """Path in 3D space."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        path = Path(points=points)
        assert path.ndim == 3
        assert path.total_length.item() == pytest.approx(3.0)

    def test_midpoints(self):
        """Midpoints of path segments."""
        points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        path = Path(points=points)
        midpoints = path.midpoints
        assert midpoints.shape == (2, 2)
        torch.testing.assert_close(midpoints[0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(midpoints[1], torch.tensor([2.0, 1.0]))


class TestSurface:
    """Tests for Surface class."""

    def test_from_grid(self):
        """Create surface from grid."""
        u = torch.linspace(0, 1, 5)
        v = torch.linspace(0, 1, 5)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)
        assert surface.shape == (5, 5)
        assert surface.ndim == 3

    def test_flat_surface_normals(self):
        """Flat xy-plane has z-direction normals."""
        u = torch.linspace(0, 1, 5)
        v = torch.linspace(0, 1, 5)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)
        normals = surface.normals

        assert normals.shape == (5, 5, 3)
        # All normals should point in z direction (either +z or -z)
        z_component = normals[..., 2].abs()
        torch.testing.assert_close(
            z_component, torch.ones(5, 5), atol=0.1, rtol=0.1
        )

    def test_area_elements(self):
        """Area elements for flat surface."""
        u = torch.linspace(0, 1, 5)
        v = torch.linspace(0, 1, 5)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)
        areas = surface.area_elements

        assert areas.shape == (5, 5)
        # All area elements should be similar for uniform grid
        assert areas.std() < 0.1

    def test_from_parametric(self):
        """Create surface from parametric function."""

        # Flat plane z = 0
        def flat_plane(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.stack([u, v, torch.zeros_like(u)])

        u = torch.linspace(0, 1, 10)
        v = torch.linspace(0, 1, 10)
        surface = Surface.from_parametric(flat_plane, u, v)

        assert surface.shape == (10, 10)
        assert surface.ndim == 3

    def test_sphere_surface(self):
        """Create spherical surface."""

        def sphere(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            return torch.stack([x, y, z])

        theta = torch.linspace(0.1, math.pi - 0.1, 20)  # Avoid poles
        phi = torch.linspace(0, 2 * math.pi, 20)
        surface = Surface.from_parametric(sphere, theta, phi)

        assert surface.shape == (20, 20)

        # Normals should point radially (same direction as points for unit sphere)
        normals = surface.normals
        # Check that normals are approximately normalized
        norms = torch.linalg.norm(normals, dim=-1)
        torch.testing.assert_close(
            norms, torch.ones_like(norms), atol=0.1, rtol=0.1
        )

    def test_total_area_flat(self):
        """Total area of flat unit square."""
        u = torch.linspace(0, 1, 50)
        v = torch.linspace(0, 1, 50)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        surface = Surface(points=points)

        # The area_elements give |du x dv| which for a uniform flat grid
        # equals the area of each cell. Sum all to get total area.
        # For a unit square with 50x50 points (49x49 cells), each cell
        # has area (1/49)^2 ~ 0.000416
        total_area = surface.area_elements.sum()
        assert total_area.item() == pytest.approx(1.0, rel=0.1)


class TestPathGradients:
    """Test autograd support for Path."""

    def test_path_gradient_flow(self):
        """Gradients flow through path computations."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True
        )
        path = Path(points=points)
        loss = path.total_length
        loss.backward()
        assert points.grad is not None
        assert not torch.isnan(points.grad).any()


class TestSurfaceGradients:
    """Test autograd support for Surface."""

    def test_surface_gradient_flow(self):
        """Gradients flow through surface computations."""
        u = torch.linspace(0, 1, 5)
        v = torch.linspace(0, 1, 5)
        U, V = torch.meshgrid(u, v, indexing="ij")
        points = torch.stack([U, V, torch.zeros_like(U)], dim=-1)
        points = points.clone().detach().requires_grad_(True)

        surface = Surface(points=points)
        loss = surface.area_elements.sum()
        loss.backward()
        assert points.grad is not None
        assert not torch.isnan(points.grad).any()
