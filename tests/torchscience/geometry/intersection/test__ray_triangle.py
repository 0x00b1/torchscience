"""Tests for ray_triangle intersection."""

import pytest
import torch

from torchscience.geometry.intersection import IntersectionResult, ray_triangle


class TestRayTriangleBasicHits:
    """Basic intersection tests."""

    def test_ray_hits_triangle_at_origin(self):
        """Ray from (0,0,-1) toward (0,0,1) hitting triangle in XY plane.

        Triangle v0=(1,0,0), v1=(0,1,0), v2=(-1,-1,0) has centroid at (0,0,0).
        Ray hits at t=1, hit_point=(0,0,0).
        """
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert isinstance(result, IntersectionResult)
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            atol=1e-5,
        ).all()

    def test_ray_hits_triangle_from_above(self):
        """Ray from above the triangle pointing down."""
        origins = torch.tensor([[0.0, 0.0, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[5.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            atol=1e-5,
        ).all()

    def test_ray_hits_triangle_at_angle(self):
        """Ray hitting the triangle at an angle."""
        origins = torch.tensor([[1.0, 1.0, -2.0]])
        directions = torch.tensor([[-1.0, -1.0, 2.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        # At t=1: origin + t*dir = (1-1, 1-1, -2+2) = (0,0,0)
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            atol=1e-5,
        ).all()

    def test_ray_hits_offset_triangle(self):
        """Ray hitting a triangle not at the origin."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 2.0]])
        v1 = torch.tensor([[0.0, 1.0, 2.0]])
        v2 = torch.tensor([[-1.0, -1.0, 2.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[5.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point,
            torch.tensor([[[0.0, 0.0, 2.0]]]),
            atol=1e-5,
        ).all()


class TestRayTriangleMissCases:
    """Tests for ray miss cases."""

    def test_ray_parallel_to_triangle(self):
        """Ray parallel to the triangle plane should miss."""
        origins = torch.tensor([[0.0, 0.0, 1.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])  # Parallel to XY plane
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_pointing_away(self):
        """Ray pointing away from the triangle should miss."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])  # Away from triangle
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_outside_triangle_edges(self):
        """Ray through triangle plane but outside the triangle boundary."""
        origins = torch.tensor([[5.0, 5.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_near_edge(self):
        """Ray just outside the triangle edge."""
        # Slightly beyond vertex v0 = (1,0,0)
        origins = torch.tensor([[1.5, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is False


class TestRayTriangleBarycentricCoordinates:
    """Tests for barycentric UV coordinates.

    Convention: P = (1-u-v)*V0 + u*V1 + v*V2.
    At V0: u=0, v=0
    At V1: u=1, v=0
    At V2: u=0, v=1
    At centroid: u=1/3, v=1/3
    """

    def test_uv_at_centroid(self):
        """Hit at triangle centroid should give u=1/3, v=1/3."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.uv[..., 0],
            torch.tensor([[1.0 / 3.0]]),
            atol=1e-5,
        ).all()
        assert torch.isclose(
            result.uv[..., 1],
            torch.tensor([[1.0 / 3.0]]),
            atol=1e-5,
        ).all()

    def test_uv_at_vertex_v1(self):
        """Hit at vertex V1 should give u=1, v=0.

        Use a triangle and ray aimed precisely at V1.
        """
        v0 = torch.tensor([[0.0, -1.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-2.0, -1.0, 0.0]])
        # V1 is at (0, 1, 0)
        origins = torch.tensor([[0.0, 1.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.uv[..., 0], torch.tensor([[1.0]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[0.0]]), atol=1e-5
        ).all()

    def test_uv_at_vertex_v2(self):
        """Hit at vertex V2 should give u=0, v=1.

        Use a triangle and ray aimed precisely at V2.
        """
        v0 = torch.tensor([[0.0, 1.0, 0.0]])
        v1 = torch.tensor([[2.0, 1.0, 0.0]])
        v2 = torch.tensor([[0.0, -1.0, 0.0]])
        # V2 is at (0, -1, 0)
        origins = torch.tensor([[0.0, -1.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.uv[..., 0], torch.tensor([[0.0]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[1.0]]), atol=1e-5
        ).all()

    def test_uv_at_edge_midpoint(self):
        """Hit at midpoint of V0-V1 edge should give u=0.5, v=0.

        Midpoint of V0-V1 = (1-u-v)*V0 + u*V1 + v*V2 with u=0.5, v=0.
        So P = 0.5*V0 + 0.5*V1.
        """
        v0 = torch.tensor([[2.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 2.0, 0.0]])
        v2 = torch.tensor([[-2.0, -2.0, 0.0]])
        # Midpoint of V0-V1: (1, 1, 0)
        origins = torch.tensor([[1.0, 1.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.uv[..., 0], torch.tensor([[0.5]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[0.0]]), atol=1e-5
        ).all()


class TestRayTriangleNormals:
    """Tests for surface normal computation.

    Normal is normalize(edge1 x edge2) where edge1 = V1 - V0, edge2 = V2 - V0.
    """

    def test_normal_xy_plane_triangle(self):
        """Normal for a triangle in the XY plane should point in +Z direction.

        Triangle: v0=(1,0,0), v1=(0,1,0), v2=(-1,-1,0)
        edge1 = (-1,1,0), edge2 = (-2,-1,0)
        edge1 x edge2 = (0, 0, (-1)(-1) - (1)(-2)) = (0, 0, 3)
        Normalized: (0, 0, 1)
        """
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.normal,
            torch.tensor([[[0.0, 0.0, 1.0]]]),
            atol=1e-5,
        ).all()

    def test_normal_xz_plane_triangle(self):
        """Triangle in the XZ plane should have normal in Y direction.

        Triangle: v0=(1,0,0), v1=(0,0,1), v2=(-1,0,-1)
        edge1 = v1-v0 = (-1,0,1), edge2 = v2-v0 = (-2,0,-1)
        cross = (0*(-1)-1*0, 1*(-2)-(-1)*(-1), (-1)*0-0*(-2)) = (0, -2-1, 0) = (0, -3, 0)
        Normalized: (0, -1, 0)
        """
        origins = torch.tensor([[0.0, -2.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 0.0, 1.0]])
        v2 = torch.tensor([[-1.0, 0.0, -1.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(
            result.normal,
            torch.tensor([[[0.0, -1.0, 0.0]]]),
            atol=1e-5,
        ).all()

    def test_normals_are_unit_length(self):
        """Surface normals should have unit length."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        normal_length = result.normal.norm(dim=-1)
        assert torch.isclose(
            normal_length, torch.ones_like(normal_length), atol=1e-5
        ).all()

    def test_normal_matches_edge_cross_product(self):
        """Normal should equal normalize(edge1 x edge2) for arbitrary triangle."""
        v0 = torch.tensor([[2.0, 1.0, 0.5]])
        v1 = torch.tensor([[0.0, 3.0, 0.5]])
        v2 = torch.tensor([[-1.0, 0.0, 0.5]])
        # Centroid: (2+0-1)/3, (1+3+0)/3, 0.5 = (1/3, 4/3, 0.5)
        centroid = (v0 + v1 + v2) / 3.0
        origins = centroid.clone()
        origins[..., 2] = -1.0
        directions = torch.tensor([[0.0, 0.0, 1.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        edge1 = v1 - v0
        edge2 = v2 - v0
        expected_normal = torch.linalg.cross(edge1, edge2)
        expected_normal = expected_normal / expected_normal.norm(
            dim=-1, keepdim=True
        )

        assert result.hit.item() is True
        assert torch.isclose(
            result.normal.squeeze(0),
            expected_normal.unsqueeze(0),
            atol=1e-5,
        ).all()


class TestRayTriangleBroadcasting:
    """Tests for batched operations with cross-product broadcasting.

    ray_triangle uses cross-product broadcasting:
    rays (N, 3) x triangles (M, 3) -> (N, M) output
    """

    def test_multiple_rays_single_triangle(self):
        """Multiple rays against a single triangle."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Hit centroid
                [5.0, 5.0, -1.0],  # Miss (outside)
                [0.5, 0.0, -1.0],  # Hit (near V0)
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Cross-product: (3, 3) x (1, 3) -> (3, 1)
        assert result.t.shape == (3, 1)
        assert result.hit[0, 0].item() is True  # Centroid hit
        assert result.hit[1, 0].item() is False  # Miss
        assert result.hit[2, 0].item() is True  # Near V0 hit

    def test_single_ray_multiple_triangles(self):
        """One ray against multiple triangles."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Triangle centered at origin
                [101.0, 100.0, 0.0],  # Triangle far away (miss)
            ]
        )
        v1 = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [100.0, 101.0, 0.0],
            ]
        )
        v2 = torch.tensor(
            [
                [-1.0, -1.0, 0.0],
                [99.0, 99.0, 0.0],
            ]
        )

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Cross-product: (1, 3) x (2, 3) -> (1, 2)
        assert result.t.shape == (1, 2)
        assert result.hit[0, 0].item() is True  # Hits first triangle
        assert result.hit[0, 1].item() is False  # Misses second triangle

    def test_broadcast_rays_and_triangles(self):
        """Rays (N,3) x Triangles (M,3) gives cross-product (N,M) output."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -5.0],
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        v0 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 3.0],
            ]
        )
        v1 = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
            ]
        )
        v2 = torch.tensor(
            [
                [-1.0, -1.0, 0.0],
                [-1.0, -1.0, 3.0],
            ]
        )

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Cross-product: (2, 3) x (2, 3) -> (2, 2)
        assert result.t.shape == (2, 2)
        # Ray 0 (z=-1) -> triangle 0 (z=0): t=1
        assert torch.isclose(result.t[0, 0], torch.tensor(1.0), atol=1e-5)
        # Ray 0 (z=-1) -> triangle 1 (z=3): t=4
        assert torch.isclose(result.t[0, 1], torch.tensor(4.0), atol=1e-5)
        # Ray 1 (z=-5) -> triangle 0 (z=0): t=5
        assert torch.isclose(result.t[1, 0], torch.tensor(5.0), atol=1e-5)
        # Ray 1 (z=-5) -> triangle 1 (z=3): t=8
        assert torch.isclose(result.t[1, 1], torch.tensor(8.0), atol=1e-5)

    def test_2d_batch(self):
        """2D batch of rays against 2D batch of triangles."""
        # 4x5 batch of rays
        origins = torch.zeros(4, 5, 3)
        origins[..., 2] = -1.0  # All rays start at z=-1
        directions = torch.zeros(4, 5, 3)
        directions[..., 2] = 1.0  # All point toward +z

        # 2x3 batch of triangles at z=0
        v0 = torch.zeros(2, 3, 3)
        v0[..., 0] = 100.0  # x=100, so ray at origin misses
        v1 = torch.zeros(2, 3, 3)
        v1[..., 0] = 100.0
        v1[..., 1] = 1.0
        v2 = torch.zeros(2, 3, 3)
        v2[..., 0] = 99.0
        v2[..., 1] = -1.0

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Cross-product: (4, 5, 3) x (2, 3, 3) -> (4, 5, 2, 3)
        assert result.t.shape == (4, 5, 2, 3)
        assert result.hit.shape == (4, 5, 2, 3)
        assert result.hit_point.shape == (4, 5, 2, 3, 3)
        assert result.normal.shape == (4, 5, 2, 3, 3)
        assert result.uv.shape == (4, 5, 2, 3, 2)


class TestRayTriangleGradients:
    """Gradient tests for differentiable rendering."""

    def _setup_hit(self):
        """Return a ray+triangle config that clearly hits center."""
        v0 = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        v1 = torch.tensor(
            [[-1.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        v2 = torch.tensor(
            [[-1.0, -1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        origins = torch.tensor(
            [[0.0, 0.0, -2.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        return origins, directions, v0, v1, v2

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        origins, directions, v0, v1, v2 = self._setup_hit()
        directions = directions.detach()
        v0 = v0.detach()
        v1 = v1.detach()
        v2 = v2.detach()

        def func(o):
            result = ray_triangle(o, directions, v0, v1, v2)
            return result.t

        torch.autograd.gradcheck(func, (origins,), raise_exception=True)

    def test_gradcheck_directions(self):
        """Gradient check for ray directions."""
        origins, directions, v0, v1, v2 = self._setup_hit()
        origins = origins.detach()
        v0 = v0.detach()
        v1 = v1.detach()
        v2 = v2.detach()
        # Use slightly off-center direction for numeric stability
        directions = torch.tensor(
            [[0.05, 0.05, 1.0]], dtype=torch.float64, requires_grad=True
        )

        def func(d):
            result = ray_triangle(origins, d, v0, v1, v2)
            return result.t

        torch.autograd.gradcheck(func, (directions,), raise_exception=True)

    def test_gradcheck_v0(self):
        """Gradient check for triangle vertex v0."""
        origins, directions, v0, v1, v2 = self._setup_hit()
        origins = origins.detach()
        directions = directions.detach()
        v1 = v1.detach()
        v2 = v2.detach()

        def func(vertex):
            result = ray_triangle(origins, directions, vertex, v1, v2)
            return result.t

        torch.autograd.gradcheck(func, (v0,), raise_exception=True)

    def test_gradcheck_v1(self):
        """Gradient check for triangle vertex v1."""
        origins, directions, v0, v1, v2 = self._setup_hit()
        origins = origins.detach()
        directions = directions.detach()
        v0 = v0.detach()
        v2 = v2.detach()

        def func(vertex):
            result = ray_triangle(origins, directions, v0, vertex, v2)
            return result.t

        torch.autograd.gradcheck(func, (v1,), raise_exception=True)

    def test_gradcheck_v2(self):
        """Gradient check for triangle vertex v2."""
        origins, directions, v0, v1, v2 = self._setup_hit()
        origins = origins.detach()
        directions = directions.detach()
        v0 = v0.detach()
        v1 = v1.detach()

        def func(vertex):
            result = ray_triangle(origins, directions, v0, v1, vertex)
            return result.t

        torch.autograd.gradcheck(func, (v2,), raise_exception=True)


class TestRayTriangleSecondOrderGradients:
    """Second-order gradient tests."""

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_triangle"
    )
    def test_gradgradcheck_origins(self):
        """Second-order gradient check for ray origins."""
        v0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        v1 = torch.tensor([[-1.0, 1.0, 0.0]], dtype=torch.float64)
        v2 = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64)
        origins = torch.tensor(
            [[0.0, 0.0, -2.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

        def func(o):
            result = ray_triangle(o, directions, v0, v1, v2)
            return result.t

        torch.autograd.gradgradcheck(func, (origins,), raise_exception=True)

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_triangle"
    )
    def test_gradgradcheck_v0(self):
        """Second-order gradient check for triangle vertex v0."""
        origins = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        v0 = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        v1 = torch.tensor([[-1.0, 1.0, 0.0]], dtype=torch.float64)
        v2 = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64)

        def func(vertex):
            result = ray_triangle(origins, directions, vertex, v1, v2)
            return result.t

        torch.autograd.gradgradcheck(func, (v0,), raise_exception=True)


class TestRayTriangleMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensors_basic(self):
        """Shape inference with meta tensors."""
        origins = torch.randn(4, 5, 3, device="meta")
        directions = torch.randn(4, 5, 3, device="meta")
        v0 = torch.zeros(2, 3, device="meta")
        v1 = torch.zeros(2, 3, device="meta")
        v2 = torch.zeros(2, 3, device="meta")

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Cross-product: (4, 5, 3) x (2, 3) -> (4, 5, 2)
        assert result.t.shape == (4, 5, 2)
        assert result.t.device.type == "meta"
        assert result.hit_point.shape == (4, 5, 2, 3)
        assert result.normal.shape == (4, 5, 2, 3)
        assert result.uv.shape == (4, 5, 2, 2)
        assert result.hit.shape == (4, 5, 2)

    def test_meta_tensors_single_ray_multiple_triangles(self):
        """Meta tensor shape inference with single ray, multiple triangles."""
        origins = torch.randn(1, 3, device="meta")
        directions = torch.randn(1, 3, device="meta")
        v0 = torch.zeros(10, 3, device="meta")
        v1 = torch.zeros(10, 3, device="meta")
        v2 = torch.zeros(10, 3, device="meta")

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.t.shape == (1, 10)
        assert result.hit.shape == (1, 10)
        assert result.hit_point.shape == (1, 10, 3)
        assert result.uv.shape == (1, 10, 2)


class TestRayTriangleEdgeCases:
    """Edge case tests."""

    def test_degenerate_triangle_zero_area(self):
        """Degenerate triangle with collinear vertices should produce no hit."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        # Three collinear points (zero area triangle)
        v0 = torch.tensor([[0.0, 0.0, 0.0]])
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[2.0, 0.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        # Degenerate triangle: edge1 x edge2 has zero cross product,
        # determinant a = 0, so parallel check triggers a miss
        assert result.hit.item() is False

    def test_large_triangle(self):
        """Intersection with a very large triangle."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        scale = 1000.0
        v0 = torch.tensor([[scale, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, scale, 0.0]])
        v2 = torch.tensor([[-scale, -scale, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-5).all()

    def test_small_triangle(self):
        """Intersection with a very small triangle."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        scale = 1e-3
        v0 = torch.tensor([[scale, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, scale, 0.0]])
        v2 = torch.tensor([[-scale, -scale, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-4).all()

    def test_unnormalized_directions(self):
        """Works with non-unit direction vectors."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 10.0]])  # Length 10
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        # t is parametric: origin + t * dir = (0,0,-1) + t*(0,0,10) = (0,0,-1+10t)
        # hit at z=0 => 10t=1 => t=0.1
        assert torch.isclose(result.t, torch.tensor([[0.1]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            atol=1e-5,
        ).all()

    def test_distant_triangle(self):
        """Intersection with a distant triangle."""
        origins = torch.tensor([[0.0, 0.0, -1e4]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        v0 = torch.tensor([[1.0, 0.0, 0.0]])
        v1 = torch.tensor([[0.0, 1.0, 0.0]])
        v2 = torch.tensor([[-1.0, -1.0, 0.0]])

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[1e4]]), rtol=1e-4).all()


class TestRayTriangleValidation:
    """Input validation tests."""

    def test_invalid_origin_shape(self):
        """Origins must have last dimension 3."""
        origins = torch.randn(10, 2)
        directions = torch.randn(10, 3)
        v0 = torch.zeros(1, 3)
        v1 = torch.zeros(1, 3)
        v2 = torch.zeros(1, 3)

        with pytest.raises(ValueError, match="origins must have shape"):
            ray_triangle(origins, directions, v0, v1, v2)

    def test_invalid_direction_shape(self):
        """Directions must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 4)
        v0 = torch.zeros(1, 3)
        v1 = torch.zeros(1, 3)
        v2 = torch.zeros(1, 3)

        with pytest.raises(ValueError, match="directions must have shape"):
            ray_triangle(origins, directions, v0, v1, v2)

    def test_invalid_v0_shape(self):
        """v0 must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        v0 = torch.zeros(1, 2)
        v1 = torch.zeros(1, 3)
        v2 = torch.zeros(1, 3)

        with pytest.raises(ValueError, match="v0 must have shape"):
            ray_triangle(origins, directions, v0, v1, v2)

    def test_invalid_v1_shape(self):
        """v1 must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        v0 = torch.zeros(1, 3)
        v1 = torch.zeros(1, 2)
        v2 = torch.zeros(1, 3)

        with pytest.raises(ValueError, match="v1 must have shape"):
            ray_triangle(origins, directions, v0, v1, v2)

    def test_invalid_v2_shape(self):
        """v2 must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        v0 = torch.zeros(1, 3)
        v1 = torch.zeros(1, 3)
        v2 = torch.zeros(1, 2)

        with pytest.raises(ValueError, match="v2 must have shape"):
            ray_triangle(origins, directions, v0, v1, v2)


class TestRayTriangleDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        v0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        v1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        v2 = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float32)

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.t.dtype == torch.float32
        assert result.hit_point.dtype == torch.float32
        assert result.normal.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        v0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        v1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        v2 = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64)

        result = ray_triangle(origins, directions, v0, v1, v2)

        assert result.t.dtype == torch.float64
        assert result.hit_point.dtype == torch.float64
        assert result.normal.dtype == torch.float64
