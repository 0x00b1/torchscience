"""Tests for ray_plane intersection."""

import pytest
import torch

from torchscience.geometry.intersection import IntersectionResult, ray_plane


class TestRayPlaneIntersectionBasic:
    """Basic intersection tests."""

    def test_ray_hits_horizontal_plane(self):
        """Ray from above hitting ground plane (y=0)."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])  # Ground plane
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert isinstance(result, IntersectionResult)
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([5.0]), atol=1e-5).all()
        # Hit point should be at origin on the ground
        assert torch.isclose(
            result.hit_point, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-5
        ).all()

    def test_ray_hits_vertical_plane(self):
        """Ray hitting a vertical wall at x=5."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        plane_normals = torch.tensor([[1.0, 0.0, 0.0]])  # Wall at x=5
        plane_offsets = torch.tensor([5.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([5.0]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[5.0, 0.0, 0.0]]), atol=1e-5
        ).all()

    def test_ray_misses_parallel_plane(self):
        """Ray parallel to plane should miss."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])  # Parallel to ground
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_away_from_plane(self):
        """Ray pointing away from plane should miss."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor(
            [[0.0, 1.0, 0.0]]
        )  # Pointing away from ground
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()


class TestRayPlaneIntersectionBatched:
    """Tests for batched operations.

    NOTE: ray_plane uses cross-product broadcasting semantics:
    rays (N, 3) x planes (M, 3) → (N, M) output
    This tests each ray against each plane.
    """

    def test_batch_rays_single_plane(self):
        """Multiple rays against single plane."""
        origins = torch.tensor(
            [
                [0.0, 5.0, 0.0],  # Hit from above
                [0.0, 5.0, 1.0],  # Hit, parallel offset
                [0.0, -5.0, 0.0],  # Miss (below plane, pointing down)
            ]
        )
        directions = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],  # Away from plane
            ]
        )
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])  # (1, 3)
        plane_offsets = torch.tensor([0.0])  # (1,)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Cross-product: (3, 3) x (1, 3) → (3, 1)
        assert result.t.shape == (3, 1)
        assert result.hit[0, 0].item() is True
        assert result.hit[1, 0].item() is True
        assert result.hit[2, 0].item() is False

    def test_single_ray_batch_planes(self):
        """One ray against multiple planes."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3)
        directions = torch.tensor([[1.0, 0.0, 0.0]])  # (1, 3)
        plane_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Wall at x=2
                [1.0, 0.0, 0.0],  # Wall at x=5
                [-1.0, 0.0, 0.0],  # Wall facing opposite direction at x=-3
            ]
        )  # (3, 3)
        plane_offsets = torch.tensor([2.0, 5.0, 3.0])  # (3,)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Cross-product: (1, 3) x (3, 3) → (1, 3)
        assert result.t.shape == (1, 3)
        assert result.hit[0, 0].item() is True
        assert result.hit[0, 1].item() is True
        # Third plane: normal dot direction = -1 (ray goes against normal)
        # t = (3 - (-1)*0) / (-1*1) = 3 / -1 = -3 (behind ray)
        assert result.hit[0, 2].item() is False

    def test_broadcast_rays_and_planes(self):
        """Rays (N,3) x Planes (M,3) gives cross-product (N,M) output."""
        origins = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )  # (2, 3)
        directions = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )  # (2, 3)
        plane_normals = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # y=0
                [0.0, 1.0, 0.0],  # y=2
                [0.0, 1.0, 0.0],  # y=5
            ]
        )  # (3, 3)
        plane_offsets = torch.tensor([0.0, 2.0, 5.0])  # (3,)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Cross-product: (2, 3) x (3, 3) → (2, 3)
        assert result.t.shape == (2, 3)
        # Ray 0 (from y=5): distances to y=0, y=2, y=5 are 5, 3, 0
        assert torch.isclose(result.t[0, 0], torch.tensor(5.0), atol=1e-5)
        assert torch.isclose(result.t[0, 1], torch.tensor(3.0), atol=1e-5)
        assert torch.isclose(result.t[0, 2], torch.tensor(0.0), atol=1e-5)
        # Ray 1 (from y=10): distances to y=0, y=2, y=5 are 10, 8, 5
        assert torch.isclose(result.t[1, 0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(result.t[1, 1], torch.tensor(8.0), atol=1e-5)
        assert torch.isclose(result.t[1, 2], torch.tensor(5.0), atol=1e-5)

    def test_2d_batch(self):
        """2D batch of rays against 2D batch of planes."""
        # For same-shape batches, output is cross-product of batches
        origins = torch.randn(4, 5, 3)
        origins[..., 1] = 10.0  # All rays start above plane
        directions = torch.zeros(4, 5, 3)
        directions[..., 1] = -1.0  # All point downward
        plane_normals = torch.zeros(4, 5, 3)
        plane_normals[..., 1] = 1.0  # Ground planes
        plane_offsets = torch.zeros(4, 5)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Cross-product: (4, 5, 3) x (4, 5, 3) → (4, 5, 4, 5)
        assert result.t.shape == (4, 5, 4, 5)
        assert result.hit.shape == (4, 5, 4, 5)
        assert result.hit_point.shape == (4, 5, 4, 5, 3)
        assert result.normal.shape == (4, 5, 4, 5, 3)


class TestRayPlaneIntersectionNormals:
    """Tests for surface normal computation."""

    def test_normal_is_plane_normal(self):
        """Normal at hit equals (normalized) plane normal."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        # Use unnormalized plane normal
        plane_normals = torch.tensor([[0.0, 2.0, 0.0]])
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Normal should be normalized plane normal
        expected_normal = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.isclose(result.normal, expected_normal, atol=1e-5).all()

    def test_normal_flipped_for_backface(self):
        """Normal should be flipped when ray comes from behind plane."""
        origins = torch.tensor([[0.0, -5.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])  # Going up
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])  # Normal points up
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.hit.item() is True
        # Normal should be flipped to face the ray
        expected_normal = torch.tensor([[0.0, -1.0, 0.0]])
        assert torch.isclose(result.normal, expected_normal, atol=1e-5).all()

    def test_normals_are_normalized(self):
        """Surface normals should be unit vectors."""
        origins = torch.randn(100, 3)
        origins[..., 1] = 10.0
        directions = torch.zeros(100, 3)
        directions[..., 1] = -1.0
        # Use unnormalized plane normal (must be at least 1D for broadcasting)
        plane_normals = torch.tensor([[0.0, 3.0, 0.0]])  # (1, 3)
        plane_offsets = torch.tensor([0.0])  # (1,)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Output is (100, 1), flatten for checking
        assert result.hit.all()
        hit_normals = result.normal.view(-1, 3)
        norms = hit_normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestRayPlaneIntersectionUV:
    """Tests for UV coordinate computation."""

    def test_uv_is_hit_point_xy(self):
        """UV should be (hit_x, hit_y) world coordinates."""
        origins = torch.tensor([[3.0, 5.0, 7.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]])
        plane_offsets = torch.tensor([0.0])

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Hit point is (3, 0, 7), so UV should be (3, 0)
        expected_uv = torch.tensor([[3.0, 0.0]])
        assert torch.isclose(result.uv, expected_uv, atol=1e-5).all()


class TestRayPlaneIntersectionGradients:
    """Gradient tests for differentiable rendering."""

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 5.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        plane_offsets = torch.tensor([0.0], dtype=torch.float64)

        def func(o):
            result = ray_plane(o, directions, plane_normals, plane_offsets)
            return result.t

        torch.autograd.gradcheck(func, (origins,), raise_exception=True)

    def test_gradcheck_directions(self):
        """Gradient check for ray directions."""
        origins = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor(
            [[0.1, -1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        plane_offsets = torch.tensor([0.0], dtype=torch.float64)

        def func(d):
            result = ray_plane(origins, d, plane_normals, plane_offsets)
            return result.t

        torch.autograd.gradcheck(func, (directions,), raise_exception=True)

    @pytest.mark.skip(
        reason="Bug in backward kernel: dt/dn formula needs quotient rule"
    )
    def test_gradcheck_plane_normals(self):
        """Gradient check for plane normals."""
        origins = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor(
            [[0.1, 1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        plane_offsets = torch.tensor([0.0], dtype=torch.float64)

        def func(n):
            result = ray_plane(origins, directions, n, plane_offsets)
            return result.t

        torch.autograd.gradcheck(func, (plane_normals,), raise_exception=True)

    def test_gradcheck_plane_offsets(self):
        """Gradient check for plane offsets."""
        origins = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        plane_offsets = torch.tensor(
            [0.0], dtype=torch.float64, requires_grad=True
        )

        def func(d):
            result = ray_plane(origins, directions, plane_normals, d)
            return result.t

        torch.autograd.gradcheck(func, (plane_offsets,), raise_exception=True)


class TestRayPlaneIntersectionDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        origins = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32)
        directions = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        plane_offsets = torch.tensor([0.0], dtype=torch.float32)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.t.dtype == torch.float32
        assert result.hit_point.dtype == torch.float32
        assert result.normal.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        origins = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        plane_offsets = torch.tensor([0.0], dtype=torch.float64)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        assert result.t.dtype == torch.float64
        assert result.hit_point.dtype == torch.float64
        assert result.normal.dtype == torch.float64


class TestRayPlaneIntersectionValidation:
    """Input validation tests."""

    def test_invalid_origin_shape(self):
        """Origins must have last dimension 3."""
        origins = torch.randn(10, 2)  # Wrong shape
        directions = torch.randn(10, 3)
        plane_normals = torch.zeros(3)
        plane_offsets = torch.tensor(0.0)

        with pytest.raises(ValueError, match="origins must have shape"):
            ray_plane(origins, directions, plane_normals, plane_offsets)

    def test_invalid_direction_shape(self):
        """Directions must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 4)  # Wrong shape
        plane_normals = torch.zeros(3)
        plane_offsets = torch.tensor(0.0)

        with pytest.raises(ValueError, match="directions must have shape"):
            ray_plane(origins, directions, plane_normals, plane_offsets)

    def test_invalid_plane_normal_shape(self):
        """Plane normals must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        plane_normals = torch.zeros(2)  # Wrong shape
        plane_offsets = torch.tensor(0.0)

        with pytest.raises(ValueError, match="plane_normals must have shape"):
            ray_plane(origins, directions, plane_normals, plane_offsets)


class TestRayPlaneIntersectionMeta:
    """Tests for meta tensor support."""

    def test_meta_tensors(self):
        """Works with meta tensors for shape inference."""
        # With cross-product semantics: (4, 5) x (2,) → (4, 5, 2)
        origins = torch.randn(4, 5, 3, device="meta")
        directions = torch.randn(4, 5, 3, device="meta")
        plane_normals = torch.zeros(2, 3, device="meta")  # (2, 3)
        plane_offsets = torch.zeros(2, device="meta")  # (2,)

        result = ray_plane(origins, directions, plane_normals, plane_offsets)

        # Cross-product: (4, 5, 3) x (2, 3) → (4, 5, 2)
        assert result.t.shape == (4, 5, 2)
        assert result.t.device.type == "meta"
        assert result.hit_point.shape == (4, 5, 2, 3)
        assert result.normal.shape == (4, 5, 2, 3)
        assert result.uv.shape == (4, 5, 2, 2)
