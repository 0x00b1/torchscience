"""Tests for ray_sphere_intersection."""

import pytest
import torch

from torchscience.geometry import RaySphereHit, ray_sphere_intersection


class TestRaySphereIntersectionBasic:
    """Basic intersection tests."""

    def test_ray_hits_sphere_from_front(self):
        """Ray pointing at sphere center should hit front face."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert isinstance(result, RaySphereHit)
        assert result.hit.item() is True
        assert result.front_face.item() is True
        assert torch.isclose(result.t, torch.tensor([2.0]), atol=1e-5).all()
        # Hit point should be on sphere surface
        assert torch.isclose(
            result.point, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5
        ).all()
        # Normal should point toward ray origin (outward)
        assert torch.isclose(
            result.normal, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5
        ).all()

    def test_ray_hits_sphere_from_inside(self):
        """Ray starting inside sphere should hit back face."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])  # Inside sphere
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True
        assert result.front_face.item() is False  # Back face
        assert torch.isclose(result.t, torch.tensor([1.0]), atol=1e-5).all()
        # Normal should still face ray origin (flipped inward)
        assert torch.isclose(
            result.normal, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5
        ).all()

    def test_ray_misses_sphere(self):
        """Ray pointing away from sphere should miss."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])  # Away from sphere
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_sphere_offset(self):
        """Ray parallel to sphere should miss."""
        origins = torch.tensor([[0.0, 2.0, -3.0]])  # Offset in Y
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is False

    def test_ray_grazes_sphere(self):
        """Ray tangent to sphere should hit exactly once."""
        origins = torch.tensor([[0.0, 1.0, -3.0]])  # Tangent point
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True
        # t should be distance to tangent point
        assert torch.isclose(result.t, torch.tensor([3.0]), atol=1e-5).all()


class TestRaySphereIntersectionTRange:
    """Tests for t_min and t_max parameters."""

    def test_t_min_excludes_near_intersection(self):
        """t_min should exclude intersections too close."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        # t=2 is the near intersection, t=4 is the far
        result = ray_sphere_intersection(
            origins, directions, center, radius, t_min=2.5
        )

        assert result.hit.item() is True
        # Should hit far intersection at t=4
        assert torch.isclose(result.t, torch.tensor([4.0]), atol=1e-5).all()

    def test_t_max_excludes_far_intersection(self):
        """t_max should exclude intersections too far."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(
            origins, directions, center, radius, t_max=1.5
        )

        # Both intersections are beyond t_max
        assert result.hit.item() is False

    def test_t_range_excludes_all(self):
        """Both intersections outside range should miss."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(
            origins, directions, center, radius, t_min=2.5, t_max=3.5
        )

        assert result.hit.item() is False

    def test_self_intersection_avoidance(self):
        """Default t_min should prevent self-intersection artifacts."""
        # Start exactly on sphere surface
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        # Should hit far side, not self-intersect at t≈0
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([2.0]), atol=1e-3).all()


class TestRaySphereIntersectionBatched:
    """Tests for batched operations."""

    def test_batch_rays_single_sphere(self):
        """Multiple rays against single sphere."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -3.0],  # Hit
                [0.0, 2.0, -3.0],  # Miss (offset)
                [0.0, 0.0, 3.0],  # Hit from other side
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.shape == (3,)
        assert result.hit[0].item() is True
        assert result.hit[1].item() is False
        assert result.hit[2].item() is True

    def test_2d_batch(self):
        """2D batch of rays."""
        origins = torch.randn(4, 5, 3)
        origins[..., 2] = -5.0  # All rays start behind sphere
        directions = torch.zeros(4, 5, 3)
        directions[..., 2] = 1.0  # All point toward origin
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.shape == (4, 5)
        assert result.hit.shape == (4, 5)
        assert result.point.shape == (4, 5, 3)
        assert result.normal.shape == (4, 5, 3)

    def test_broadcasting_centers(self):
        """Broadcast single center to batch of rays."""
        origins = torch.tensor([[0.0, 0.0, -3.0], [0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])  # Single center
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.shape == (2,)
        assert result.hit.all()

    def test_broadcasting_radii(self):
        """Scalar radius broadcasts to all rays."""
        origins = torch.randn(10, 3)
        origins[..., 2] = -5.0
        directions = torch.zeros(10, 3)
        directions[..., 2] = 1.0
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 2.0  # Scalar

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.shape == (10,)


class TestRaySphereIntersectionNormals:
    """Tests for surface normal computation."""

    def test_normal_at_poles(self):
        """Normals at sphere poles should be axis-aligned."""
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        # Ray hitting north pole
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        result = ray_sphere_intersection(origins, directions, center, radius)
        assert torch.isclose(
            result.normal, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5
        ).all()

        # Ray hitting south pole
        origins = torch.tensor([[0.0, -5.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])
        result = ray_sphere_intersection(origins, directions, center, radius)
        assert torch.isclose(
            result.normal, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5
        ).all()

    def test_normals_are_normalized(self):
        """Surface normals should be unit vectors."""
        origins = torch.randn(100, 3)
        origins[..., 2] = -5.0
        directions = -origins  # Point toward origin
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        # Check normalization for hits
        hit_normals = result.normal[result.hit]
        norms = hit_normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestRaySphereIntersectionDifferentRadii:
    """Tests for various sphere sizes."""

    @pytest.mark.parametrize("radius", [0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    def test_various_radii(self, radius):
        """Hit detection works for various sphere sizes."""
        origins = torch.tensor([[0.0, 0.0, -radius * 3]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True
        expected_t = radius * 3 - radius  # Distance to near intersection
        assert torch.isclose(
            result.t, torch.tensor([expected_t]), rtol=1e-4
        ).all()


class TestRaySphereIntersectionGradients:
    """Gradient tests for differentiable rendering."""

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 0.0, -3.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        radius = 1.0

        def func(o):
            result = ray_sphere_intersection(o, directions, center, radius)
            return result.t

        torch.autograd.gradcheck(func, (origins,), raise_exception=True)

    def test_gradcheck_directions(self):
        """Gradient check for ray directions."""
        origins = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64)
        directions = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        radius = 1.0

        def func(d):
            result = ray_sphere_intersection(origins, d, center, radius)
            return result.t

        torch.autograd.gradcheck(func, (directions,), raise_exception=True)

    def test_gradcheck_centers(self):
        """Gradient check for sphere centers."""
        origins = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        center = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True
        )
        radius = 1.0

        def func(c):
            result = ray_sphere_intersection(origins, directions, c, radius)
            return result.t

        torch.autograd.gradcheck(func, (center,), raise_exception=True)

    def test_gradcheck_radii(self):
        """Gradient check for sphere radii."""
        origins = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        radius = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(r):
            result = ray_sphere_intersection(origins, directions, center, r)
            return result.t

        torch.autograd.gradcheck(func, (radius,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        origins = torch.tensor(
            [[0.0, 0.0, -3.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        radius = 1.0

        def func(o):
            result = ray_sphere_intersection(o, directions, center, radius)
            return result.t

        torch.autograd.gradgradcheck(func, (origins,), raise_exception=True)

    def test_gradient_hit_point(self):
        """Gradient of hit point with respect to origin."""
        origins = torch.tensor(
            [[0.0, 0.0, -3.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        # Compute gradient of z-coordinate of hit point
        result.point[0, 2].backward()

        # d(hit_z)/d(origin_z) = d(origin_z + t * 1)/d(origin_z)
        # = 1 + d(t)/d(origin_z)
        # For ray along z-axis hitting sphere, d(t)/d(origin_z) = -1
        # So d(hit_z)/d(origin_z) = 0 (hit point on sphere doesn't move)
        expected_grad = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(origins.grad, expected_grad, atol=1e-5)


class TestRaySphereIntersectionDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float_dtypes(self, dtype):
        """Works with float32 and float64."""
        origins = torch.tensor([[0.0, 0.0, -3.0]], dtype=dtype)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.dtype == dtype
        assert result.point.dtype == dtype
        assert result.normal.dtype == dtype


class TestRaySphereIntersectionEdgeCases:
    """Edge case tests."""

    def test_unnormalized_directions(self):
        """Works with non-unit direction vectors."""
        origins = torch.tensor([[0.0, 0.0, -3.0]])
        directions = torch.tensor([[0.0, 0.0, 10.0]])  # Length 10
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True
        # t is parametric, so should be 2/10 = 0.2
        assert torch.isclose(result.t, torch.tensor([0.2]), atol=1e-5).all()
        # Hit point should still be correct
        hit_point = origins + result.t.unsqueeze(-1) * directions
        assert torch.isclose(
            hit_point, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5
        ).all()

    def test_very_small_sphere(self):
        """Intersection with very small sphere."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1e-6

        result = ray_sphere_intersection(
            origins, directions, center, radius, t_min=0.0
        )

        assert result.hit.item() is True

    def test_very_large_sphere(self):
        """Intersection with very large sphere."""
        origins = torch.tensor([[0.0, 0.0, -1e6]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        radius = 1e5

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True

    def test_sphere_at_nonzero_center(self):
        """Intersection with off-center sphere."""
        origins = torch.tensor([[5.0, 3.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        center = torch.tensor([5.0, 3.0, 2.0])  # Offset center
        radius = 1.0

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.hit.item() is True
        # Distance to near intersection
        expected_t = 2.0 - (-5.0) - 1.0  # center_z - origin_z - radius
        assert torch.isclose(result.t, torch.tensor([6.0]), atol=1e-5).all()


class TestRaySphereIntersectionValidation:
    """Input validation tests."""

    def test_invalid_origins_shape(self):
        """Origins must have last dimension 3."""
        origins = torch.randn(10, 2)  # Wrong shape
        directions = torch.randn(10, 3)
        center = torch.zeros(3)

        with pytest.raises(ValueError, match="origins must have shape"):
            ray_sphere_intersection(origins, directions, center, 1.0)

    def test_invalid_directions_shape(self):
        """Directions must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 4)  # Wrong shape
        center = torch.zeros(3)

        with pytest.raises(ValueError, match="directions must have shape"):
            ray_sphere_intersection(origins, directions, center, 1.0)

    def test_invalid_centers_shape(self):
        """Centers must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        center = torch.zeros(2)  # Wrong shape

        with pytest.raises(ValueError, match="centers must have shape"):
            ray_sphere_intersection(origins, directions, center, 1.0)


class TestRaySphereIntersectionMeta:
    """Tests for meta tensor support."""

    def test_meta_tensors(self):
        """Works with meta tensors for shape inference."""
        origins = torch.randn(4, 5, 3, device="meta")
        directions = torch.randn(4, 5, 3, device="meta")
        center = torch.zeros(3, device="meta")
        radius = torch.tensor(1.0, device="meta")

        result = ray_sphere_intersection(origins, directions, center, radius)

        assert result.t.shape == (4, 5)
        assert result.t.device.type == "meta"
        assert result.point.shape == (4, 5, 3)
        assert result.normal.shape == (4, 5, 3)
