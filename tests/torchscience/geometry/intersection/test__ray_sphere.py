"""Tests for ray_sphere intersection."""

import pytest
import torch

from torchscience.geometry.intersection import IntersectionResult, ray_sphere


class TestRaySphereIntersectionBasic:
    """Basic intersection tests."""

    def test_ray_hits_sphere_from_front(self):
        """Ray from (0,0,-5) toward (0,0,1) hitting unit sphere at origin."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert isinstance(result, IntersectionResult)
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        # Hit point should be at (0, 0, -1) on the sphere surface
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()
        # Normal should point outward (toward ray origin)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_sphere_from_behind(self):
        """Ray hitting sphere from behind (z positive direction)."""
        origins = torch.tensor([[0.0, 0.0, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        # Hit point should be at (0, 0, 1)
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()

    def test_ray_misses_sphere_parallel(self):
        """Ray from (10,0,-5) toward (0,0,1) should miss unit sphere at origin."""
        origins = torch.tensor([[10.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_sphere_away(self):
        """Ray pointing away from sphere should miss."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])  # Pointing away
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()


class TestRaySphereGrazingHit:
    """Tests for ray tangent to sphere (single intersection)."""

    def test_ray_grazes_sphere_xy(self):
        """Ray tangent to sphere at (1, 0, 0)."""
        # Origin at (1, 0, -5), direction (0, 0, 1) should graze at (1, 0, 0)
        origins = torch.tensor([[1.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        # Should graze at t=5, point=(1, 0, 0)
        assert torch.isclose(result.t, torch.tensor([[5.0]]), atol=1e-4).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-4
        ).all()

    def test_ray_grazes_sphere_negative_y(self):
        """Ray tangent to sphere at (0, -1, 0)."""
        origins = torch.tensor([[0.0, -1.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[5.0]]), atol=1e-4).all()

    def test_barely_miss(self):
        """Ray that barely misses the sphere."""
        # Slightly outside the tangent line
        origins = torch.tensor([[1.001, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is False


class TestRaySphereInsideSphere:
    """Tests for ray originating inside sphere."""

    def test_ray_from_center(self):
        """Ray from sphere center should hit at radius distance."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])  # At center
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-5).all()
        # Hit point should be (0, 0, 1)
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()

    def test_ray_from_inside_not_center(self):
        """Ray from inside sphere but not at center."""
        origins = torch.tensor([[0.5, 0.0, 0.0]])  # Inside sphere
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        # Should hit at (1, 0, 0), distance = 0.5
        assert torch.isclose(result.t, torch.tensor([[0.5]]), atol=1e-5).all()

    def test_ray_from_inside_opposite_direction(self):
        """Ray from inside going opposite direction."""
        origins = torch.tensor([[0.5, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        # Should hit at (-1, 0, 0), distance = 1.5
        assert torch.isclose(result.t, torch.tensor([[1.5]]), atol=1e-5).all()


class TestRaySphereBroadcasting:
    """Tests for batched operations with cross-product broadcasting.

    NOTE: ray_sphere uses cross-product broadcasting semantics:
    rays (N, 3) x spheres (M, 3) -> (N, M) output
    """

    def test_batch_rays_single_sphere(self):
        """Multiple rays against single sphere."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -5.0],  # Hit from front
                [10.0, 0.0, -5.0],  # Miss (offset)
                [0.0, 0.0, 5.0],  # Hit from behind
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        centers = torch.tensor([[0.0, 0.0, 0.0]])  # (1, 3)
        radii = torch.tensor([1.0])  # (1,)

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (3, 3) x (1, 3) -> (3, 1)
        assert result.t.shape == (3, 1)
        assert result.hit[0, 0].item() is True  # Front hit
        assert result.hit[1, 0].item() is False  # Miss
        assert result.hit[2, 0].item() is True  # Behind hit

    def test_single_ray_batch_spheres(self):
        """One ray against multiple spheres."""
        origins = torch.tensor([[0.0, 0.0, -10.0]])  # (1, 3)
        directions = torch.tensor([[0.0, 0.0, 1.0]])  # (1, 3)
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Centered sphere
                [0.0, 0.0, 5.0],  # Sphere at z=5
                [100.0, 0.0, 0.0],  # Offset sphere (miss)
            ]
        )  # (3, 3)
        radii = torch.tensor([1.0, 1.0, 1.0])  # (3,)

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (1, 3) x (3, 3) -> (1, 3)
        assert result.t.shape == (1, 3)
        assert result.hit[0, 0].item() is True  # Hits centered sphere
        assert result.hit[0, 1].item() is True  # Hits z=5 sphere
        assert result.hit[0, 2].item() is False  # Misses offset sphere

    def test_broadcast_rays_and_spheres(self):
        """Rays (N,3) x Spheres (M,3) gives cross-product (N,M) output."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -10.0],
                [0.0, 0.0, -20.0],
            ]
        )  # (2, 3)
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )  # (2, 3)
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 10.0],
            ]
        )  # (3, 3)
        radii = torch.tensor([1.0, 1.0, 1.0])  # (3,)

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (2, 3) x (3, 3) -> (2, 3)
        assert result.t.shape == (2, 3)
        # Ray 0 (from z=-10): distances to spheres at z=0, z=5, z=10
        assert torch.isclose(
            result.t[0, 0], torch.tensor(9.0), atol=1e-5
        )  # z=-10 to z=-1
        assert torch.isclose(
            result.t[0, 1], torch.tensor(14.0), atol=1e-5
        )  # z=-10 to z=4
        assert torch.isclose(
            result.t[0, 2], torch.tensor(19.0), atol=1e-5
        )  # z=-10 to z=9
        # Ray 1 (from z=-20): distances to spheres at z=0, z=5, z=10
        assert torch.isclose(result.t[1, 0], torch.tensor(19.0), atol=1e-5)
        assert torch.isclose(result.t[1, 1], torch.tensor(24.0), atol=1e-5)
        assert torch.isclose(result.t[1, 2], torch.tensor(29.0), atol=1e-5)

    def test_2d_batch(self):
        """2D batch of rays against 2D batch of spheres."""
        origins = torch.randn(4, 5, 3)
        origins[..., 2] = -10.0  # All rays start far back
        directions = torch.zeros(4, 5, 3)
        directions[..., 2] = 1.0  # All point toward +z
        centers = torch.zeros(2, 3, 3)
        centers[..., 2] = 0.0  # Spheres at z=0
        radii = torch.ones(2, 3)

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (4, 5, 3) x (2, 3, 3) -> (4, 5, 2, 3)
        assert result.t.shape == (4, 5, 2, 3)
        assert result.hit.shape == (4, 5, 2, 3)
        assert result.hit_point.shape == (4, 5, 2, 3, 3)
        assert result.normal.shape == (4, 5, 2, 3, 3)


class TestRaySphereNormals:
    """Tests for surface normal computation."""

    def test_normal_at_poles(self):
        """Normals at sphere poles should be axis-aligned."""
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        # Ray hitting +z pole
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

        # Ray hitting +y pole
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-5
        ).all()

        # Ray hitting +x pole
        origins = torch.tensor([[5.0, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.normal, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()

    def test_normals_are_normalized(self):
        """Surface normals should be unit vectors."""
        origins = torch.randn(100, 3)
        origins = (
            origins / origins.norm(dim=-1, keepdim=True) * 5.0
        )  # On sphere of radius 5
        directions = -origins  # Point toward center
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        # Check normalization for hits
        hit_mask = result.hit.squeeze(-1)
        hit_normals = result.normal[hit_mask]
        norms = hit_normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_normal_points_outward_from_outside(self):
        """Normal should point away from center when hit from outside."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        # Hit point is (0, 0, -1), normal should point toward (0, 0, -1)
        expected_normal = torch.tensor([[[0.0, 0.0, -1.0]]])
        assert torch.isclose(result.normal, expected_normal, atol=1e-5).all()


class TestRaySphereUV:
    """Tests for UV coordinate computation (spherical coordinates).

    UV convention: u = (atan2(y, x) + pi) / (2*pi), v = acos(z) / pi.
    Poles are along the z-axis: +z has v=0, -z has v=1.
    The y-axis points are on the equator (v=0.5).
    """

    def test_uv_at_z_poles(self):
        """UV at z-axis poles should have v=0 (+z) or v=1 (-z)."""
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        # Hit +z pole: outward normal is (0, 0, 1), phi=acos(1)=0, v=0
        origins = torch.tensor([[0.0, 0.0, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[0.0]]), atol=1e-5
        ).all()

        # Hit -z pole: outward normal is (0, 0, -1), phi=acos(-1)=pi, v=1
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[1.0]]), atol=1e-5
        ).all()

    def test_uv_at_equator(self):
        """UV at equator (y-axis) should have v=0.5."""
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        # Hit +y point: outward normal is (0, 1, 0), phi=acos(0)=pi/2, v=0.5
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.uv[..., 1], torch.tensor([[0.5]]), atol=1e-5
        ).all()

    def test_uv_u_wraps_around(self):
        """U coordinate should wrap around the sphere [0, 1]."""
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        # Hit +x point: outward normal is (1, 0, 0)
        # theta = atan2(0, 1) = 0, u = (0 + pi) / (2*pi) = 0.5
        origins = torch.tensor([[5.0, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        result = ray_sphere(origins, directions, centers, radii)
        assert torch.isclose(
            result.uv[..., 0], torch.tensor([[0.5]]), atol=1e-5
        ).all()


class TestRaySphereGradients:
    """Gradient tests for differentiable rendering."""

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 0.0, -5.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64)

        def func(o):
            result = ray_sphere(o, directions, centers, radii)
            return result.t

        torch.autograd.gradcheck(func, (origins,), raise_exception=True)

    def test_gradcheck_directions(self):
        """Gradient check for ray directions."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor(
            [[0.1, 0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64)

        def func(d):
            result = ray_sphere(origins, d, centers, radii)
            return result.t

        torch.autograd.gradcheck(func, (directions,), raise_exception=True)

    def test_gradcheck_centers(self):
        """Gradient check for sphere centers."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        radii = torch.tensor([1.0], dtype=torch.float64)

        def func(c):
            result = ray_sphere(origins, directions, c, radii)
            return result.t

        torch.autograd.gradcheck(func, (centers,), raise_exception=True)

    def test_gradcheck_radii(self):
        """Gradient check for sphere radii."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(r):
            result = ray_sphere(origins, directions, centers, r)
            return result.t

        torch.autograd.gradcheck(func, (radii,), raise_exception=True)


class TestRaySphereSecondOrderGradients:
    """Second-order gradient tests."""

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_sphere"
    )
    def test_gradgradcheck_origins(self):
        """Second-order gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 0.0, -5.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64)

        def func(o):
            result = ray_sphere(o, directions, centers, radii)
            return result.t

        torch.autograd.gradgradcheck(func, (origins,), raise_exception=True)

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_sphere"
    )
    def test_gradgradcheck_centers(self):
        """Second-order gradient check for sphere centers."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        radii = torch.tensor([1.0], dtype=torch.float64)

        def func(c):
            result = ray_sphere(origins, directions, c, radii)
            return result.t

        torch.autograd.gradgradcheck(func, (centers,), raise_exception=True)

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_sphere"
    )
    def test_gradgradcheck_radii(self):
        """Second-order gradient check for sphere radii."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def func(r):
            result = ray_sphere(origins, directions, centers, r)
            return result.t

        torch.autograd.gradgradcheck(func, (radii,), raise_exception=True)


class TestRaySphereMeta:
    """Tests for meta tensor support."""

    def test_meta_tensors_basic(self):
        """Works with meta tensors for shape inference."""
        origins = torch.randn(4, 5, 3, device="meta")
        directions = torch.randn(4, 5, 3, device="meta")
        centers = torch.zeros(2, 3, device="meta")  # (2, 3)
        radii = torch.zeros(2, device="meta")  # (2,)

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (4, 5, 3) x (2, 3) -> (4, 5, 2)
        assert result.t.shape == (4, 5, 2)
        assert result.t.device.type == "meta"
        assert result.hit_point.shape == (4, 5, 2, 3)
        assert result.normal.shape == (4, 5, 2, 3)
        assert result.uv.shape == (4, 5, 2, 2)

    def test_meta_tensors_single_ray_multiple_spheres(self):
        """Meta tensor shape inference with single ray, multiple spheres."""
        origins = torch.randn(1, 3, device="meta")
        directions = torch.randn(1, 3, device="meta")
        centers = torch.zeros(10, 3, device="meta")
        radii = torch.zeros(10, device="meta")

        result = ray_sphere(origins, directions, centers, radii)

        assert result.t.shape == (1, 10)
        assert result.hit.shape == (1, 10)


class TestRaySphereSpecialValues:
    """Tests for special values: large/small radii, distant spheres."""

    @pytest.mark.parametrize("radius", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_various_radii(self, radius):
        """Hit detection works for various sphere sizes."""
        distance = radius * 5
        origins = torch.tensor([[0.0, 0.0, -distance]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([radius])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        expected_t = distance - radius
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-4
        ).all()

    def test_distant_sphere(self):
        """Intersection with distant sphere."""
        origins = torch.tensor([[0.0, 0.0, -1e6]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        expected_t = 1e6 - 1.0
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-4
        ).all()

    def test_very_small_radius(self):
        """Intersection with very small sphere."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1e-4])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        expected_t = 1.0 - 1e-4
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-3
        ).all()

    def test_sphere_at_nonzero_center(self):
        """Intersection with off-center sphere."""
        origins = torch.tensor([[5.0, 3.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[5.0, 3.0, 2.0]])  # Offset center
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        # Distance to near intersection: center_z - origin_z - radius = 2 - (-5) - 1 = 6
        assert torch.isclose(result.t, torch.tensor([[6.0]]), atol=1e-5).all()


class TestRaySphereDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        radii = torch.tensor([1.0], dtype=torch.float32)

        result = ray_sphere(origins, directions, centers, radii)

        assert result.t.dtype == torch.float32
        assert result.hit_point.dtype == torch.float32
        assert result.normal.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        centers = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        radii = torch.tensor([1.0], dtype=torch.float64)

        result = ray_sphere(origins, directions, centers, radii)

        assert result.t.dtype == torch.float64
        assert result.hit_point.dtype == torch.float64
        assert result.normal.dtype == torch.float64


class TestRaySphereValidation:
    """Input validation tests."""

    def test_invalid_origin_shape(self):
        """Origins must have last dimension 3."""
        origins = torch.randn(10, 2)  # Wrong shape
        directions = torch.randn(10, 3)
        centers = torch.zeros(1, 3)
        radii = torch.tensor([1.0])

        with pytest.raises(ValueError, match="origins must have shape"):
            ray_sphere(origins, directions, centers, radii)

    def test_invalid_direction_shape(self):
        """Directions must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 4)  # Wrong shape
        centers = torch.zeros(1, 3)
        radii = torch.tensor([1.0])

        with pytest.raises(ValueError, match="directions must have shape"):
            ray_sphere(origins, directions, centers, radii)

    def test_invalid_centers_shape(self):
        """Centers must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        centers = torch.zeros(1, 2)  # Wrong shape
        radii = torch.tensor([1.0])

        with pytest.raises(ValueError, match="centers must have shape"):
            ray_sphere(origins, directions, centers, radii)


class TestRaySphereEdgeCases:
    """Edge case tests."""

    def test_unnormalized_directions(self):
        """Works with non-unit direction vectors."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 10.0]])  # Length 10
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        assert result.hit.item() is True
        # t is parametric, so should be 4/10 = 0.4
        assert torch.isclose(result.t, torch.tensor([[0.4]]), atol=1e-5).all()
        # Hit point should still be correct at (0, 0, -1)
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

    def test_ray_on_surface(self):
        """Ray starting exactly on sphere surface.

        The kernel uses t >= 0, so t=0 is a valid intersection
        (self-intersection at the start point).
        """
        origins = torch.tensor([[0.0, 0.0, -1.0]])  # On surface
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor([[0.0, 0.0, 0.0]])
        radii = torch.tensor([1.0])

        result = ray_sphere(origins, directions, centers, radii)

        # t=0 is the near intersection (self-intersection at surface)
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[0.0]]), atol=1e-4).all()

    def test_multiple_spheres_same_line(self):
        """Multiple spheres along the same ray should find nearest hit."""
        origins = torch.tensor([[0.0, 0.0, -20.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Nearest sphere
                [0.0, 0.0, 5.0],  # Middle sphere
                [0.0, 0.0, 10.0],  # Far sphere
            ]
        )
        radii = torch.tensor([1.0, 1.0, 1.0])

        result = ray_sphere(origins, directions, centers, radii)

        # Cross-product: (1, 3) x (3, 3) -> (1, 3)
        # Each column is the hit for one sphere
        assert result.t.shape == (1, 3)
        # Distances to front surface of each sphere
        assert torch.isclose(
            result.t[0, 0], torch.tensor(19.0), atol=1e-5
        )  # z=-1
        assert torch.isclose(
            result.t[0, 1], torch.tensor(24.0), atol=1e-5
        )  # z=4
        assert torch.isclose(
            result.t[0, 2], torch.tensor(29.0), atol=1e-5
        )  # z=9
