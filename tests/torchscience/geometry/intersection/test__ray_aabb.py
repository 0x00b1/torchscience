"""Tests for ray_aabb intersection."""

import pytest
import torch

from torchscience.geometry.intersection import IntersectionResult, ray_aabb


class TestRayAABBBasicHits:
    """Basic intersection tests."""

    def test_ray_hits_unit_box_from_z_negative(self):
        """Ray from (0,0,-5) toward (0,0,1) hitting unit box [-1,1]^3."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert isinstance(result, IntersectionResult)
        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        # Hit point should be (0, 0, -1) on the -z face
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()
        # Normal should point outward from the -z face
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_positive_x_face(self):
        """Ray hitting the +x face of the unit box."""
        origins = torch.tensor([[5.0, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.normal, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_negative_x_face(self):
        """Ray hitting the -x face of the unit box."""
        origins = torch.tensor([[-5.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[-1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.normal, torch.tensor([[[-1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_positive_y_face(self):
        """Ray hitting the +y face of the unit box."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_negative_y_face(self):
        """Ray hitting the -y face of the unit box."""
        origins = torch.tensor([[0.0, -5.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, -1.0, 0.0]]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, -1.0, 0.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_positive_z_face(self):
        """Ray hitting the +z face of the unit box."""
        origins = torch.tensor([[0.0, 0.0, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        assert torch.isclose(result.t, torch.tensor([[4.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()

    def test_ray_hits_non_unit_box(self):
        """Ray hitting a non-unit box [0,2] x [0,3] x [0,4]."""
        origins = torch.tensor([[1.0, 1.5, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[0.0, 0.0, 0.0]])
        box_max = torch.tensor([[2.0, 3.0, 4.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # t = (0 - (-5)) / 1 = 5
        assert torch.isclose(result.t, torch.tensor([[5.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[1.0, 1.5, 0.0]]]), atol=1e-5
        ).all()
        # Hits -z face (min face on z axis)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()


class TestRayAABBMissCases:
    """Tests for rays that miss the box."""

    def test_ray_misses_parallel_offset(self):
        """Ray parallel to a face but offset misses the box."""
        origins = torch.tensor([[10.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_pointing_away(self):
        """Ray pointing away from the box misses."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_diagonally(self):
        """Ray passing diagonally between box faces misses."""
        origins = torch.tensor([[3.0, 3.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is False
        assert torch.isinf(result.t).all()

    def test_ray_misses_along_y_axis(self):
        """Ray offset along y axis misses."""
        origins = torch.tensor([[0.0, 5.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is False


class TestRayAABBFaceNormals:
    """Tests for axis-aligned face normal computation."""

    def test_normal_negative_x_face(self):
        """Normal on -x face should be (-1, 0, 0)."""
        origins = torch.tensor([[-5.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[-1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()

    def test_normal_positive_x_face(self):
        """Normal on +x face should be (1, 0, 0)."""
        origins = torch.tensor([[5.0, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-5
        ).all()

    def test_normal_negative_y_face(self):
        """Normal on -y face should be (0, -1, 0)."""
        origins = torch.tensor([[0.0, -5.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, -1.0, 0.0]]]), atol=1e-5
        ).all()

    def test_normal_positive_y_face(self):
        """Normal on +y face should be (0, 1, 0)."""
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[0.0, -1.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-5
        ).all()

    def test_normal_negative_z_face(self):
        """Normal on -z face should be (0, 0, -1)."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

    def test_normal_positive_z_face(self):
        """Normal on +z face should be (0, 0, 1)."""
        origins = torch.tensor([[0.0, 0.0, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)
        assert torch.isclose(
            result.normal, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()


class TestRayAABBInsideBox:
    """Tests for ray originating inside the box."""

    def test_ray_from_center_of_box(self):
        """Ray from center should hit an exit face."""
        origins = torch.tensor([[0.0, 0.0, 0.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # Should exit through +z face at t=1
        assert torch.isclose(result.t, torch.tensor([[1.0]]), atol=1e-5).all()
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, 1.0]]]), atol=1e-5
        ).all()

    def test_ray_from_inside_not_center(self):
        """Ray from inside the box but not at center."""
        origins = torch.tensor([[0.5, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # Should exit through +x face at t=0.5
        assert torch.isclose(result.t, torch.tensor([[0.5]]), atol=1e-5).all()

    def test_ray_from_inside_negative_direction(self):
        """Ray from inside going in negative direction."""
        origins = torch.tensor([[0.5, 0.0, 0.0]])
        directions = torch.tensor([[-1.0, 0.0, 0.0]])
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # Should exit through -x face at t=1.5
        assert torch.isclose(result.t, torch.tensor([[1.5]]), atol=1e-5).all()


class TestRayAABBBroadcasting:
    """Tests for batched operations with cross-product broadcasting.

    NOTE: ray_aabb uses cross-product broadcasting semantics:
    rays (N, 3) x boxes (M, 3) -> (N, M) output
    """

    def test_batch_rays_single_box(self):
        """Multiple rays against single box."""
        origins = torch.tensor(
            [
                [0.0, 0.0, -5.0],  # Hit from -z
                [10.0, 0.0, -5.0],  # Miss (offset)
                [0.0, 0.0, 5.0],  # Hit from +z
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])  # (1, 3)
        box_max = torch.tensor([[1.0, 1.0, 1.0]])  # (1, 3)

        result = ray_aabb(origins, directions, box_min, box_max)

        # Cross-product: (3, 3) x (1, 3) -> (3, 1)
        assert result.t.shape == (3, 1)
        assert result.hit[0, 0].item() is True  # Front hit
        assert result.hit[1, 0].item() is False  # Miss
        assert result.hit[2, 0].item() is True  # Behind hit

    def test_single_ray_batch_boxes(self):
        """One ray against multiple boxes."""
        origins = torch.tensor([[0.0, 0.0, -10.0]])  # (1, 3)
        directions = torch.tensor([[0.0, 0.0, 1.0]])  # (1, 3)
        box_min = torch.tensor(
            [
                [-1.0, -1.0, -1.0],  # Box at origin
                [-1.0, -1.0, 4.0],  # Box at z=5
                [100.0, 100.0, 100.0],  # Far away box (miss)
            ]
        )  # (3, 3)
        box_max = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 6.0],
                [101.0, 101.0, 101.0],
            ]
        )  # (3, 3)

        result = ray_aabb(origins, directions, box_min, box_max)

        # Cross-product: (1, 3) x (3, 3) -> (1, 3)
        assert result.t.shape == (1, 3)
        assert result.hit[0, 0].item() is True  # Hits origin box
        assert result.hit[0, 1].item() is True  # Hits z=5 box
        assert result.hit[0, 2].item() is False  # Misses far box

    def test_broadcast_rays_and_boxes(self):
        """Rays (N,3) x Boxes (M,3) gives cross-product (N,M) output."""
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
        box_min = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 4.0],
                [-1.0, -1.0, 9.0],
            ]
        )  # (3, 3)
        box_max = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 6.0],
                [1.0, 1.0, 11.0],
            ]
        )  # (3, 3)

        result = ray_aabb(origins, directions, box_min, box_max)

        # Cross-product: (2, 3) x (3, 3) -> (2, 3)
        assert result.t.shape == (2, 3)
        # Ray 0 (from z=-10): hits -z face of each box
        assert torch.isclose(
            result.t[0, 0], torch.tensor(9.0), atol=1e-5
        )  # z=-10 to z=-1
        assert torch.isclose(
            result.t[0, 1], torch.tensor(14.0), atol=1e-5
        )  # z=-10 to z=4
        assert torch.isclose(
            result.t[0, 2], torch.tensor(19.0), atol=1e-5
        )  # z=-10 to z=9
        # Ray 1 (from z=-20): hits -z face of each box
        assert torch.isclose(result.t[1, 0], torch.tensor(19.0), atol=1e-5)
        assert torch.isclose(result.t[1, 1], torch.tensor(24.0), atol=1e-5)
        assert torch.isclose(result.t[1, 2], torch.tensor(29.0), atol=1e-5)

    def test_2d_batch(self):
        """2D batch of rays against 2D batch of boxes."""
        origins = torch.zeros(4, 5, 3)
        origins[..., 2] = -10.0
        directions = torch.zeros(4, 5, 3)
        directions[..., 2] = 1.0
        box_min = torch.zeros(2, 3, 3)
        box_min[..., :] = -1.0
        box_max = torch.zeros(2, 3, 3)
        box_max[..., :] = 1.0

        result = ray_aabb(origins, directions, box_min, box_max)

        # Cross-product: (4, 5, 3) x (2, 3, 3) -> (4, 5, 2, 3)
        assert result.t.shape == (4, 5, 2, 3)
        assert result.hit.shape == (4, 5, 2, 3)
        assert result.hit_point.shape == (4, 5, 2, 3, 3)
        assert result.normal.shape == (4, 5, 2, 3, 3)


class TestRayAABBGradients:
    """Gradient tests for differentiable rendering."""

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 0.0, -5.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        def func(o):
            result = ray_aabb(o, directions, box_min, box_max)
            return result.t

        torch.autograd.gradcheck(func, (origins,), raise_exception=True)

    def test_gradcheck_directions(self):
        """Gradient check for ray directions."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        def func(d):
            result = ray_aabb(origins, d, box_min, box_max)
            return result.t

        torch.autograd.gradcheck(func, (directions,), raise_exception=True)

    def test_gradcheck_box_min(self):
        """Gradient check for box_min."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor(
            [[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True
        )
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        def func(bmin):
            result = ray_aabb(origins, directions, bmin, box_max)
            return result.t

        torch.autograd.gradcheck(func, (box_min,), raise_exception=True)

    def test_gradcheck_box_max(self):
        """Gradient check for box_max."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor(
            [[1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True
        )

        def func(bmax):
            result = ray_aabb(origins, directions, box_min, bmax)
            return result.t

        torch.autograd.gradcheck(func, (box_max,), raise_exception=True)

    def test_gradcheck_all_inputs(self):
        """Gradient check with all inputs requiring gradients.

        Uses a ray hitting well in the center of a face for gradient stability.
        """
        origins = torch.tensor(
            [[0.0, 0.0, -5.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True
        )
        box_min = torch.tensor(
            [[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True
        )
        box_max = torch.tensor(
            [[1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True
        )

        def func(o, d, bmin, bmax):
            result = ray_aabb(o, d, bmin, bmax)
            return result.t

        torch.autograd.gradcheck(
            func, (origins, directions, box_min, box_max), raise_exception=True
        )


class TestRayAABBSecondOrderGradients:
    """Second-order gradient tests."""

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_aabb"
    )
    def test_gradgradcheck_origins(self):
        """Second-order gradient check for ray origins."""
        origins = torch.tensor(
            [[0.0, 0.0, -5.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        def func(o):
            result = ray_aabb(o, directions, box_min, box_max)
            return result.t

        torch.autograd.gradgradcheck(func, (origins,), raise_exception=True)

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_aabb"
    )
    def test_gradgradcheck_box_min(self):
        """Second-order gradient check for box_min."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor(
            [[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True
        )
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        def func(bmin):
            result = ray_aabb(origins, directions, bmin, box_max)
            return result.t

        torch.autograd.gradgradcheck(func, (box_min,), raise_exception=True)

    @pytest.mark.skip(
        reason="backward_backward kernel not yet implemented for ray_aabb"
    )
    def test_gradgradcheck_box_max(self):
        """Second-order gradient check for box_max."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor(
            [[1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True
        )

        def func(bmax):
            result = ray_aabb(origins, directions, box_min, bmax)
            return result.t

        torch.autograd.gradgradcheck(func, (box_max,), raise_exception=True)


class TestRayAABBMeta:
    """Tests for meta tensor support."""

    def test_meta_tensors_basic(self):
        """Works with meta tensors for shape inference."""
        origins = torch.randn(4, 5, 3, device="meta")
        directions = torch.randn(4, 5, 3, device="meta")
        box_min = torch.zeros(2, 3, device="meta")  # (2, 3)
        box_max = torch.zeros(2, 3, device="meta")  # (2, 3)

        result = ray_aabb(origins, directions, box_min, box_max)

        # Cross-product: (4, 5, 3) x (2, 3) -> (4, 5, 2)
        assert result.t.shape == (4, 5, 2)
        assert result.t.device.type == "meta"
        assert result.hit_point.shape == (4, 5, 2, 3)
        assert result.normal.shape == (4, 5, 2, 3)
        assert result.uv.shape == (4, 5, 2, 2)

    def test_meta_tensors_single_ray_multiple_boxes(self):
        """Meta tensor shape inference with single ray, multiple boxes."""
        origins = torch.randn(1, 3, device="meta")
        directions = torch.randn(1, 3, device="meta")
        box_min = torch.zeros(10, 3, device="meta")
        box_max = torch.zeros(10, 3, device="meta")

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.t.shape == (1, 10)
        assert result.hit.shape == (1, 10)


class TestRayAABBEdgeCases:
    """Edge case tests."""

    def test_unnormalized_directions(self):
        """Works with non-unit direction vectors."""
        origins = torch.tensor([[0.0, 0.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 10.0]])  # Length 10
        box_min = torch.tensor([[-1.0, -1.0, -1.0]])
        box_max = torch.tensor([[1.0, 1.0, 1.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # t is parametric: ((-1) - (-5)) / 10 = 0.4
        assert torch.isclose(result.t, torch.tensor([[0.4]]), atol=1e-5).all()
        # Hit point should still be correct at (0, 0, -1)
        assert torch.isclose(
            result.hit_point, torch.tensor([[[0.0, 0.0, -1.0]]]), atol=1e-5
        ).all()

    def test_very_large_box(self):
        """Intersection with a very large box."""
        origins = torch.tensor([[0.0, 0.0, -1e5]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1e4, -1e4, -1e4]])
        box_max = torch.tensor([[1e4, 1e4, 1e4]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        expected_t = 1e5 - 1e4
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-4
        ).all()

    def test_very_small_box(self):
        """Intersection with a very small box."""
        origins = torch.tensor([[0.0, 0.0, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-1e-4, -1e-4, -1e-4]])
        box_max = torch.tensor([[1e-4, 1e-4, 1e-4]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        expected_t = 1.0 - 1e-4
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-3
        ).all()

    def test_box_at_nonzero_position(self):
        """Intersection with box at a non-origin position."""
        origins = torch.tensor([[5.0, 3.0, -5.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[4.0, 2.0, 1.0]])
        box_max = torch.tensor([[6.0, 4.0, 3.0]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        # t = (1 - (-5)) / 1 = 6
        assert torch.isclose(result.t, torch.tensor([[6.0]]), atol=1e-5).all()

    @pytest.mark.parametrize(
        "half_size", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    def test_various_box_sizes(self, half_size):
        """Hit detection works for various box sizes."""
        distance = half_size * 5
        origins = torch.tensor([[0.0, 0.0, -distance]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])
        box_min = torch.tensor([[-half_size, -half_size, -half_size]])
        box_max = torch.tensor([[half_size, half_size, half_size]])

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.hit.item() is True
        expected_t = distance - half_size
        assert torch.isclose(
            result.t, torch.tensor([[expected_t]]), rtol=1e-4
        ).all()


class TestRayAABBValidation:
    """Input validation tests."""

    def test_invalid_origin_shape(self):
        """Origins must have last dimension 3."""
        origins = torch.randn(10, 2)  # Wrong shape
        directions = torch.randn(10, 3)
        box_min = torch.zeros(1, 3)
        box_max = torch.ones(1, 3)

        with pytest.raises(ValueError, match="origins must have shape"):
            ray_aabb(origins, directions, box_min, box_max)

    def test_invalid_direction_shape(self):
        """Directions must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 4)  # Wrong shape
        box_min = torch.zeros(1, 3)
        box_max = torch.ones(1, 3)

        with pytest.raises(ValueError, match="directions must have shape"):
            ray_aabb(origins, directions, box_min, box_max)

    def test_invalid_box_min_shape(self):
        """box_min must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        box_min = torch.zeros(1, 2)  # Wrong shape
        box_max = torch.ones(1, 3)

        with pytest.raises(ValueError, match="box_min must have shape"):
            ray_aabb(origins, directions, box_min, box_max)

    def test_invalid_box_max_shape(self):
        """box_max must have last dimension 3."""
        origins = torch.randn(10, 3)
        directions = torch.randn(10, 3)
        box_min = torch.zeros(1, 3)
        box_max = torch.ones(1, 2)  # Wrong shape

        with pytest.raises(ValueError, match="box_max must have shape"):
            ray_aabb(origins, directions, box_min, box_max)


class TestRayAABBDtypes:
    """Tests for different data types."""

    def test_float32(self):
        """Works with float32."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.t.dtype == torch.float32
        assert result.hit_point.dtype == torch.float32
        assert result.normal.dtype == torch.float32

    def test_float64(self):
        """Works with float64."""
        origins = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
        directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        box_min = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
        box_max = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)

        result = ray_aabb(origins, directions, box_min, box_max)

        assert result.t.dtype == torch.float64
        assert result.hit_point.dtype == torch.float64
        assert result.normal.dtype == torch.float64
