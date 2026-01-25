"""Tests for ray_intersect."""

import pytest
import torch

from torchscience.geometry import RayHit, ray_intersect
from torchscience.geometry.space_partitioning import bounding_volume_hierarchy


class TestRayIntersectBasic:
    """Basic ray intersection tests."""

    @pytest.fixture
    def unit_triangle_bvh(self):
        """BVH with single unit triangle in XY plane at z=0."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]])
        return bounding_volume_hierarchy([(vertices, faces)])

    def test_ray_hits_triangle(self, unit_triangle_bvh):
        """Ray pointing at triangle center should hit."""
        origins = torch.tensor([[0.25, 0.25, 1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        hits = ray_intersect(unit_triangle_bvh, origins, directions)

        assert isinstance(hits, RayHit)
        assert hits.hit.item() is True
        assert torch.isclose(hits.t, torch.tensor([1.0]), atol=1e-5).all()

    def test_ray_misses_triangle(self, unit_triangle_bvh):
        """Ray pointing away from triangle should miss."""
        origins = torch.tensor([[0.25, 0.25, 1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])  # Away from triangle

        hits = ray_intersect(unit_triangle_bvh, origins, directions)

        assert hits.hit.item() is False
        assert torch.isinf(hits.t).all()

    def test_batch_rays(self, unit_triangle_bvh):
        """Multiple rays should return batch results."""
        origins = torch.tensor(
            [
                [0.25, 0.25, 1.0],
                [0.25, 0.25, -1.0],
                [5.0, 5.0, 1.0],  # Miss
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

        hits = ray_intersect(unit_triangle_bvh, origins, directions)

        assert hits.t.shape == (3,)
        assert hits.hit[0].item() is True
        assert hits.hit[1].item() is True
        assert hits.hit[2].item() is False


class TestRayIntersectBarycentrics:
    """Test barycentric coordinates."""

    def test_barycentrics_at_vertex(self):
        """Ray hitting vertex should have correct barycentrics."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]])
        bvh = bounding_volume_hierarchy([(vertices, faces)])

        # Ray hitting vertex 0 (origin)
        origins = torch.tensor([[0.0, 0.0, 1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        hits = ray_intersect(bvh, origins, directions)

        # At vertex 0: u=0, v=0 (w = 1-u-v = 1)
        assert torch.isclose(hits.u, torch.tensor([0.0]), atol=1e-5).all()
        assert torch.isclose(hits.v, torch.tensor([0.0]), atol=1e-5).all()


class TestRayIntersectGradient:
    """Gradient tests for ray_intersect."""

    def test_gradcheck_origins(self):
        """Gradient check for ray origins."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        faces = torch.tensor([[0, 1, 2]])
        bvh = bounding_volume_hierarchy([(vertices, faces)])

        origins = torch.tensor(
            [[0.25, 0.25, 1.0]], dtype=torch.float64, requires_grad=True
        )
        directions = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)

        def func(o):
            hits = ray_intersect(bvh, o, directions)
            # Return hit point
            return o + hits.t.unsqueeze(-1) * directions

        # Use relaxed tolerances because gradient approximation has small errors
        torch.autograd.gradcheck(
            func, (origins,), atol=0.02, rtol=0.1, raise_exception=True
        )
