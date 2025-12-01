"""Integration tests for BVH + geometry queries."""

import pytest
import torch

from torchscience.geometry import closest_point, ray_intersect, ray_occluded
from torchscience.space_partitioning import bounding_volume_hierarchy


class TestBVHIntegration:
    """End-to-end BVH tests."""

    @pytest.fixture
    def cube_bvh(self):
        """BVH with unit cube (12 triangles)."""
        vertices = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],  # front
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],  # back
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],  # front
                [4, 6, 5],
                [4, 7, 6],  # back
                [0, 4, 5],
                [0, 5, 1],  # bottom
                [2, 6, 7],
                [2, 7, 3],  # top
                [0, 3, 7],
                [0, 7, 4],  # left
                [1, 5, 6],
                [1, 6, 2],  # right
            ]
        )
        return bounding_volume_hierarchy([(vertices, faces)])

    def test_ray_through_cube(self, cube_bvh):
        """Ray through cube center."""
        origins = torch.tensor([[0.5, 0.5, -1.0]])
        directions = torch.tensor([[0.0, 0.0, 1.0]])

        hits = ray_intersect(cube_bvh, origins, directions)

        assert hits.hit.all()
        torch.testing.assert_close(
            hits.t, torch.tensor([1.0]), atol=1e-4, rtol=0
        )

    def test_shadow_ray(self, cube_bvh):
        """Shadow ray test."""
        # Light above, point inside cube should be occluded
        origins = torch.tensor([[0.5, 0.5, 0.5]])
        directions = torch.tensor([[0.0, 1.0, 0.0]])

        assert ray_occluded(cube_bvh, origins, directions).item()

    def test_closest_point_inside_cube(self, cube_bvh):
        """Point inside cube - closest to a face."""
        query = torch.tensor([[0.5, 0.5, 0.1]])  # Close to front face

        result = closest_point(cube_bvh, query)

        # Should be close to front face at z=0
        assert result.distance.item() < 0.2


class TestBVHMultiMesh:
    """Test BVH with multiple meshes."""

    @pytest.fixture
    def two_triangles_bvh(self):
        """BVH with two separate triangles."""
        # Triangle 1 at z=0
        v1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        f1 = torch.tensor([[0, 1, 2]])

        # Triangle 2 at z=2
        v2 = torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [0.5, 1.0, 2.0],
            ],
            dtype=torch.float32,
        )
        f2 = torch.tensor([[0, 1, 2]])

        return bounding_volume_hierarchy([(v1, f1), (v2, f2)])

    def test_ray_hits_first_triangle(self, two_triangles_bvh):
        """Ray from above should hit the farther triangle first."""
        origins = torch.tensor([[0.5, 0.5, 5.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        hits = ray_intersect(two_triangles_bvh, origins, directions)

        assert hits.hit.item()
        # Should hit triangle at z=2, distance is 5-2=3
        torch.testing.assert_close(
            hits.t, torch.tensor([3.0]), atol=1e-4, rtol=0
        )

    def test_ray_between_triangles(self, two_triangles_bvh):
        """Ray from between triangles going down."""
        origins = torch.tensor([[0.5, 0.5, 1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        hits = ray_intersect(two_triangles_bvh, origins, directions)

        assert hits.hit.item()
        # Should hit triangle at z=0, distance is 1
        torch.testing.assert_close(
            hits.t, torch.tensor([1.0]), atol=1e-4, rtol=0
        )


class TestBVHBatchOperations:
    """Test batch ray operations."""

    @pytest.fixture
    def simple_bvh(self):
        """Simple single triangle BVH."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]])
        return bounding_volume_hierarchy([(vertices, faces)])

    def test_batch_ray_intersect(self, simple_bvh):
        """Test multiple rays in a batch."""
        origins = torch.tensor(
            [
                [1.0, 1.0, 2.0],  # Above triangle center
                [1.0, 1.0, -2.0],  # Below triangle center
                [10.0, 10.0, 2.0],  # Far away, should miss
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Pointing down
                [0.0, 0.0, 1.0],  # Pointing up
                [0.0, 0.0, -1.0],  # Pointing down but will miss
            ]
        )

        hits = ray_intersect(simple_bvh, origins, directions)

        assert hits.t.shape == (3,)
        assert hits.hit[0].item()
        assert hits.hit[1].item()
        assert not hits.hit[2].item()

    def test_batch_closest_point(self, simple_bvh):
        """Test multiple closest point queries."""
        queries = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # Above triangle
                [1.0, 1.0, -1.0],  # Below triangle
            ]
        )

        result = closest_point(simple_bvh, queries)

        assert result.distance.shape == (2,)
        assert result.point.shape == (2, 3)
        # Both should have distance ~1
        torch.testing.assert_close(
            result.distance, torch.tensor([1.0, 1.0]), atol=1e-4, rtol=0
        )

    def test_batch_ray_occluded(self, simple_bvh):
        """Test multiple occlusion queries."""
        origins = torch.tensor(
            [
                [1.0, 1.0, 2.0],  # Above triangle
                [10.0, 10.0, 2.0],  # Far away
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Pointing at triangle
                [0.0, 0.0, -1.0],  # Pointing down but will miss
            ]
        )

        result = ray_occluded(simple_bvh, origins, directions)

        assert result.shape == (2,)
        assert result[0].item()
        assert not result[1].item()
