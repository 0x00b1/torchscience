"""Tests for bounding_volume_hierarchy."""

import torch

from torchscience.space_partitioning import (
    BoundingVolumeHierarchy,
    bounding_volume_hierarchy,
)


class TestBVHConstruction:
    """Test BVH construction."""

    def test_single_mesh_construction(self):
        """Test constructing BVH from single mesh."""
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = torch.tensor([[0, 1, 2]])

        bvh = bounding_volume_hierarchy([(vertices, faces)])

        assert isinstance(bvh, BoundingVolumeHierarchy)
        assert bvh.vertices.shape == (3, 3)
        assert bvh.faces.shape == (1, 3)

    def test_multi_mesh_construction(self):
        """Test constructing BVH from multiple meshes."""
        v1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        f1 = torch.tensor([[0, 1, 2]])
        v2 = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        f2 = torch.tensor([[0, 1, 2]])

        bvh = bounding_volume_hierarchy([(v1, f1), (v2, f2)])

        assert bvh.vertices.shape == (6, 3)
        assert bvh.faces.shape == (2, 3)
        assert bvh.mesh_offsets.shape == (3,)  # M + 1 = 2 + 1

    def test_returns_tensorclass(self):
        """Test that BVH is a tensorclass with expected attributes."""
        vertices = torch.randn(10, 3)
        faces = torch.randint(0, 10, (5, 3))

        bvh = bounding_volume_hierarchy([(vertices, faces)])

        # Should have all required attributes
        assert hasattr(bvh, "vertices")
        assert hasattr(bvh, "faces")
        assert hasattr(bvh, "mesh_offsets")
        assert hasattr(bvh, "face_offsets")
        assert hasattr(bvh, "_scene_handles")
