"""Tests for ray_occluded."""

import pytest
import torch

from torchscience.geometry import ray_occluded
from torchscience.space_partitioning import bounding_volume_hierarchy


class TestRayOccluded:
    """Ray occlusion tests."""

    @pytest.fixture
    def unit_triangle_bvh(self):
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

    def test_occluded_ray(self, unit_triangle_bvh):
        """Ray hitting triangle should be occluded."""
        origins = torch.tensor([[0.25, 0.25, 1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        result = ray_occluded(unit_triangle_bvh, origins, directions)

        assert result.dtype == torch.bool
        assert result.item() is True

    def test_unoccluded_ray(self, unit_triangle_bvh):
        """Ray missing triangle should not be occluded."""
        origins = torch.tensor([[5.0, 5.0, 1.0]])
        directions = torch.tensor([[0.0, 0.0, -1.0]])

        result = ray_occluded(unit_triangle_bvh, origins, directions)

        assert result.item() is False

    def test_batch_occlusion(self, unit_triangle_bvh):
        """Multiple rays."""
        origins = torch.tensor(
            [
                [0.25, 0.25, 1.0],  # Hit
                [5.0, 5.0, 1.0],  # Miss
            ]
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

        result = ray_occluded(unit_triangle_bvh, origins, directions)

        assert result.shape == (2,)
        assert result[0].item() is True
        assert result[1].item() is False
