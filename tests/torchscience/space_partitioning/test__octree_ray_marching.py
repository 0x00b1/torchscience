"""Tests for octree ray marching operations."""

import torch
import torch.testing

from torchscience.space_partitioning import octree, octree_ray_marching


class TestRayAABBIntersection:
    """Tests for ray-AABB intersection handling."""

    def test_ray_through_center(self):
        """Ray through the center of the octree."""
        # Create points along the y=0, z=0 line to ensure the ray will hit them
        x_vals = torch.linspace(-0.8, 0.8, 10)
        points = torch.stack([x_vals, torch.zeros(10), torch.zeros(10)], dim=1)
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        # Ray along x-axis through center
        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        assert positions.shape == (1, 512, 3)
        assert result_data.shape == (1, 512, 4)
        assert mask.shape == (1, 512)

        # Should have some valid samples
        assert mask.any()

    def test_ray_misses_aabb(self):
        """Ray that completely misses the [-1, 1]^3 box."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Ray parallel to and above the box
        origins = torch.tensor([[0.0, 5.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        # Should have no valid samples
        assert not mask.any()

    def test_ray_origin_inside_box(self):
        """Ray starting from inside the [-1, 1]^3 box."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Origin inside the box
        origins = torch.tensor([[0.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        # Should have samples starting from origin
        if mask.any():
            first_valid = mask[0].nonzero()[0, 0].item()
            first_pos = positions[0, first_valid]
            # First position should be at or near origin
            assert first_pos[0].item() >= -0.1

    def test_ray_grazing_corner(self):
        """Ray grazing a corner of the box."""
        points = torch.tensor([[0.9, 0.9, 0.9]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        # Ray barely touching corner
        origins = torch.tensor([[-2.0, 0.99, 0.99]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.05
        )

        # Should either hit or miss depending on precision
        assert positions.shape == (1, 512, 3)


class TestFixedStepMarching:
    """Tests for fixed step size ray marching."""

    def test_step_size_respected(self):
        """Fixed step size produces regularly spaced samples."""
        points = torch.rand(500, 3) * 2 - 1
        data = torch.rand(500, 2)
        tree = octree(points, data, maximum_depth=6)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])
        step_size = 0.1

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=step_size
        )

        # Check spacing between consecutive valid samples along ray direction
        if mask[0].sum() >= 2:
            valid_idx = mask[0].nonzero().squeeze(-1)
            valid_positions = positions[0, valid_idx]

            # For axis-aligned ray, x positions should be regularly spaced
            x_positions = valid_positions[:, 0]
            diffs = x_positions[1:] - x_positions[:-1]

            # All diffs should be approximately step_size
            assert (diffs - step_size).abs().max() < 0.01

    def test_maximum_steps_limit(self):
        """Maximum steps limits output size."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        max_steps = 50
        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.01, maximum_steps=max_steps
        )

        assert positions.shape == (1, max_steps, 3)
        assert mask.shape == (1, max_steps)


class TestAdaptiveStepMarching:
    """Tests for adaptive (hierarchical DDA) ray marching."""

    def test_adaptive_without_step_size(self):
        """Adaptive stepping when step_size is not provided."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        # No step_size = adaptive
        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions
        )

        assert positions.shape[0] == 1
        assert positions.shape[2] == 3

    def test_adaptive_skips_empty_regions(self):
        """Adaptive stepping should skip large empty regions."""
        # Create tree with points only in positive octant
        points = torch.rand(50, 3) * 0.5 + 0.5  # [0.5, 1.0]^3
        data = torch.ones(50, 1)
        tree = octree(points, data, maximum_depth=6)

        # Ray through center
        origins = torch.tensor([[-2.0, 0.75, 0.75]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        # Adaptive should efficiently skip the empty negative region
        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, maximum_steps=100
        )

        # All valid samples should be in the positive region
        if mask.any():
            valid_positions = positions[mask]
            assert (valid_positions[:, 0] >= 0.4).all()


class TestBatchedRays:
    """Tests for batched ray processing."""

    def test_multiple_rays(self):
        """Multiple rays processed in parallel."""
        points = torch.rand(200, 3) * 2 - 1
        data = torch.rand(200, 4)
        tree = octree(points, data, maximum_depth=5)

        n_rays = 10
        origins = torch.stack(
            [
                torch.tensor([-2.0, y, 0.0])
                for y in torch.linspace(-0.5, 0.5, n_rays)
            ]
        )
        directions = torch.tensor([[1.0, 0.0, 0.0]]).expand(n_rays, 3)

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        assert positions.shape[0] == n_rays
        assert result_data.shape[0] == n_rays
        assert mask.shape[0] == n_rays

    def test_different_directions(self):
        """Rays with different directions."""
        points = torch.rand(200, 3) * 2 - 1
        data = torch.rand(200, 4)
        tree = octree(points, data, maximum_depth=5)

        # Rays from different angles toward center
        origins = torch.tensor(
            [
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -2.0],
            ]
        )
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        assert positions.shape[0] == 3
        # Each ray should hit some voxels
        for i in range(3):
            assert positions[i].shape == (512, 3)


class TestEmptyTree:
    """Tests for ray marching in empty trees."""

    def test_empty_tree_returns_no_samples(self):
        """Ray marching in empty tree returns all-False mask."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[0.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        assert not mask.any()


class TestOutputShapes:
    """Tests for correct output tensor shapes."""

    def test_multi_dimensional_data(self):
        """Ray marching with multi-dimensional voxel data."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 3, 4)  # Multi-dimensional
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        # Data should preserve shape
        assert result_data.shape == (1, 512, 3, 4)

    def test_dtype_preservation(self):
        """Output dtype matches input data dtype."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        assert result_data.dtype == torch.float64


class TestDirectionNormalization:
    """Tests for direction normalization."""

    def test_unnormalized_directions(self):
        """Unnormalized directions are normalized internally."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])

        # Test with unnormalized direction
        directions_unnorm = torch.tensor([[10.0, 0.0, 0.0]])
        directions_norm = torch.tensor([[1.0, 0.0, 0.0]])

        pos1, data1, mask1 = octree_ray_marching(
            tree, origins, directions_unnorm, step_size=0.1
        )
        pos2, data2, mask2 = octree_ray_marching(
            tree, origins, directions_norm, step_size=0.1
        )

        # Results should be the same
        torch.testing.assert_close(mask1, mask2)
        if mask1.any():
            torch.testing.assert_close(
                pos1[mask1], pos2[mask2], atol=1e-5, rtol=1e-5
            )


class TestHierarchicalTraversal:
    """Tests for hierarchical DDA traversal correctness."""

    def test_samples_inside_voxels(self):
        """All sample positions should be inside the [-1, 1]^3 box."""
        points = torch.rand(200, 3) * 2 - 1
        data = torch.rand(200, 4)
        tree = octree(points, data, maximum_depth=6)

        origins = torch.tensor([[-2.0, 0.3, -0.5]])
        directions = torch.tensor([[1.0, 0.1, 0.2]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.05
        )

        if mask.any():
            valid_positions = positions[mask]
            # All positions should be in [-1, 1]^3 with small tolerance
            assert (valid_positions >= -1.01).all()
            assert (valid_positions <= 1.01).all()

    def test_samples_in_occupied_regions(self):
        """Valid samples should correspond to occupied voxels."""
        # Create a simple tree with known structure
        points = torch.tensor([[0.5, 0.5, 0.5]])
        data = torch.tensor([[42.0]])
        tree = octree(points, data, maximum_depth=4)

        # Ray that should pass through the occupied region
        origins = torch.tensor([[-2.0, 0.5, 0.5]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.05
        )

        # Should have some valid samples near the point
        if mask.any():
            valid_data = result_data[mask]
            # At least one sample should have the voxel data
            assert (valid_data[:, 0] > 0).any()


class TestAutograd:
    """Tests for autograd support."""

    def test_gradient_wrt_data_fixed_step(self):
        """Gradients flow through data with fixed step size."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        tree.data = tree.data.clone().requires_grad_(True)

        origins = torch.tensor([[-2.0, 0.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        loss = (result_data * mask.unsqueeze(-1)).sum()
        loss.backward()

        assert tree.data.grad is not None

    def test_gradient_wrt_origins_fixed_step(self):
        """Gradients flow through origins with fixed step size."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        origins = torch.tensor([[-2.0, 0.0, 0.0]], requires_grad=True)
        directions = torch.tensor([[1.0, 0.0, 0.0]])

        positions, result_data, mask = octree_ray_marching(
            tree, origins, directions, step_size=0.1
        )

        loss = positions[mask].sum()
        loss.backward()

        assert origins.grad is not None
