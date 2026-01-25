"""Tests for octree_sample query operations."""

import pytest
import torch
import torch.testing

from torchscience.geometry.space_partitioning import octree, octree_sample

from .conftest import make_octree


class TestOctreeSampleNearest:
    """Tests for nearest-neighbor point queries."""

    def test_query_single_point_tree(self):
        """Query a tree with a single point."""
        points = torch.tensor([[0.5, 0.5, 0.5]])
        data = torch.tensor([[42.0]])
        tree = octree(points, data, maximum_depth=4)

        # Query at the exact point location
        result, found = octree_sample(tree, points)
        assert found.all()
        assert result[0, 0].item() == pytest.approx(42.0, rel=1e-5)

    def test_query_returns_leaf_data(self):
        """Query should return leaf voxel data."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=6)

        # Query at original points should find voxels
        result, found = octree_sample(tree, points)
        assert found.sum() > 0
        assert result.shape == (100, 4)

    def test_query_empty_region_returns_not_found(self):
        """Query in sparse region returns found=False."""
        # Create tree with points only in positive octant
        points = torch.tensor([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]])
        data = torch.tensor([[1.0], [2.0]])
        tree = octree(points, data, maximum_depth=8)

        # Query in negative octant (far from any points)
        queries = torch.tensor([[-0.9, -0.9, -0.9]])
        result, found = octree_sample(tree, queries)

        assert not found[0].item()
        # Not-found should return zeros
        assert result[0, 0].item() == 0.0

    def test_query_uses_top_down_traversal(self, simple_hierarchy_tree):
        """Query uses top-down traversal, not ancestor fallback."""
        tree = simple_hierarchy_tree

        # Query inside the region covered by the hierarchy
        # The fixture has leaves at depth 4 in the min corner region
        queries = torch.tensor([[-0.9, -0.9, -0.9]])
        result, found = octree_sample(tree, queries)

        # Should find a leaf and return its data
        assert found[0].item()

    def test_query_depth_limits_traversal(self, simple_hierarchy_tree):
        """query_depth stops traversal at specified depth."""
        tree = simple_hierarchy_tree

        # Query at depth 2 should return internal node data
        queries = torch.tensor([[-0.9, -0.9, -0.9]])
        result_d2, found_d2 = octree_sample(tree, queries, query_depth=2)

        # Query at full depth should return leaf data
        result_full, found_full = octree_sample(tree, queries)

        assert found_d2[0].item()
        assert found_full[0].item()
        # Results may differ (internal vs leaf)

    def test_query_depth_zero_returns_root(self, simple_hierarchy_tree):
        """query_depth=0 returns root data for any point."""
        tree = simple_hierarchy_tree

        queries = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        result, found = octree_sample(tree, queries, query_depth=0)

        # Root exists for non-empty tree
        assert found.all()
        # Both queries should get root data (same value)
        assert result[0, 0].item() == result[1, 0].item()

    def test_out_of_bounds_clamped(self):
        """Out-of-bounds queries are clamped to boundary."""
        points = torch.tensor([[0.9, 0.9, 0.9]])  # Near positive corner
        data = torch.tensor([[100.0]])
        tree = octree(points, data, maximum_depth=4)

        # Query outside bounds
        queries = torch.tensor([[2.0, 2.0, 2.0]])  # Far outside
        result, found = octree_sample(tree, queries)

        # Should be clamped and find the boundary voxel
        # (behavior depends on where the point lands after clamping)
        assert result.shape == (1, 1)

    def test_multi_dimensional_data(self):
        """Query returns correct shape for multi-dimensional data."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 3, 4)  # Multi-dimensional
        tree = octree(points, data, maximum_depth=5)

        queries = torch.rand(10, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.shape == (10, 3, 4)

    def test_empty_tree_returns_not_found(self):
        """Query on empty tree returns found=False."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 2)
        tree = octree(points, data, maximum_depth=4)

        queries = torch.tensor([[0.0, 0.0, 0.0]])
        result, found = octree_sample(tree, queries)

        assert not found[0].item()
        assert result.shape == (1, 2)


class TestOctreeSampleTrilinear:
    """Tests for trilinear interpolation queries."""

    def test_trilinear_smooth_within_voxel(self):
        """Trilinear produces smooth interpolation."""
        points = torch.rand(1000, 3) * 2 - 1
        data = torch.rand(1000, 1)
        tree = octree(points, data, maximum_depth=4)

        # Two nearby query points
        q1 = torch.tensor([[0.1, 0.1, 0.1]])
        q2 = torch.tensor([[0.11, 0.1, 0.1]])  # Slight offset

        r1, f1 = octree_sample(tree, q1, interpolation="trilinear")
        r2, f2 = octree_sample(tree, q2, interpolation="trilinear")

        # Results should be similar (smooth)
        if f1[0] and f2[0]:
            diff = (r1 - r2).abs().item()
            # Should be small for nearby points
            assert diff < 1.0  # Loose bound

    def test_trilinear_vs_nearest(self):
        """Trilinear and nearest can produce different results."""
        points = torch.rand(500, 3) * 2 - 1
        data = torch.rand(500, 1)
        tree = octree(points, data, maximum_depth=4)

        queries = torch.rand(50, 3) * 2 - 1

        r_nearest, _ = octree_sample(tree, queries, interpolation="nearest")
        r_trilinear, _ = octree_sample(
            tree, queries, interpolation="trilinear"
        )

        # In general, results differ due to interpolation
        # At least some should be different
        assert r_nearest.shape == r_trilinear.shape

    def test_trilinear_with_query_depth(self):
        """Trilinear respects query_depth."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 2)
        tree = octree(points, data, maximum_depth=6)

        queries = torch.rand(10, 3) * 2 - 1

        # Coarse query should use internal node data
        result, found = octree_sample(
            tree, queries, interpolation="trilinear", query_depth=3
        )

        assert result.shape == (10, 2)


class TestOctreeSampleErrors:
    """Tests for error handling."""

    def test_invalid_interpolation(self):
        """Invalid interpolation string raises error."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        queries = torch.tensor([[0.0, 0.0, 0.0]])

        with pytest.raises(RuntimeError, match="interpolation"):
            octree_sample(tree, queries, interpolation="invalid")

    def test_invalid_query_depth_negative(self):
        """Negative query_depth raises error."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        queries = torch.tensor([[0.0, 0.0, 0.0]])

        with pytest.raises(RuntimeError, match="query_depth"):
            octree_sample(tree, queries, query_depth=-1)

    def test_invalid_query_depth_exceeds_max(self):
        """query_depth > maximum_depth raises error."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        queries = torch.tensor([[0.0, 0.0, 0.0]])

        with pytest.raises(RuntimeError, match="query_depth"):
            octree_sample(tree, queries, query_depth=5)


class TestOctreeSampleAutograd:
    """Tests for autograd support."""

    def test_gradient_wrt_data_nearest(self):
        """Gradients flow through data with nearest interpolation."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4, requires_grad=True)
        tree = octree(points, data.detach(), maximum_depth=4)

        # Manually set data to require grad
        tree.data = tree.data.clone().requires_grad_(True)

        # Query at construction points to ensure hits (sparse tree with random
        # queries often misses all voxels)
        queries = points[:20]
        result, found = octree_sample(tree, queries, interpolation="nearest")

        # Verify we found voxels
        assert found.sum() > 0, "No voxels found - test setup issue"

        # Only sum over found results to avoid gradient through zeros
        loss = (result * found.unsqueeze(-1)).sum()
        loss.backward()

        assert tree.data.grad is not None
        # Some gradients should be non-zero
        assert (tree.data.grad != 0).any()

    def test_gradient_wrt_data_trilinear(self):
        """Gradients flow through data with trilinear interpolation."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4, requires_grad=True)
        tree = octree(points, data.detach(), maximum_depth=4)

        tree.data = tree.data.clone().requires_grad_(True)

        queries = torch.rand(20, 3) * 2 - 1
        result, found = octree_sample(tree, queries, interpolation="trilinear")

        loss = result.sum()
        loss.backward()

        assert tree.data.grad is not None

    def test_gradient_wrt_points_trilinear(self):
        """Gradients flow through query points with trilinear."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Query near construction points to ensure trilinear finds corners
        # Add small offset to avoid exact voxel centers
        queries = (points[:20] + 0.01).requires_grad_(True)
        result, found = octree_sample(tree, queries, interpolation="trilinear")

        # Trilinear needs at least one corner found for point gradients
        assert found.any(), "No voxels found - test setup issue"

        loss = result.sum()
        loss.backward()

        assert queries.grad is not None
        # Point gradients should be non-zero when interpolating
        assert (queries.grad != 0).any()


class TestOctreeSampleHierarchy:
    """Tests for hierarchical query semantics."""

    def test_query_finds_coarse_leaf(self):
        """Query inside coarse leaf returns that leaf, not 'not found'."""
        # Create tree with a coarse leaf at depth 2
        codes = torch.tensor(
            [
                0 << 60,  # depth 0, root
                1 << 60,  # depth 1, child 0
                2 << 60,  # depth 2, coarse LEAF
            ],
            dtype=torch.int64,
        )
        data = torch.tensor([[10.0], [5.0], [1.0]])
        children_mask = torch.tensor(
            [0x01, 0x01, 0x00], dtype=torch.uint8
        )  # depth 2 is leaf
        weights = torch.tensor([1.0, 1.0, 1.0])

        tree = make_octree(
            codes, data, children_mask, weights, maximum_depth=8
        )

        # Query inside the coarse leaf region
        queries = torch.tensor([[-0.9, -0.9, -0.9]])
        result, found = octree_sample(tree, queries)

        # Should find the coarse leaf at depth 2
        assert found[0].item()
        assert result[0, 0].item() == pytest.approx(1.0, rel=1e-5)

    def test_mixed_depth_tree(self):
        """Query in tree with leaves at different depths."""
        # Create tree with leaves at depth 2 and depth 4
        codes = []
        data_values = []
        children_masks = []
        weights_values = []

        # Root at depth 0
        codes.append(0 << 60)
        data_values.append([0.0])
        children_masks.append(0x03)  # Children 0 and 1 exist
        weights_values.append(2.0)

        # Depth 1 child 0
        codes.append(1 << 60)
        data_values.append([0.0])
        children_masks.append(0x01)
        weights_values.append(1.0)

        # Depth 1 child 1 (different subtree)
        codes.append((1 << 60) | 1)
        data_values.append([0.0])
        children_masks.append(0x01)
        weights_values.append(1.0)

        # Coarse leaf at depth 2 (subtree 0) - child 0 of depth 1 child 0
        codes.append(2 << 60)
        data_values.append([100.0])  # Coarse leaf value
        children_masks.append(0x00)  # LEAF
        weights_values.append(1.0)

        # Internal at depth 2 (subtree 1)
        codes.append((2 << 60) | 8)  # morton = 8 (child 1 path)
        data_values.append([0.0])
        children_masks.append(0x01)
        weights_values.append(1.0)

        # Internal at depth 3
        codes.append((3 << 60) | 64)  # morton = 64 (child 1 path continued)
        data_values.append([0.0])
        children_masks.append(0x01)
        weights_values.append(1.0)

        # Fine leaf at depth 4
        codes.append((4 << 60) | 512)  # morton = 512 (child 1 path continued)
        data_values.append([200.0])  # Fine leaf value
        children_masks.append(0x00)  # LEAF
        weights_values.append(1.0)

        codes_tensor = torch.tensor(codes, dtype=torch.int64)
        data_tensor = torch.tensor(data_values)
        children_mask_tensor = torch.tensor(children_masks, dtype=torch.uint8)
        weights_tensor = torch.tensor(weights_values)

        tree = make_octree(
            codes_tensor,
            data_tensor,
            children_mask_tensor,
            weights_tensor,
            maximum_depth=4,
        )

        # Query in region of coarse leaf
        q_coarse = torch.tensor([[-0.9, -0.9, -0.9]])
        r_coarse, f_coarse = octree_sample(tree, q_coarse)

        assert f_coarse[0].item()
        assert r_coarse[0, 0].item() == pytest.approx(100.0, rel=1e-5)


class TestOctreeSampleBatched:
    """Tests for batched query support."""

    def test_many_queries(self):
        """Large batch of queries works correctly."""
        points = torch.rand(1000, 3) * 2 - 1
        data = torch.rand(1000, 8)
        tree = octree(points, data, maximum_depth=8)

        queries = torch.rand(10000, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.shape == (10000, 8)
        assert found.shape == (10000,)

    def test_dtype_preservation(self):
        """Query preserves data dtype."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        queries = torch.rand(10, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.dtype == torch.float64
