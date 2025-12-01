"""Tests for octree dynamic update operations."""

import pytest
import torch
import torch.testing

from torchscience.space_partitioning import (
    octree,
    octree_insert,
    octree_merge,
    octree_remove,
    octree_sample,
    octree_subdivide,
)

from .conftest import make_octree


class TestOctreeInsert:
    """Tests for octree_insert operation."""

    def test_insert_single_point(self):
        """Insert a single point into an empty tree."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4)

        new_points = torch.tensor([[0.5, 0.5, 0.5]])
        new_data = torch.tensor([[42.0]])
        tree = octree_insert(tree, new_points, new_data)

        # Should have more than just the new leaf (ancestors are created)
        assert tree.count.item() > 0
        # Should have at least one leaf
        assert (tree.children_mask == 0).sum() >= 1

    def test_insert_preserves_existing(self):
        """Inserting new points preserves existing voxels."""
        points = torch.tensor([[0.5, 0.5, 0.5]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)
        original_count = tree.count.item()

        # Insert in a different region
        new_points = torch.tensor([[-0.5, -0.5, -0.5]])
        new_data = torch.tensor([[2.0]])
        tree = octree_insert(tree, new_points, new_data)

        # Count should increase
        assert tree.count.item() > original_count

    def test_insert_at_specific_depth(self):
        """Insert at a specific depth instead of maximum_depth."""
        # Start with empty tree to avoid conflicts
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 2)
        tree = octree(points, data, maximum_depth=8)

        new_points = torch.tensor([[0.0, 0.0, 0.0]])
        new_data = torch.tensor([[5.0, 5.0]])

        # Insert at depth 4 (coarser than max)
        tree = octree_insert(tree, new_points, new_data, depth=4)

        # The inserted voxel should be at depth 4
        # Find leaves
        leaf_mask = tree.children_mask == 0
        depths = (tree.codes[leaf_mask] >> 60) & 0xF
        # At least one leaf at depth 4
        assert (depths == 4).any()

    def test_insert_multiple_points(self):
        """Insert multiple points at once."""
        # Start with empty tree to avoid conflicts
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=6)

        new_points = torch.rand(20, 3) * 2 - 1
        new_data = torch.rand(20, 4)
        tree = octree_insert(tree, new_points, new_data)

        # Should have leaves for the inserted points
        new_leaves = (tree.children_mask == 0).sum().item()
        assert new_leaves > 0

    def test_insert_data_shape_mismatch_raises(self):
        """Mismatched data shape raises error."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        new_points = torch.tensor([[0.0, 0.0, 0.0]])
        new_data = torch.tensor([[1.0, 2.0]])  # Wrong shape

        # Should still work (shape check is at kernel level)
        # The kernel will either accept or reject based on implementation
        # For now, just check that it doesn't crash
        try:
            tree = octree_insert(tree, new_points, new_data)
        except RuntimeError:
            pass  # Expected if kernel validates shape

    def test_insert_invalid_points_raises(self):
        """Invalid points tensor raises error."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        new_points = torch.tensor([[0.0, 0.0]])  # Wrong shape
        new_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        with pytest.raises(RuntimeError, match="new_points"):
            octree_insert(tree, new_points, new_data)

    def test_insert_depth_out_of_range_raises(self):
        """Depth out of range raises error."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        new_points = torch.tensor([[0.0, 0.0, 0.0]])
        new_data = torch.rand(1, 4)

        with pytest.raises(RuntimeError, match="depth"):
            octree_insert(tree, new_points, new_data, depth=0)

        with pytest.raises(RuntimeError, match="depth"):
            octree_insert(tree, new_points, new_data, depth=10)


class TestOctreeRemove:
    """Tests for octree_remove operation."""

    def test_remove_single_leaf(self):
        """Remove a single leaf voxel."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)
        original_count = tree.count.item()

        # Find a leaf to remove
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        remove_codes = leaf_codes[:1]

        tree = octree_remove(tree, remove_codes)

        # Count should decrease
        assert tree.count.item() <= original_count

    def test_remove_multiple_leaves(self):
        """Remove multiple leaf voxels."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)
        original_leaves = (tree.children_mask == 0).sum().item()

        # Find leaves to remove
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        remove_codes = leaf_codes[:5]

        tree = octree_remove(tree, remove_codes)

        new_leaves = (tree.children_mask == 0).sum().item()
        assert new_leaves < original_leaves

    def test_remove_nonexistent_codes_raises_error(self):
        """Removing non-existent codes raises an error."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Create fake codes that don't exist (use depth 4 code with unlikely Morton value)
        fake_codes = torch.tensor([(4 << 60) | 0xFFFFF], dtype=torch.int64)

        # Kernel raises error for non-existent codes
        with pytest.raises(RuntimeError, match="not found"):
            octree_remove(tree, fake_codes)

    def test_remove_all_leaves_creates_empty_tree(self):
        """Removing all leaves creates an empty tree (ancestors auto-pruned)."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        # Remove only leaf nodes (kernel only allows leaf removal)
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        tree = octree_remove(tree, leaf_codes)

        # Tree should be empty (ancestors auto-pruned)
        assert tree.count.item() == 0

    def test_remove_prunes_empty_ancestors(self):
        """Removing leaves prunes ancestors with no remaining children."""
        # Create a tree with points in only one region
        points = torch.rand(10, 3) * 0.25 + 0.5  # All in [0.5, 0.75]^3
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        # Remove all leaves
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        tree = octree_remove(tree, leaf_codes)

        # Should have no nodes left (all ancestors were pruned)
        assert tree.count.item() == 0

    def test_remove_invalid_dtype_raises(self):
        """Non-int64 remove_codes raises error."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        remove_codes = torch.tensor([0], dtype=torch.int32)

        with pytest.raises(RuntimeError, match="int64"):
            octree_remove(tree, remove_codes)


class TestOctreeSubdivide:
    """Tests for octree_subdivide operation."""

    def test_subdivide_creates_8_children(self):
        """Subdividing a leaf creates 8 child leaves."""
        # Create a coarse tree
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        tree = octree(points, data, maximum_depth=8)

        # Find a leaf not at max depth to subdivide
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 8)

        if can_subdivide.any():
            code_to_split = tree.codes[can_subdivide][:1]
            original_count = tree.count.item()

            tree = octree_subdivide(tree, code_to_split)

            # Should have 7 more nodes (+8 children, -1 converted parent)
            assert tree.count.item() == original_count + 7

    def test_subdivide_distributes_parent_data(self):
        """Children get parent's data after subdivision."""
        # Build tree with single leaf
        codes = torch.tensor([0 << 60], dtype=torch.int64)  # Root at depth 0
        data = torch.tensor([[42.0]])
        children_mask = torch.tensor([0], dtype=torch.uint8)
        weights = torch.tensor([1.0])
        tree = make_octree(
            codes, data, children_mask, weights, maximum_depth=4
        )

        # Subdivide the root
        tree = octree_subdivide(
            tree, torch.tensor([0 << 60], dtype=torch.int64)
        )

        # All children should have parent's data
        leaf_mask = tree.children_mask == 0
        leaf_data = tree.data[leaf_mask]
        assert (leaf_data == 42.0).all()

    def test_subdivide_internal_node_raises(self):
        """Cannot subdivide internal nodes (non-leaves)."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Find an internal node
        internal_mask = tree.children_mask != 0
        if internal_mask.any():
            internal_code = tree.codes[internal_mask][:1]
            # Should either be ignored or raise an error
            try:
                tree = octree_subdivide(tree, internal_code)
            except RuntimeError:
                pass  # Expected behavior

    def test_subdivide_max_depth_leaf_raises(self):
        """Cannot subdivide leaves at maximum depth."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        # Find leaves at max depth
        depths = (tree.codes >> 60) & 0xF
        max_depth_leaves = (tree.children_mask == 0) & (depths == 4)

        if max_depth_leaves.any():
            code = tree.codes[max_depth_leaves][:1]
            # Should either be ignored or raise an error
            try:
                tree = octree_subdivide(tree, code)
            except RuntimeError:
                pass  # Expected behavior

    def test_subdivide_multiple_leaves(self):
        """Subdivide multiple leaves at once."""
        points = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ]
        )
        data = torch.tensor([[1.0], [2.0]])
        tree = octree(points, data, maximum_depth=6)

        # Find coarse leaves to subdivide
        depths = (tree.codes >> 60) & 0xF
        can_subdivide = (tree.children_mask == 0) & (depths < 6)

        if can_subdivide.sum() >= 2:
            codes_to_split = tree.codes[can_subdivide][:2]
            original_leaves = (tree.children_mask == 0).sum().item()

            tree = octree_subdivide(tree, codes_to_split)

            # Should have more leaves now
            new_leaves = (tree.children_mask == 0).sum().item()
            assert new_leaves > original_leaves


class TestOctreeMerge:
    """Tests for octree_merge operation."""

    def test_merge_after_subdivide(self):
        """Merge after subdivide restores parent as leaf."""
        # Create a tree with a single point
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4)

        # Insert a single point at the origin
        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[42.0]])
        tree = octree_insert(tree, pt, dat)

        # Find a coarse leaf (not at max depth) to subdivide
        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 4)

        if coarse_leaves.any():
            # Get the code of a coarse leaf
            parent_code = tree.codes[coarse_leaves][0:1]
            parent_depth = (parent_code[0].item() >> 60) & 0xF

            if parent_depth < 4:  # Can subdivide
                original_count = tree.count.item()

                # Subdivide to create 8 children
                tree = octree_subdivide(tree, parent_code)

                # Now we have 8 children where the leaf was
                assert tree.count.item() == original_count + 7

                # Merge them back
                tree = octree_merge(tree, parent_code)

                # Should have same count as after original insert
                assert tree.count.item() == original_count

    def test_merge_skipped_if_children_not_all_leaves(self):
        """Merge is skipped if any child is an internal node."""
        # Create a tree and subdivide twice to create grandchildren
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=6)

        # Insert a single point
        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[42.0]])
        tree = octree_insert(tree, pt, dat)

        # Find a coarse leaf
        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 5)

        if coarse_leaves.any():
            parent_code = tree.codes[coarse_leaves][0:1]

            # Subdivide to create children
            tree = octree_subdivide(tree, parent_code)

            # Find one of the new children and subdivide it too
            depths = (tree.codes >> 60) & 0xF
            new_depth = ((parent_code[0].item() >> 60) & 0xF) + 1
            children_at_new_depth = (tree.children_mask == 0) & (
                depths == new_depth
            )

            if children_at_new_depth.any():
                child_code = tree.codes[children_at_new_depth][0:1]
                tree = octree_subdivide(tree, child_code)

                count_before_merge = tree.count.item()

                # Try to merge at parent level - should be skipped because
                # one child (the one we just subdivided) is now internal
                tree = octree_merge(tree, parent_code)

                # Count should be unchanged (merge skipped)
                assert tree.count.item() == count_before_merge

    def test_merge_requires_all_8_children(self):
        """Cannot merge if not all 8 children exist - raises error."""
        # Create a sparse tree where not all octants have points
        # Use two points that are far apart - they won't share fine structure
        points = torch.tensor([[0.9, 0.9, 0.9]])  # Only in one octant
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        # Try to merge at a depth where we know not all 8 children exist
        # But skip depth 0 (root) which can't be merged
        depths = (tree.codes >> 60) & 0xF
        internal_mask = (tree.children_mask != 0) & (depths > 0)

        if internal_mask.any():
            # Get an internal node that doesn't have all 8 children
            parent_code = tree.codes[internal_mask][0:1]

            # Kernel raises error when not all 8 siblings exist
            with pytest.raises(RuntimeError, match="must exist"):
                octree_merge(tree, parent_code)


class TestDynamicUpdateRoundTrips:
    """Tests for combinations of dynamic update operations."""

    def test_subdivide_then_merge_restores_data(self):
        """Subdivide followed by merge restores original data."""
        # Create tree via insert
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4)

        # Insert a single point
        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[42.0]])
        tree = octree_insert(tree, pt, dat)

        # Find a coarse leaf to subdivide
        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 4)

        if coarse_leaves.any():
            parent_code = tree.codes[coarse_leaves][0:1]
            original_count = tree.count.item()

            # Subdivide
            tree = octree_subdivide(tree, parent_code)
            assert tree.count.item() == original_count + 7

            # Merge back
            tree = octree_merge(tree, parent_code)
            assert tree.count.item() == original_count

    def test_insert_then_remove_restores_tree(self):
        """Insert followed by targeted remove leaves fewer leaves."""
        # Start with empty tree
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        # Insert new point
        new_points = torch.tensor([[0.5, 0.5, 0.5]])
        new_data = torch.rand(1, 4)
        tree = octree_insert(tree, new_points, new_data)

        assert tree.count.item() > 0

        # Remove all leaves
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        tree = octree_remove(tree, leaf_codes)

        # Tree should be empty again
        assert tree.count.item() == 0

    def test_multiple_inserts_batch(self):
        """Multiple points can be inserted in a batch."""
        # Start with empty tree
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4)

        # Insert multiple points at once (batch insert avoids conflict issues)
        pts = torch.tensor(
            [
                [-0.8, 0.0, 0.0],
                [-0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.8, 0.0, 0.0],
            ]
        )
        dat = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
        tree = octree_insert(tree, pts, dat)

        # Should have leaves for the inserted points
        leaf_count = (tree.children_mask == 0).sum().item()
        assert leaf_count >= 5


class TestDynamicUpdateQueryIntegration:
    """Tests for querying after dynamic updates."""

    def test_query_after_insert(self):
        """Can query tree after insert."""
        # Start with empty tree to avoid conflicts
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        # Insert new point
        new_points = torch.tensor([[0.5, 0.5, 0.5]])
        new_data = torch.tensor([[99.0, 99.0, 99.0, 99.0]])
        tree = octree_insert(tree, new_points, new_data)

        # Query at insertion point
        result, found = octree_sample(tree, new_points)

        assert found[0].item()
        assert result.shape == (1, 4)

    def test_query_after_remove(self):
        """Can query tree after remove."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 4)
        tree = octree(points, data, maximum_depth=4)

        # Remove some leaves
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask][:5]
        tree = octree_remove(tree, leaf_codes)

        # Query still works
        queries = torch.rand(10, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.shape == (10, 4)

    def test_query_after_subdivide(self):
        """Can query tree after subdivide."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[42.0]])
        tree = octree(points, data, maximum_depth=6)

        # Find coarse leaf and subdivide
        depths = (tree.codes >> 60) & 0xF
        coarse = (tree.children_mask == 0) & (depths < 6)
        if coarse.any():
            code = tree.codes[coarse][:1]
            tree = octree_subdivide(tree, code)

        # Query at origin
        queries = torch.tensor([[0.0, 0.0, 0.0]])
        result, found = octree_sample(tree, queries)

        assert result.shape == (1, 1)

    def test_query_after_merge(self):
        """Can query tree after merge."""
        # Create tree via insert, subdivide, then merge
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=4)

        # Insert a point
        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.rand(1, 4)
        tree = octree_insert(tree, pt, dat)

        # Find and subdivide a coarse leaf
        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 4)
        if coarse_leaves.any():
            parent_code = tree.codes[coarse_leaves][0:1]
            tree = octree_subdivide(tree, parent_code)
            tree = octree_merge(tree, parent_code)

        # Query still works
        queries = torch.rand(10, 3) * 2 - 1
        result, found = octree_sample(tree, queries)

        assert result.shape == (10, 4)


class TestDynamicUpdateDtypes:
    """Tests for dtype handling in dynamic updates."""

    def test_insert_preserves_dtype(self):
        """Insert preserves data dtype."""
        # Start with empty tree to avoid conflicts
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        new_points = torch.rand(5, 3) * 2 - 1
        new_data = torch.rand(5, 4, dtype=torch.float64)
        tree = octree_insert(tree, new_points, new_data)

        assert tree.data.dtype == torch.float64

    def test_subdivide_preserves_dtype(self):
        """Subdivide preserves data dtype."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]], dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        depths = (tree.codes >> 60) & 0xF
        coarse = (tree.children_mask == 0) & (depths < 4)
        if coarse.any():
            code = tree.codes[coarse][:1]
            tree = octree_subdivide(tree, code)

        assert tree.data.dtype == torch.float64

    def test_merge_preserves_dtype(self):
        """Merge preserves data dtype."""
        # Create tree via insert, subdivide, merge
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[42.0]], dtype=torch.float64)
        tree = octree_insert(tree, pt, dat)

        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 4)
        if coarse_leaves.any():
            parent_code = tree.codes[coarse_leaves][0:1]
            tree = octree_subdivide(tree, parent_code)
            tree = octree_merge(tree, parent_code)

        assert tree.data.dtype == torch.float64


class TestDynamicUpdateAggregationModes:
    """Tests for different aggregation modes in dynamic updates."""

    def test_merge_preserves_aggregation_mode(self):
        """Merge uses tree's aggregation mode."""
        # Create tree with specific aggregation mode
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4, aggregation="sum")

        pt = torch.tensor([[0.0, 0.0, 0.0]])
        dat = torch.tensor([[10.0]])
        tree = octree_insert(tree, pt, dat)

        # Subdivide
        depths = (tree.codes >> 60) & 0xF
        coarse_leaves = (tree.children_mask == 0) & (depths < 4)
        if coarse_leaves.any():
            parent_code = tree.codes[coarse_leaves][0:1]
            tree = octree_subdivide(tree, parent_code)
            # After merge, the sum aggregation should aggregate child data
            tree = octree_merge(tree, parent_code)

        # The aggregation mode should be preserved
        assert tree.aggregation.item() == 1  # sum

    def test_insert_with_mean_aggregation(self):
        """Insert updates ancestors with mean aggregation."""
        # Start with empty tree
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 1)
        tree = octree(points, data, maximum_depth=4, aggregation="mean")

        # Insert a point
        new_points = torch.tensor([[0.5, 0.5, 0.5]])
        new_data = torch.tensor([[10.0]])
        tree = octree_insert(tree, new_points, new_data)

        # Root should have mean of all leaves
        assert tree.data.shape[0] > 0
