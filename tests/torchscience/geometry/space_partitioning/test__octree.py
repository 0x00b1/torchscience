"""Tests for octree construction."""

import pytest
import torch
import torch.testing

from torchscience.geometry.space_partitioning import octree

from .conftest import _decode_depth, _hash_lookup


class TestOctreeConstruction:
    """Tests for octree construction from points."""

    def test_basic_construction(self):
        """Build octree from random points."""
        points = torch.rand(1000, 3) * 2 - 1
        data = torch.rand(1000, 8)
        tree = octree(points, data, maximum_depth=8)

        assert tree.count.item() > 0
        assert tree.data.shape[1] == 8
        # Should have more nodes than unique leaves (includes internal)
        n_leaves = (tree.children_mask == 0).sum().item()
        assert tree.count.item() >= n_leaves

    def test_full_hierarchy_created(self):
        """Construction creates internal nodes at all depths."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        # Should have nodes at depths 0, 1, 2, 3, 4 (5 total for single point)
        # CRITICAL: Use masked extraction for correct depth >= 8 handling
        depths = (tree.codes >> 60) & 0xF
        unique_depths = depths.unique()
        assert len(unique_depths) == 5
        assert 0 in unique_depths  # Root
        assert 4 in unique_depths  # Leaf

    def test_internal_nodes_have_aggregated_data_mean(self):
        """Internal nodes store mean of descendant data."""
        # Two points that will be in different leaves but same parent
        # Use well-separated points to ensure different leaves
        points = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1]])
        data = torch.tensor([[2.0], [4.0]])
        tree = octree(points, data, maximum_depth=8, aggregation="mean")

        # Find root node (depth 0)
        # CRITICAL: Use masked extraction for correct depth handling
        depths = (tree.codes >> 60) & 0xF
        root_mask = depths == 0
        assert root_mask.any(), "Root node should exist"
        root_idx = root_mask.nonzero().item()
        # Root should have mean of all data
        assert tree.data[root_idx, 0].item() == pytest.approx(3.0, rel=1e-5)

    def test_weights_tracking(self):
        """Weights track point contributions."""
        # Two points very close together (likely same leaf at depth 4)
        points = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
        data = torch.tensor([[1.0], [3.0]])
        tree = octree(points, data, maximum_depth=4)

        # Root weight should be 2.0 (two points total)
        depths = (tree.codes >> 60) & 0xF
        root_idx = (depths == 0).nonzero().item()
        assert tree.weights[root_idx].item() == 2.0

    def test_out_of_bounds_silently_clamped(self):
        """Points outside [-1, 1]³ are silently clamped."""
        points = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        data = torch.ones(2, 1)
        # Should not raise, should not warn
        tree = octree(points, data, maximum_depth=8)
        assert tree.count.item() > 0

    def test_aggregation_modes(self):
        """Different aggregation modes produce different internal data."""
        points = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1]])
        data = torch.tensor([[2.0], [4.0]])

        tree_mean = octree(points, data, maximum_depth=8, aggregation="mean")
        tree_max = octree(points, data, maximum_depth=8, aggregation="max")
        tree_sum = octree(points, data, maximum_depth=8, aggregation="sum")

        # CRITICAL: Use masked extraction for correct depth handling
        def get_root_data(tree):
            depths = (tree.codes >> 60) & 0xF
            root_idx = (depths == 0).nonzero().item()
            return tree.data[root_idx, 0].item()

        assert get_root_data(tree_mean) == pytest.approx(3.0, rel=1e-5)  # mean
        assert get_root_data(tree_max) == pytest.approx(4.0, rel=1e-5)  # max
        assert get_root_data(tree_sum) == pytest.approx(6.0, rel=1e-5)  # sum

    def test_single_point(self):
        """Construction with a single point."""
        points = torch.tensor([[0.5, 0.5, 0.5]])
        data = torch.tensor([[42.0]])
        tree = octree(points, data, maximum_depth=4)

        # Should have 5 nodes (depth 0, 1, 2, 3, 4)
        assert tree.count.item() == 5

        # Leaf should have the original data
        leaf_mask = tree.children_mask == 0
        leaf_idx = leaf_mask.nonzero().item()
        assert tree.data[leaf_idx, 0].item() == 42.0

    def test_empty_input(self):
        """Construction with empty input."""
        points = torch.zeros(0, 3)
        data = torch.zeros(0, 4)
        tree = octree(points, data, maximum_depth=8)

        assert tree.count.item() == 0
        assert tree.codes.shape[0] == 0
        assert tree.data.shape[0] == 0

    def test_duplicate_points(self):
        """Multiple points at exact same location are aggregated."""
        # All points at same location
        points = torch.tensor(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        data = torch.tensor([[1.0], [2.0], [3.0]])
        tree = octree(points, data, maximum_depth=8, aggregation="mean")

        # Should have only one leaf (but all ancestors)
        n_leaves = (tree.children_mask == 0).sum().item()
        assert n_leaves == 1

        # Leaf data should be mean
        leaf_idx = (tree.children_mask == 0).nonzero().item()
        assert tree.data[leaf_idx, 0].item() == pytest.approx(2.0, rel=1e-5)
        assert tree.weights[leaf_idx].item() == 3.0

    def test_multi_dimensional_data(self):
        """Construction with multi-dimensional data."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 3, 4)  # Shape (100, 3, 4)
        tree = octree(points, data, maximum_depth=6)

        # Data should preserve shape
        assert tree.data.shape[1:] == (3, 4)

    def test_maximum_depth_limits(self):
        """Test maximum_depth boundary conditions."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])

        # Minimum depth
        tree = octree(points, data, maximum_depth=1)
        assert tree.count.item() == 2  # root + 1 leaf

        # Maximum depth (15)
        tree = octree(points, data, maximum_depth=15)
        assert tree.count.item() == 16  # 16 levels

    def test_invalid_maximum_depth(self):
        """Invalid maximum_depth raises error."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])

        with pytest.raises(RuntimeError, match="maximum_depth"):
            octree(points, data, maximum_depth=0)

        with pytest.raises(RuntimeError, match="maximum_depth"):
            octree(points, data, maximum_depth=16)

    def test_invalid_aggregation(self):
        """Invalid aggregation raises error."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])

        with pytest.raises(RuntimeError, match="aggregation"):
            octree(points, data, aggregation="invalid")


class TestHashTableCorrectness:
    """Tests for bounded probing correctness guarantees."""

    def test_max_displacement_guarantee(self):
        """Construction enforces max_probes bound on displacement."""
        # Create a tree with enough entries to potentially exceed probe bound
        points = torch.rand(10000, 3) * 2 - 1
        data = torch.rand(10000, 1)
        tree = octree(points, data, maximum_depth=10)

        # Verify all entries are findable within max_probes
        # (This tests the rebuild guarantee implicitly)
        for i in range(min(100, tree.count.item())):
            code = tree.codes[i].item()
            idx = _hash_lookup(tree.structure, tree.codes, code, max_probes=64)
            assert idx == i, f"Entry {i} not found within max_probes=64"

    def test_no_false_negatives_under_bound(self):
        """Bounded probing never returns false negative for existing keys."""
        points = torch.rand(1000, 3) * 2 - 1
        data = torch.rand(1000, 4)
        tree = octree(points, data, maximum_depth=8)

        # All codes should be findable
        for i in range(tree.count.item()):
            code = tree.codes[i].item()
            idx = _hash_lookup(tree.structure, tree.codes, code)
            assert idx == i, f"Code at index {i} not found in hash table"


class TestOctreeAttributes:
    """Tests for octree tensorclass attributes."""

    def test_dtype_preservation(self):
        """Data dtype is preserved."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=6)
        assert tree.data.dtype == torch.float64

    def test_children_mask_values(self):
        """children_mask correctly identifies leaves and internals."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=6)

        # Leaves have children_mask == 0
        leaf_mask = tree.children_mask == 0
        n_leaves = leaf_mask.sum().item()

        # Non-leaves have children_mask != 0
        internal_mask = tree.children_mask != 0
        n_internal = internal_mask.sum().item()

        assert n_leaves + n_internal == tree.count.item()
        assert n_leaves > 0  # Should have at least one leaf
        assert n_internal > 0  # Should have at least root

    def test_codes_depth_bits(self):
        """Codes have correct depth in bits 60-63."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=8)

        for i in range(tree.count.item()):
            code = tree.codes[i].item()
            depth = _decode_depth(code)
            assert 0 <= depth <= 8, f"Invalid depth {depth} at index {i}"

    def test_device_movement(self):
        """Octree can be moved between devices."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=6)

        # Should be on CPU by default
        assert tree.codes.device.type == "cpu"

        # Can move to same device (CUDA test skipped if not available)
        tree_cpu = tree.to("cpu")
        assert tree_cpu.codes.device.type == "cpu"
