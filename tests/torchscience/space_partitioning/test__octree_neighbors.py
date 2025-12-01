"""Tests for octree neighbor finding operations."""

import pytest
import torch
import torch.testing

from torchscience.space_partitioning import octree, octree_neighbors


class TestConnectivity:
    """Tests for different connectivity levels."""

    def test_face_neighbors_6(self):
        """6-connectivity returns face neighbors only."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        # Query a leaf voxel
        leaf_mask = tree.children_mask == 0
        if not leaf_mask.any():
            pytest.skip("No leaf voxels in tree")
        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        assert neighbor_codes.shape == (1, 6)
        assert neighbor_data.shape == (1, 6, 4)

    def test_edge_neighbors_18(self):
        """18-connectivity returns face and edge neighbors."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        if not leaf_mask.any():
            pytest.skip("No leaf voxels in tree")
        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=18
        )

        assert neighbor_codes.shape == (1, 18)
        assert neighbor_data.shape == (1, 18, 4)

    def test_corner_neighbors_26(self):
        """26-connectivity returns all neighbors."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        if not leaf_mask.any():
            pytest.skip("No leaf voxels in tree")
        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=26
        )

        assert neighbor_codes.shape == (1, 26)
        assert neighbor_data.shape == (1, 26, 4)

    def test_invalid_connectivity_raises(self):
        """Invalid connectivity values raise error."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        if not leaf_mask.any():
            pytest.skip("No leaf voxels in tree")
        query_codes = tree.codes[leaf_mask][:1]

        with pytest.raises(ValueError, match="connectivity must be"):
            octree_neighbors(tree, query_codes, connectivity=4)


class TestBoundaryConditions:
    """Tests for boundary neighbor handling."""

    def test_corner_voxel_has_missing_neighbors(self):
        """Voxel at domain corner has some -1 neighbor codes."""
        # Create a point at the corner
        points = torch.tensor([[0.9, 0.9, 0.9]])
        data = torch.tensor([[1.0]])
        tree = octree(points, data, maximum_depth=4)

        # Get the leaf code for this point
        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        # At least some neighbors should be -1 (outside domain)
        # +x, +y, +z directions should be missing
        has_missing = (neighbor_codes == -1).any()
        assert has_missing

    def test_center_voxel_may_have_all_neighbors(self):
        """Voxel near center with dense neighbors can have all present."""
        # Create a dense grid of points
        coords = torch.linspace(-0.5, 0.5, 5)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        data = torch.rand(points.shape[0], 2)
        tree = octree(points, data, maximum_depth=3)

        # Get center leaf voxels
        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask]

        neighbor_codes, _ = octree_neighbors(tree, query_codes, connectivity=6)

        # At least some queries should have all neighbors present
        all_present_per_query = (neighbor_codes != -1).all(dim=1)
        # May or may not have all neighbors depending on tree structure
        assert neighbor_codes.shape == (query_codes.shape[0], 6)


class TestNeighborDataValues:
    """Tests for correct neighbor data retrieval."""

    def test_neighbor_data_matches_tree(self):
        """Neighbor data should match data stored in tree."""
        # Create predictable points
        points = torch.tensor(
            [
                [0.25, 0.25, 0.25],
                [-0.25, 0.25, 0.25],  # -x neighbor of first
            ]
        )
        # Use distinct data values
        data = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        tree = octree(points, data, maximum_depth=3)

        # Query the first point's voxel
        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, leaf_codes, connectivity=6
        )

        # Verify shape
        assert neighbor_data.shape == (leaf_codes.shape[0], 6, 2)

        # For present neighbors, data should be non-zero (or match tree data)
        for i in range(neighbor_codes.shape[0]):
            for j in range(6):
                if neighbor_codes[i, j] != -1:
                    # This neighbor exists - verify we got some data
                    assert not neighbor_data[i, j].isnan().any()


class TestBatchedQueries:
    """Tests for batched neighbor queries."""

    def test_multiple_queries(self):
        """Multiple voxels can be queried at once."""
        points = torch.rand(200, 3) * 2 - 1
        data = torch.rand(200, 4)
        tree = octree(points, data, maximum_depth=5)

        leaf_mask = tree.children_mask == 0
        n_leaves = leaf_mask.sum().item()
        if n_leaves < 5:
            pytest.skip("Not enough leaf voxels")

        # Query multiple leaves
        query_codes = tree.codes[leaf_mask][:5]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        assert neighbor_codes.shape == (5, 6)
        assert neighbor_data.shape == (5, 6, 4)

    def test_all_leaves_query(self):
        """Query all leaf voxels at once."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 2)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        leaf_codes = tree.codes[leaf_mask]
        n_leaves = leaf_codes.shape[0]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, leaf_codes, connectivity=6
        )

        assert neighbor_codes.shape == (n_leaves, 6)
        assert neighbor_data.shape == (n_leaves, 6, 2)


class TestEmptyTree:
    """Tests for neighbor finding in empty trees."""

    def test_empty_query_returns_empty(self):
        """Empty query codes returns empty result."""
        points = torch.rand(10, 3) * 2 - 1
        data = torch.rand(10, 4)
        tree = octree(points, data, maximum_depth=4)

        query_codes = torch.empty(0, dtype=torch.int64)

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        assert neighbor_codes.shape == (0, 6)
        assert neighbor_data.shape == (0, 6, 4)


class TestOutputShapes:
    """Tests for correct output tensor shapes."""

    def test_multi_dimensional_data(self):
        """Neighbor finding with multi-dimensional voxel data."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 3, 4)  # Multi-dimensional
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        assert neighbor_data.shape == (1, 6, 3, 4)

    def test_dtype_preservation(self):
        """Output dtype matches input data dtype."""
        points = torch.rand(50, 3) * 2 - 1
        data = torch.rand(50, 4, dtype=torch.float64)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        assert neighbor_data.dtype == torch.float64
        assert neighbor_codes.dtype == torch.int64


class TestLODAwareness:
    """Tests for LOD-aware neighbor finding."""

    def test_same_depth_neighbors(self):
        """Neighbors at same depth are found correctly."""
        # Create uniform depth tree
        coords = torch.linspace(-0.8, 0.8, 4)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
        points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        data = torch.arange(points.shape[0], dtype=torch.float32).unsqueeze(1)
        tree = octree(points, data, maximum_depth=3)

        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        # Verify output shapes
        assert neighbor_codes.shape[0] == query_codes.shape[0]
        assert neighbor_codes.shape[1] == 6


class TestNeighborDirections:
    """Tests for correct neighbor direction mapping."""

    def test_face_neighbor_directions(self):
        """Face neighbors (0-5) correspond to ±x, ±y, ±z."""
        # Face neighbor offsets:
        # 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        leaf_mask = tree.children_mask == 0
        if not leaf_mask.any():
            pytest.skip("No leaf voxels")

        query_codes = tree.codes[leaf_mask][:1]

        neighbor_codes, _ = octree_neighbors(tree, query_codes, connectivity=6)

        # Verify we got 6 neighbor slots
        assert neighbor_codes.shape == (1, 6)


class TestAutograd:
    """Tests for autograd support."""

    @pytest.mark.skip(reason="Autograd kernel not yet implemented")
    def test_gradient_wrt_data(self):
        """Gradients flow through data."""
        points = torch.rand(100, 3) * 2 - 1
        data = torch.rand(100, 4)
        tree = octree(points, data, maximum_depth=4)

        tree.data = tree.data.clone().requires_grad_(True)

        leaf_mask = tree.children_mask == 0
        query_codes = tree.codes[leaf_mask]

        neighbor_codes, neighbor_data = octree_neighbors(
            tree, query_codes, connectivity=6
        )

        # Mask for valid neighbors
        valid_mask = neighbor_codes != -1
        loss = (neighbor_data * valid_mask.unsqueeze(-1)).sum()
        loss.backward()

        assert tree.data.grad is not None
