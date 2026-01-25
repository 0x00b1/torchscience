import pytest
import torch

from torchscience.geometry.space_partitioning import kd_tree, range_search


class TestRangeSearchBasic:
    """Tests for range_search query function."""

    def test_returns_nested_tensors(self):
        """range_search returns nested tensors."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=1.0)

        assert indices.is_nested
        assert distances.is_nested

    def test_nested_tensor_length_matches_queries(self):
        """Nested tensors have one entry per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=1.0)

        assert len(indices.unbind()) == 10
        assert len(distances.unbind()) == 10

    def test_distances_are_within_radius(self):
        """All returned distances are within radius."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)
        radius = 1.0

        indices, distances = range_search(tree, queries, radius=radius)

        for dist_tensor in distances.unbind():
            if dist_tensor.numel() > 0:
                assert (dist_tensor <= radius + 1e-6).all()

    def test_distances_are_sorted(self):
        """Distances are sorted in ascending order per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=2.0)

        for dist_tensor in distances.unbind():
            if dist_tensor.numel() > 1:
                sorted_dists = torch.sort(dist_tensor)[0]
                torch.testing.assert_close(dist_tensor, sorted_dists)

    def test_indices_are_valid(self):
        """Indices are in valid range [0, n)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = range_search(tree, queries, radius=2.0)

        for idx_tensor in indices.unbind():
            if idx_tensor.numel() > 0:
                assert (idx_tensor >= 0).all()
                assert (idx_tensor < 100).all()


class TestRangeSearchCorrectness:
    """Tests for correctness against brute force."""

    def test_matches_brute_force(self):
        """Results match brute-force search."""
        torch.manual_seed(42)
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64)
        radius = 1.5

        indices, distances = range_search(tree, queries, radius=radius)

        # Brute force
        for i, query in enumerate(queries):
            dists = torch.sqrt(((points - query) ** 2).sum(dim=1))
            bf_mask = dists <= radius
            bf_dists = dists[bf_mask]
            bf_indices = torch.where(bf_mask)[0]

            # Sort for comparison
            bf_sorted_idx = torch.argsort(bf_dists)
            bf_dists_sorted = bf_dists[bf_sorted_idx]
            bf_indices_sorted = bf_indices[bf_sorted_idx]

            result_indices = indices.unbind()[i]
            result_distances = distances.unbind()[i]

            assert result_indices.numel() == bf_indices_sorted.numel()
            torch.testing.assert_close(
                result_distances, bf_dists_sorted, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(result_indices, bf_indices_sorted)

    def test_query_point_in_dataset(self):
        """Query point that exists in dataset is found with distance 0."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        query = points[42:43]

        indices, distances = range_search(tree, query, radius=0.1)

        idx_list = indices.unbind()[0]
        dist_list = distances.unbind()[0]

        assert 42 in idx_list.tolist()
        min_dist = dist_list.min()
        torch.testing.assert_close(
            min_dist, torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_empty_result(self):
        """Handles queries with no neighbors within radius."""
        points = torch.zeros(10, 3)  # All at origin
        tree = kd_tree(points)
        query = torch.tensor([[100.0, 100.0, 100.0]])  # Far away

        indices, distances = range_search(tree, query, radius=1.0)

        assert indices.unbind()[0].numel() == 0
        assert distances.unbind()[0].numel() == 0


class TestRangeSearchGradient:
    """Tests for gradient support."""

    @pytest.mark.xfail(
        reason="PyTorch nested tensors don't support autograd: "
        "NestedTensorImpl doesn't support sizes()"
    )
    def test_gradient_exists(self):
        """Gradient exists for query points."""
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        indices, distances = range_search(tree, queries, radius=2.0)

        # Sum all distances across nested tensor
        total = sum(d.sum() for d in distances.unbind())
        total.backward()

        assert queries.grad is not None
        assert torch.isfinite(queries.grad).all()

    @pytest.mark.xfail(
        reason="PyTorch nested tensors don't support autograd: "
        "NestedTensorImpl doesn't support sizes()"
    )
    def test_zero_distance_gradient_is_finite(self):
        """Zero distance has finite gradient."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = points[5:6].clone().requires_grad_(True)

        _, distances = range_search(tree, queries, radius=1.0)
        total = sum(d.sum() for d in distances.unbind())
        total.backward()

        assert torch.isfinite(queries.grad).all()


class TestRangeSearchValidation:
    """Tests for input validation."""

    def test_wrong_tree_type_raises(self):
        """Raises RuntimeError for wrong tree type."""
        from tensordict import TensorDict

        fake_tree = TensorDict({"_type": "wrong_type"}, batch_size=[])
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="Unsupported tree type"):
            range_search(fake_tree, queries, radius=1.0)

    def test_dimension_mismatch_raises(self):
        """Raises RuntimeError when query dimension doesn't match tree."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 5)

        with pytest.raises(RuntimeError, match="dimension"):
            range_search(tree, queries, radius=1.0)

    def test_negative_radius_raises(self):
        """Raises RuntimeError for negative radius."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        with pytest.raises(RuntimeError, match="radius"):
            range_search(tree, queries, radius=-1.0)
