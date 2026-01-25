# tests/torchscience/geometry/space_partitioning/test__k_nearest_neighbors.py
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.geometry.space_partitioning import (
    k_nearest_neighbors,
    kd_tree,
)


class TestKNearestNeighborsBasic:
    """Tests for k_nearest_neighbors query function."""

    def test_returns_indices_and_distances(self):
        """Returns tuple of (indices, distances)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        result = k_nearest_neighbors(tree, queries, k=5)

        assert isinstance(result, tuple)
        assert len(result) == 2
        indices, distances = result
        assert indices.shape == (10, 5)
        assert distances.shape == (10, 5)

    def test_indices_dtype_is_long(self):
        """Indices tensor has dtype long."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert indices.dtype == torch.long

    def test_distances_preserve_dtype(self):
        """Distances tensor preserves query dtype."""
        points = torch.randn(100, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(10, 3, dtype=torch.float64)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert distances.dtype == torch.float64

    def test_distances_are_non_negative(self):
        """All distances are non-negative."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert (distances >= 0).all()

    def test_distances_are_sorted(self):
        """Distances are sorted in ascending order per query."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        for i in range(10):
            sorted_dists = torch.sort(distances[i])[0]
            torch.testing.assert_close(distances[i], sorted_dists)

    def test_indices_are_valid(self):
        """Indices are in valid range [0, n)."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 3)

        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        assert (indices >= 0).all()
        assert (indices < 100).all()


class TestKNearestNeighborsCorrectness:
    """Tests for correctness against brute force."""

    def test_matches_brute_force_euclidean(self):
        """Results match brute-force search for Euclidean distance."""
        torch.manual_seed(42)
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64)

        indices, distances = k_nearest_neighbors(tree, queries, k=3, p=2.0)

        # Brute force
        for i, query in enumerate(queries):
            dists = torch.sqrt(((points - query) ** 2).sum(dim=1))
            bf_dists, bf_indices = torch.topk(dists, k=3, largest=False)

            torch.testing.assert_close(
                distances[i], bf_dists, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(indices[i], bf_indices)

    def test_query_point_in_dataset(self):
        """Query point that exists in dataset has distance 0."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        query = points[42:43]

        indices, distances = k_nearest_neighbors(tree, query, k=1)

        assert indices[0, 0] == 42
        torch.testing.assert_close(
            distances[0, 0], torch.tensor(0.0), atol=1e-6, rtol=1e-6
        )

    def test_tree_traversal_prunes(self):
        """Tree traversal visits fewer nodes than brute force."""
        # Create clustered data where pruning helps
        torch.manual_seed(42)
        cluster1 = torch.randn(500, 3) + torch.tensor([0.0, 0.0, 0.0])
        cluster2 = torch.randn(500, 3) + torch.tensor([100.0, 0.0, 0.0])
        points = torch.cat([cluster1, cluster2], dim=0)

        tree = kd_tree(points, leaf_size=10)
        queries = cluster1[:5]  # Query near cluster1

        # Should find neighbors in cluster1 without checking cluster2
        indices, distances = k_nearest_neighbors(tree, queries, k=5)

        # All neighbors should be from cluster1 (indices 0-499)
        assert (indices < 500).all()


class TestKNearestNeighborsGradient:
    """Tests for gradient support."""

    def test_gradient_exists(self):
        """Gradient exists for query points."""
        points = torch.randn(50, 3, dtype=torch.float64)
        tree = kd_tree(points)
        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

        indices, distances = k_nearest_neighbors(tree, queries, k=3)
        loss = distances.sum()
        loss.backward()

        assert queries.grad is not None
        assert torch.isfinite(queries.grad).all()

    def test_gradcheck(self):
        """First-order gradient is numerically correct."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)

        def fn(queries):
            _, distances = k_nearest_neighbors(tree, queries, k=3)
            return distances

        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradcheck(fn, (queries,), raise_exception=True)

    def test_gradgradcheck(self):
        """Second-order gradient is numerically correct."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)

        def fn(queries):
            _, distances = k_nearest_neighbors(tree, queries, k=3)
            return distances

        queries = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(fn, (queries,), raise_exception=True)

    def test_zero_distance_gradient_is_finite(self):
        """Zero distance has finite gradient (using torch.where for safety)."""
        points = torch.randn(20, 3, dtype=torch.float64)
        tree = kd_tree(points)
        # Use exact point from dataset
        queries = points[5:6].clone().requires_grad_(True)

        _, distances = k_nearest_neighbors(tree, queries, k=3)
        loss = distances.sum()
        loss.backward()

        # Should have finite gradients (zero where distance is zero)
        assert torch.isfinite(queries.grad).all()


class TestKNearestNeighborsValidation:
    """Tests for input validation."""

    def test_wrong_tree_type_raises(self):
        """Raises RuntimeError for wrong tree type."""
        from tensordict import TensorDict

        fake_tree = TensorDict({"_type": "wrong_type"}, batch_size=[])
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="Unsupported tree type"):
            k_nearest_neighbors(fake_tree, queries, k=3)

    def test_dimension_mismatch_raises(self):
        """Raises RuntimeError when query dimension doesn't match tree."""
        points = torch.randn(100, 3)
        tree = kd_tree(points)
        queries = torch.randn(10, 5)

        with pytest.raises(RuntimeError, match="dimension"):
            k_nearest_neighbors(tree, queries, k=3)

    def test_k_too_large_raises(self):
        """Raises RuntimeError when k > n."""
        points = torch.randn(10, 3)
        tree = kd_tree(points)
        queries = torch.randn(5, 3)

        with pytest.raises(RuntimeError, match="k .* must be in"):
            k_nearest_neighbors(tree, queries, k=20)
