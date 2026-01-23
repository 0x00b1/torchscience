"""Tests for breadth_first_search."""

import pytest
import torch

from torchscience.graph import breadth_first_search


class TestBFSBasic:
    """Basic functionality tests."""

    def test_simple_chain(self):
        """Chain graph: 0 -> 1 -> 2."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = breadth_first_search(adj, source=0)

        assert dist.tolist() == [0, 1, 2]  # Hop counts
        assert pred[0] == -1
        assert pred[1] == 0
        assert pred[2] == 1

    def test_branching(self):
        """Star graph: 0 -> 1, 0 -> 2, 0 -> 3."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [inf, 0.0, inf, inf],
                [inf, inf, 0.0, inf],
                [inf, inf, inf, 0.0],
            ]
        )
        dist, pred = breadth_first_search(adj, source=0)

        assert dist[0] == 0
        assert dist[1] == dist[2] == dist[3] == 1

    def test_unreachable(self):
        """Unreachable nodes have distance -1."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = breadth_first_search(adj, source=0)

        assert dist[2] == -1
        assert pred[2] == -1

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        dist, pred = breadth_first_search(adj, source=0)

        assert dist[0] == 0
        assert pred[0] == -1


class TestBFSUndirected:
    """Tests for undirected graphs."""

    def test_undirected(self):
        """Undirected graph traversal."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ]
        )
        # From node 2, can reach 1 and 0 in undirected mode
        dist, pred = breadth_first_search(adj, source=2, directed=False)

        assert dist[2] == 0
        assert dist[1] == 1
        assert dist[0] == 2


class TestBFSValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            breadth_first_search(torch.tensor([1.0, 2.0]), source=0)

    def test_rejects_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            breadth_first_search(torch.rand(3, 3), source=5)


class TestBFSBatched:
    """Tests for batched input."""

    def test_batched(self):
        """Batch of graphs."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [[0.0, 1.0], [inf, 0.0]],
                [[0.0, 1.0], [inf, 0.0]],
            ]
        )
        dist, pred = breadth_first_search(adj, source=0)
        assert dist.shape == (2, 2)
        assert pred.shape == (2, 2)


class TestBFSMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        adj = torch.rand(5, 5, device="meta")
        dist, pred = breadth_first_search(adj, source=0)
        assert dist.shape == (5,)
        assert pred.shape == (5,)


class TestBFSReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10])
    def test_matches_scipy(self, N):
        """Compare to scipy.sparse.csgraph.breadth_first_order."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        adj_np = numpy.random.rand(N, N)
        adj_np[adj_np > 0.3] = numpy.inf
        numpy.fill_diagonal(adj_np, 0)

        adj = torch.from_numpy(adj_np).float()
        dist_ours, _ = breadth_first_search(adj, source=0)

        # Scipy BFS
        csr = scipy_sparse.csr_matrix(numpy.where(numpy.isinf(adj_np), 0, 1))
        order_scipy, pred_scipy = scipy_sparse.csgraph.breadth_first_order(
            csr, i_start=0, directed=True
        )

        # Check reachable nodes match
        reachable_ours = set((dist_ours >= 0).nonzero().squeeze(-1).tolist())
        reachable_scipy = set(order_scipy.tolist())
        assert reachable_ours == reachable_scipy
