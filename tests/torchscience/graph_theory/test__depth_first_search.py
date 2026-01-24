"""Tests for depth_first_search."""

import pytest
import torch

from torchscience.graph import depth_first_search


class TestDFSBasic:
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
        disc, finish, pred = depth_first_search(adj, source=0)

        # Discovery times should increase along path
        assert disc[0] < disc[1] < disc[2]
        # Finish times should decrease (last discovered finishes first)
        assert finish[2] < finish[1] < finish[0]
        # Predecessors
        assert pred[0] == -1
        assert pred[1] == 0
        assert pred[2] == 1

    def test_branching(self):
        """Star graph: 0 -> 1, 0 -> 2."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        disc, finish, pred = depth_first_search(adj, source=0)

        assert disc[0] == 0
        # Both children discovered after parent
        assert disc[1] > disc[0]
        assert disc[2] > disc[0]

    def test_unreachable(self):
        """Unreachable nodes have -1 for all outputs."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        disc, finish, pred = depth_first_search(adj, source=0)

        assert disc[2] == -1
        assert finish[2] == -1
        assert pred[2] == -1

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        disc, finish, pred = depth_first_search(adj, source=0)

        assert disc[0] == 0
        assert finish[0] == 1
        assert pred[0] == -1

    def test_cycle(self):
        """Graph with cycle: 0 -> 1 -> 2 -> 0."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [1.0, inf, 0.0],
            ]
        )
        disc, finish, pred = depth_first_search(adj, source=0)

        # All nodes should be reachable
        assert all(d >= 0 for d in disc.tolist())
        assert all(f >= 0 for f in finish.tolist())


class TestDFSUndirected:
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
        disc, finish, pred = depth_first_search(adj, source=2, directed=False)

        # All nodes reachable in undirected mode
        assert disc[0] >= 0
        assert disc[1] >= 0
        assert disc[2] >= 0


class TestDFSValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            depth_first_search(torch.tensor([1.0]), source=0)

    def test_rejects_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            depth_first_search(torch.rand(3, 3), source=5)


class TestDFSBatched:
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
        disc, finish, pred = depth_first_search(adj, source=0)
        assert disc.shape == (2, 2)
        assert finish.shape == (2, 2)
        assert pred.shape == (2, 2)


class TestDFSMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        adj = torch.rand(5, 5, device="meta")
        disc, finish, pred = depth_first_search(adj, source=0)
        assert disc.shape == (5,)
        assert finish.shape == (5,)
        assert pred.shape == (5,)


class TestDFSTimingProperties:
    """Tests for DFS timing properties (parenthesis theorem)."""

    def test_discovery_before_finish(self):
        """Discovery time always before finish time for reachable nodes."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf, inf],
                [inf, 0.0, 1.0, inf],
                [inf, inf, 0.0, 1.0],
                [inf, inf, inf, 0.0],
            ]
        )
        disc, finish, pred = depth_first_search(adj, source=0)

        for i in range(4):
            if disc[i] >= 0:  # Node is reachable
                assert disc[i] < finish[i]

    def test_times_are_unique(self):
        """All discovery and finish times should be unique."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        disc, finish, pred = depth_first_search(adj, source=0)

        # Get all times for reachable nodes
        reachable_disc = [d.item() for d in disc if d >= 0]
        reachable_finish = [f.item() for f in finish if f >= 0]

        # All times should be unique
        all_times = reachable_disc + reachable_finish
        assert len(all_times) == len(set(all_times))
