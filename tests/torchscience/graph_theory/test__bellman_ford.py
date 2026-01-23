"""Tests for bellman_ford."""

import pytest
import torch

from torchscience.graph import (
    BellmanFordNegativeCycleError,
    bellman_ford,
)


class TestBellmanFordBasic:
    """Basic functionality tests."""

    def test_simple_chain(self):
        """Chain graph: 0 -> 1 -> 2."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = bellman_ford(adj, source=0)

        assert torch.allclose(dist, torch.tensor([0.0, 1.0, 3.0]))
        assert pred[0] == -1  # source
        assert pred[1] == 0
        assert pred[2] == 1

    def test_negative_edge(self):
        """Graph with negative edges but no negative cycle."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 4.0, inf],
                [inf, 0.0, -2.0],  # Negative edge
                [inf, inf, 0.0],
            ]
        )
        dist, pred = bellman_ford(adj, source=0)

        assert torch.allclose(dist, torch.tensor([0.0, 4.0, 2.0]))
        assert pred[2] == 1  # Via node 1

    def test_negative_edge_shortcut(self):
        """Negative edge creates shorter path."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 5.0, 3.0],
                [inf, 0.0, -4.0],  # This makes 0->1->2 = 5-4=1 < 3
                [inf, inf, 0.0],
            ]
        )
        dist, pred = bellman_ford(adj, source=0)

        assert dist[2] == 1.0  # Via 0->1->2 = 5 + (-4) = 1
        assert pred[2] == 1

    def test_unreachable_node(self):
        """Node not reachable from source."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = bellman_ford(adj, source=0)

        assert dist[2] == inf
        assert pred[2] == -1

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        dist, pred = bellman_ford(adj, source=0)

        assert dist.shape == (1,)
        assert dist[0] == 0.0
        assert pred[0] == -1


class TestBellmanFordNegativeCycle:
    """Tests for negative cycle detection."""

    def test_simple_negative_cycle(self):
        """Simple triangle with negative total weight."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [-3.0, inf, 0.0],  # Creates cycle 0->1->2->0 with weight -1
            ]
        )

        with pytest.raises(BellmanFordNegativeCycleError):
            bellman_ford(adj, source=0)

    def test_negative_cycle_reachable(self):
        """Negative cycle reachable from source."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf, inf],
                [inf, 0.0, 1.0, inf],
                [inf, -3.0, 0.0, inf],  # Cycle: 1->2->1 weight=-2
                [inf, inf, inf, 0.0],
            ]
        )

        with pytest.raises(BellmanFordNegativeCycleError):
            bellman_ford(adj, source=0)

    def test_no_negative_cycle(self):
        """Negative edges but no cycle."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, -2.0],
                [inf, inf, 0.0],
            ]
        )

        # Should not raise
        dist, pred = bellman_ford(adj, source=0)
        assert dist[2] == -1.0


class TestBellmanFordUndirected:
    """Tests for undirected graphs."""

    def test_undirected_symmetry(self):
        """Undirected graph finds paths in both directions."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )

        # From node 0
        dist0, _ = bellman_ford(adj, source=0, directed=False)
        # From node 2
        dist2, _ = bellman_ford(adj, source=2, directed=False)

        # In undirected graph, distance is symmetric
        assert dist0[2] == dist2[0]


class TestBellmanFordDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        adj = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=dtype,
        )
        dist, pred = bellman_ford(adj, source=0)

        assert dist.dtype == dtype
        assert pred.dtype == torch.int64


class TestBellmanFordBatched:
    """Tests for batched input support."""

    def test_batch_2d(self):
        """Batched with single batch dimension."""
        inf = float("inf")
        # Create 3 small graphs
        adj = torch.tensor(
            [
                [[0.0, 1.0, inf], [inf, 0.0, 2.0], [inf, inf, 0.0]],
                [[0.0, 3.0, inf], [inf, 0.0, 1.0], [inf, inf, 0.0]],
                [[0.0, 2.0, 4.0], [inf, 0.0, 1.0], [inf, inf, 0.0]],
            ]
        )

        dist, pred = bellman_ford(adj, source=0)
        assert dist.shape == (3, 3)
        assert pred.shape == (3, 3)

    def test_batch_3d(self):
        """Batched with multiple batch dimensions."""
        inf = float("inf")
        # Create 2x2 batch of small graphs
        adj = torch.tensor(
            [
                [
                    [[0.0, 1.0, inf], [inf, 0.0, 2.0], [inf, inf, 0.0]],
                    [[0.0, 3.0, inf], [inf, 0.0, 1.0], [inf, inf, 0.0]],
                ],
                [
                    [[0.0, 2.0, 4.0], [inf, 0.0, 1.0], [inf, inf, 0.0]],
                    [[0.0, 5.0, inf], [inf, 0.0, 3.0], [inf, inf, 0.0]],
                ],
            ]
        )

        dist, pred = bellman_ford(adj, source=0)
        assert dist.shape == (2, 2, 3)
        assert pred.shape == (2, 2, 3)

    def test_batch_consistency(self):
        """Batched result matches individual computations."""
        inf = float("inf")
        adj1 = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )
        adj2 = torch.tensor(
            [
                [0.0, 3.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ]
        )

        dist1, pred1 = bellman_ford(adj1, source=0)
        dist2, pred2 = bellman_ford(adj2, source=0)

        batch_adj = torch.stack([adj1, adj2])
        dist_batch, pred_batch = bellman_ford(batch_adj, source=0)

        torch.testing.assert_close(dist_batch[0], dist1)
        torch.testing.assert_close(dist_batch[1], dist2)
        torch.testing.assert_close(pred_batch[0], pred1)
        torch.testing.assert_close(pred_batch[1], pred2)

    def test_batch_with_negative_edges(self):
        """Batched computation with negative edges."""
        inf = float("inf")
        adj1 = torch.tensor(
            [
                [0.0, 4.0, inf],
                [inf, 0.0, -2.0],  # Negative edge
                [inf, inf, 0.0],
            ]
        )
        adj2 = torch.tensor(
            [
                [0.0, 5.0, 3.0],
                [inf, 0.0, -4.0],  # Negative edge makes 0->1->2 shorter
                [inf, inf, 0.0],
            ]
        )

        dist1, _ = bellman_ford(adj1, source=0)
        dist2, _ = bellman_ford(adj2, source=0)

        batch_adj = torch.stack([adj1, adj2])
        dist_batch, _ = bellman_ford(batch_adj, source=0)

        torch.testing.assert_close(dist_batch[0], dist1)
        torch.testing.assert_close(dist_batch[1], dist2)
        assert dist_batch[0, 2] == 2.0  # 0->1->2 = 4 + (-2) = 2
        assert dist_batch[1, 2] == 1.0  # 0->1->2 = 5 + (-4) = 1

    def test_batch_gradient(self):
        """Gradient works with batched input."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [[0.0, 1.0, inf], [inf, 0.0, 2.0], [inf, inf, 0.0]],
                [[0.0, 3.0, inf], [inf, 0.0, 1.0], [inf, inf, 0.0]],
            ],
            requires_grad=True,
        )

        dist, _ = bellman_ford(adj, source=0)
        dist.sum().backward()

        assert adj.grad is not None
        assert adj.grad.shape == adj.shape
        # Each graph: edges (0,1) and (1,2) are on shortest paths
        assert adj.grad[0, 0, 1] != 0
        assert adj.grad[0, 1, 2] != 0
        assert adj.grad[1, 0, 1] != 0
        assert adj.grad[1, 1, 2] != 0


class TestBellmanFordValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            bellman_ford(torch.tensor([1.0, 2.0, 3.0]), source=0)

    def test_accepts_3d_input(self):
        """Should accept 3D input (batched)."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ]
        )
        dist, pred = bellman_ford(adj, source=0)
        assert dist.shape == (2, 2)
        assert pred.shape == (2, 2)

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            bellman_ford(torch.rand(3, 4), source=0)

    def test_rejects_negative_source(self):
        """Should reject negative source index."""
        with pytest.raises(ValueError, match="source"):
            bellman_ford(torch.rand(3, 3), source=-1)

    def test_rejects_source_out_of_range(self):
        """Should reject source >= N."""
        with pytest.raises(ValueError, match="source"):
            bellman_ford(torch.rand(3, 3), source=5)


class TestBellmanFordGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check via finite differences."""
        # Use values that give unique shortest paths
        large = 1e6
        adj = torch.tensor(
            [
                [1.0, 2.0, 10.0],
                [large, 1.0, 3.0],
                [large, large, 1.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            dist, _ = bellman_ford(adj, source=0)
            return dist[dist < large].sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    def test_gradient_with_negative_edge(self):
        """Gradient works with negative edges."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 5.0, 10.0],
                [inf, 0.0, -3.0],  # Negative edge
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = bellman_ford(adj, source=0)
        dist.sum().backward()

        # Path to 2 uses edges (0,1) and (1,2)
        assert adj.grad[0, 1] != 0
        assert adj.grad[1, 2] != 0

    def test_gradient_accumulates(self):
        """Gradient accumulates for edges used in multiple paths."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = bellman_ford(adj, source=0)
        dist.sum().backward()

        # Edge (0,1) is used for both path to 1 and path to 2
        assert adj.grad[0, 1] == 2.0
        # Edge (1,2) is only used for path to 2
        assert adj.grad[1, 2] == 1.0


class TestBellmanFordReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy(self, N):
        """Compare to scipy.sparse.csgraph.bellman_ford."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random graph with ~30% edge density, allowing negative edges
        adj_np = numpy.random.rand(N, N) * 10 - 2  # Range [-2, 8]
        adj_np[numpy.random.rand(N, N) > 0.3] = numpy.inf
        numpy.fill_diagonal(adj_np, 0)

        adj = torch.from_numpy(adj_np).float()

        for source in [0, N // 2]:
            try:
                # Scipy implementation
                dist_scipy, pred_scipy = scipy_sparse.csgraph.bellman_ford(
                    adj_np, indices=source, return_predecessors=True
                )

                # Our implementation
                dist_ours, pred_ours = bellman_ford(adj, source=source)

                # Compare distances
                assert torch.allclose(
                    dist_ours,
                    torch.from_numpy(dist_scipy).float(),
                    rtol=1e-5,
                    atol=1e-5,
                    equal_nan=True,
                ), f"Distance mismatch for source={source}"

            except (
                BellmanFordNegativeCycleError,
                scipy_sparse.csgraph.NegativeCycleError,
            ):
                # Both should detect negative cycles
                pass


class TestBellmanFordMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        dist, pred = bellman_ford(adj, source=0)

        assert dist.shape == (5,)
        assert pred.shape == (5,)
        assert dist.device.type == "meta"
        assert pred.device.type == "meta"
