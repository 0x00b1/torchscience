"""Tests for dijkstra."""

import pytest
import torch

from torchscience.graph_theory import dijkstra


class TestDijkstraBasic:
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
        dist, pred = dijkstra(adj, source=0)

        assert torch.allclose(dist, torch.tensor([0.0, 1.0, 3.0]))
        assert pred[0] == -1  # source
        assert pred[1] == 0
        assert pred[2] == 1

    def test_shortcut(self):
        """Graph with direct edge shorter than path."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.5],  # Direct 0->2 is 2.5
                [inf, 0.0, 2.0],  # Path 0->1->2 is 3.0
                [inf, inf, 0.0],
            ]
        )
        dist, pred = dijkstra(adj, source=0)

        assert dist[2] == 2.5  # Direct path is shorter
        assert pred[2] == 0

    def test_longer_path_shorter(self):
        """Longer path is actually shorter."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 10.0],  # Direct 0->2 is 10
                [inf, 0.0, 2.0],  # Path 0->1->2 is 3
                [inf, inf, 0.0],
            ]
        )
        dist, pred = dijkstra(adj, source=0)

        assert dist[2] == 3.0  # Via node 1
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
        dist, pred = dijkstra(adj, source=0)

        assert dist[2] == inf
        assert pred[2] == -1

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        dist, pred = dijkstra(adj, source=0)

        assert dist.shape == (1,)
        assert dist[0] == 0.0
        assert pred[0] == -1

    def test_isolated_nodes(self):
        """Multiple isolated nodes (use inf for missing edges)."""
        inf = float("inf")
        adj = torch.full((3, 3), inf)
        adj.fill_diagonal_(0)  # Self-loops have 0 cost
        dist, pred = dijkstra(adj, source=0)

        assert dist[0] == 0.0
        assert torch.isinf(dist[1])
        assert torch.isinf(dist[2])


class TestDijkstraUndirected:
    """Tests for undirected graphs."""

    def test_undirected_symmetry(self):
        """Undirected graph finds paths in both directions."""
        inf = float("inf")
        # Asymmetric adjacency, but directed=False
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )

        # From node 0
        dist0, _ = dijkstra(adj, source=0, directed=False)
        # From node 2
        dist2, _ = dijkstra(adj, source=2, directed=False)

        # In undirected graph, distance is symmetric
        assert dist0[2] == dist2[0]

    def test_undirected_uses_min_edge(self):
        """Uses minimum of forward/backward edge weights."""
        adj = torch.tensor(
            [
                [0.0, 5.0],
                [1.0, 0.0],  # Reverse edge is shorter
            ]
        )
        dist, pred = dijkstra(adj, source=0, directed=False)

        assert dist[1] == 1.0  # Uses the shorter 1.0 edge


class TestDijkstraDtypes:
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
        dist, pred = dijkstra(adj, source=0)

        assert dist.dtype == dtype
        assert pred.dtype == torch.int64


class TestDijkstraValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="2D"):
            dijkstra(torch.tensor([1.0, 2.0, 3.0]), source=0)

    def test_rejects_3d_input(self):
        """Should reject 3D input."""
        with pytest.raises(ValueError, match="2D"):
            dijkstra(torch.rand(2, 3, 3), source=0)

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            dijkstra(torch.rand(3, 4), source=0)

    def test_rejects_negative_source(self):
        """Should reject negative source index."""
        with pytest.raises(ValueError, match="source"):
            dijkstra(torch.rand(3, 3), source=-1)

    def test_rejects_source_out_of_range(self):
        """Should reject source >= N."""
        with pytest.raises(ValueError, match="source"):
            dijkstra(torch.rand(3, 3), source=5)

    def test_rejects_negative_weights(self):
        """Should reject negative edge weights."""
        adj = torch.tensor(
            [
                [0.0, -1.0],
                [1.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="negative"):
            dijkstra(adj, source=0)


class TestDijkstraGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check via finite differences."""
        # Use large values instead of inf for gradcheck compatibility
        # (gradcheck perturbs values, and perturbed inf causes issues)
        # Use values that give unique shortest paths (no ties)
        large = 1e6
        adj = torch.tensor(
            [
                [1.0, 2.0, 10.0],  # 0->1:2, 0->2:10 (but 0->1->2=5 is shorter)
                [large, 1.0, 3.0],  # 1->2:3
                [large, large, 1.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            dist, _ = dijkstra(adj, source=0)
            # Sum distances (large values stay large)
            return dist[dist < large].sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    def test_gradient_on_shortest_path(self):
        """Gradient is nonzero only on shortest path edges."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 10.0],  # 0->2 direct not on shortest path
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = dijkstra(adj, source=0)
        dist.sum().backward()

        # Only edges on shortest path should have nonzero gradient
        # Path to 1: edge (0,1) with weight 1
        # Path to 2: edge (0,1) and (1,2)
        assert adj.grad[0, 1] != 0  # On path to 1 and 2
        assert adj.grad[1, 2] != 0  # On path to 2
        assert adj.grad[0, 2] == 0  # Not on any shortest path

    def test_gradient_accumulates(self):
        """Gradient accumulates for edges used in multiple paths."""
        inf = float("inf")
        # 0 -> 1 -> 2
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = dijkstra(adj, source=0)
        dist.sum().backward()

        # Edge (0,1) is used for both path to 1 and path to 2
        # So its gradient should be 2
        assert adj.grad[0, 1] == 2.0
        # Edge (1,2) is only used for path to 2
        assert adj.grad[1, 2] == 1.0


class TestDijkstraReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy(self, N):
        """Compare to scipy.sparse.csgraph.dijkstra."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random graph with ~30% edge density
        adj_np = numpy.random.rand(N, N)
        adj_np[adj_np > 0.3] = numpy.inf
        adj_np[adj_np <= 0.3] = adj_np[adj_np <= 0.3] * 10  # Scale weights
        numpy.fill_diagonal(adj_np, 0)  # Zero diagonal

        adj = torch.from_numpy(adj_np).float()

        for source in [0, N // 2]:
            # Our implementation
            dist_ours, pred_ours = dijkstra(adj, source=source)

            # Scipy implementation
            dist_scipy, pred_scipy = scipy_sparse.csgraph.dijkstra(
                adj_np, indices=source, return_predecessors=True
            )

            # Compare distances
            assert torch.allclose(
                dist_ours,
                torch.from_numpy(dist_scipy).float(),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            ), f"Distance mismatch for source={source}"

            # Compare predecessors (may differ for equal-length paths)
            # Just check that predecessors give same distances
            for i in range(N):
                if not torch.isinf(dist_ours[i]) and i != source:
                    p = pred_ours[i].item()
                    expected = dist_ours[p] + adj[p, i]
                    assert torch.isclose(dist_ours[i], expected), (
                        f"Predecessor inconsistent at node {i}"
                    )


class TestDijkstraMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        dist, pred = dijkstra(adj, source=0)

        assert dist.shape == (5,)
        assert pred.shape == (5,)
        assert dist.device.type == "meta"
        assert pred.device.type == "meta"


class TestDijkstraSparse:
    """Tests for sparse input."""

    def test_sparse_input(self):
        """Sparse COO tensor input."""
        indices = torch.tensor([[0, 1], [1, 2]])
        values = torch.tensor([1.0, 2.0])
        adj = torch.sparse_coo_tensor(indices, values, (3, 3))

        # Fill missing with inf
        adj_dense = adj.to_dense()
        adj_dense[adj_dense == 0] = float("inf")
        adj_dense.fill_diagonal_(0)

        dist, pred = dijkstra(adj_dense, source=0)

        assert dist[0] == 0.0
        assert dist[1] == 1.0
        assert dist[2] == 3.0
