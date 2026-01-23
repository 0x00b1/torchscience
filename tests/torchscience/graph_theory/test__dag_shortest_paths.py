"""Tests for dag_shortest_paths."""

import pytest
import torch

from torchscience.graph import dag_shortest_paths


class TestDAGShortestPathsBasic:
    def test_simple_dag(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 4.0],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = dag_shortest_paths(adj, source=0)

        assert torch.allclose(dist, torch.tensor([0.0, 1.0, 3.0]))
        assert pred[2] == 1  # Via node 1

    def test_matches_dijkstra(self):
        """For DAGs, should match Dijkstra."""
        from torchscience.graph import dijkstra

        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 2.0, 5.0, inf],
                [inf, 0.0, 1.0, 3.0],
                [inf, inf, 0.0, 1.0],
                [inf, inf, inf, 0.0],
            ]
        )
        dist_dag, _ = dag_shortest_paths(adj, source=0)
        dist_dij, _ = dijkstra(adj, source=0)

        assert torch.allclose(dist_dag, dist_dij)

    def test_unreachable_node(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = dag_shortest_paths(adj, source=0)

        assert torch.isinf(dist[2])
        assert pred[2] == -1

    def test_single_node(self):
        adj = torch.tensor([[0.0]])
        dist, pred = dag_shortest_paths(adj, source=0)

        assert dist[0] == 0.0
        assert pred[0] == -1

    def test_diamond_dag(self):
        """Diamond shape: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.0, inf],
                [inf, 0.0, inf, 3.0],
                [inf, inf, 0.0, 1.0],
                [inf, inf, inf, 0.0],
            ]
        )
        dist, pred = dag_shortest_paths(adj, source=0)

        # Path 0 -> 2 -> 3 costs 2 + 1 = 3
        # Path 0 -> 1 -> 3 costs 1 + 3 = 4
        assert dist[3] == 3.0
        assert pred[3] == 2  # Via node 2

    def test_non_zero_source(self):
        """Test with non-zero source."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ]
        )
        dist, pred = dag_shortest_paths(adj, source=1)

        assert torch.isinf(dist[0])  # Can't reach 0 from 1
        assert dist[1] == 0.0
        assert dist[2] == 2.0


class TestDAGShortestPathsCycle:
    def test_rejects_cycle(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [1.0, inf, 0.0],  # Cycle
            ]
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            dag_shortest_paths(adj, source=0)

    def test_ignores_self_loop(self):
        """Self-loops are ignored (don't affect shortest paths)."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [1.0, 1.0, inf],  # Self-loop at node 0 with weight 1.0
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ]
        )
        # Should succeed - self-loops are harmless for shortest path computation
        dist, pred = dag_shortest_paths(adj, source=0)
        assert dist[0] == 0.0  # Self-loop doesn't help
        assert dist[1] == 1.0
        assert dist[2] == 2.0


class TestDAGShortestPathsGradients:
    def test_gradcheck(self):
        # Use inf for missing edges (not large values) so gradcheck doesn't perturb them
        # into finite values that could be detected as back-edges
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 2.0, 10.0],
                [inf, 0.0, 3.0],
                [inf, inf, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            dist, _ = dag_shortest_paths(adj, source=0)
            # Only sum finite distances
            finite_mask = ~torch.isinf(dist)
            return dist[finite_mask].sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    def test_gradient_on_shortest_path(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 10.0],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = dag_shortest_paths(adj, source=0)
        dist.sum().backward()

        assert adj.grad[0, 1] != 0
        assert adj.grad[1, 2] != 0
        assert adj.grad[0, 2] == 0  # Not on shortest path

    def test_gradient_accumulates(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = dag_shortest_paths(adj, source=0)
        dist.sum().backward()

        # Edge (0,1) used for paths to both 1 and 2
        assert adj.grad[0, 1] == 2.0
        assert adj.grad[1, 2] == 1.0

    def test_gradient_diamond(self):
        """Gradient flows through shortest path in diamond."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.0, inf],
                [inf, 0.0, inf, 10.0],  # Long path through 1
                [inf, inf, 0.0, 1.0],  # Short path through 2
                [inf, inf, inf, 0.0],
            ],
            requires_grad=True,
        )

        dist, _ = dag_shortest_paths(adj, source=0)
        dist[3].backward()  # Gradient w.r.t. dist to node 3

        # Shortest path to 3 is 0 -> 2 -> 3
        assert adj.grad[0, 2] == 1.0
        assert adj.grad[2, 3] == 1.0
        # Not on shortest path
        assert adj.grad[0, 1] == 0.0
        assert adj.grad[1, 3] == 0.0


class TestDAGShortestPathsValidation:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            dag_shortest_paths(torch.tensor([1.0]), source=0)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2D"):
            dag_shortest_paths(torch.rand(2, 3, 3), source=0)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            dag_shortest_paths(torch.rand(3, 4), source=0)

    def test_rejects_negative_source(self):
        with pytest.raises(ValueError, match="source"):
            dag_shortest_paths(torch.rand(3, 3), source=-1)

    def test_rejects_source_out_of_range(self):
        with pytest.raises(ValueError, match="source"):
            dag_shortest_paths(torch.rand(3, 3), source=5)


class TestDAGShortestPathsMeta:
    def test_meta_shape(self):
        adj = torch.rand(5, 5, device="meta")
        dist, pred = dag_shortest_paths(adj, source=0)
        assert dist.shape == (5,)
        assert pred.shape == (5,)
        assert dist.device.type == "meta"
        assert pred.device.type == "meta"


class TestDAGShortestPathsDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 2.0],
                [inf, inf, 0.0],
            ],
            dtype=dtype,
        )
        dist, pred = dag_shortest_paths(adj, source=0)

        assert dist.dtype == dtype
        assert pred.dtype == torch.int64
