"""Tests for betweenness_centrality."""

import pytest
import torch

from torchscience.graph import betweenness_centrality


class TestBetweennessCentralityBasic:
    """Basic functionality tests."""

    def test_star_graph_center_highest(self):
        """Center of star graph has highest betweenness."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        bc = betweenness_centrality(adj)

        assert bc[0] > bc[1]  # Center on all shortest paths

    def test_chain_graph_middle_highest(self):
        """Chain: middle nodes have highest betweenness."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        bc = betweenness_centrality(adj)

        # Node 2 (center) should have highest betweenness
        assert bc[2] > bc[0]
        assert bc[2] > bc[4]

    def test_complete_graph_all_equal(self):
        """Complete graph: all nodes have equal betweenness."""
        N = 5
        adj = torch.ones(N, N) - torch.eye(N)
        bc = betweenness_centrality(adj)

        assert torch.allclose(bc, bc[0].expand(N), rtol=1e-4)

    def test_endpoint_zero_betweenness(self):
        """Endpoints have zero betweenness (not on any path between others)."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        bc = betweenness_centrality(adj, normalized=False)

        # Endpoints 0 and 2 are not on path between any other pair
        assert bc[0] == 0.0
        assert bc[2] == 0.0
        assert bc[1] > 0.0


class TestBetweennessCentralityGradients:
    """Tests for gradient computation."""

    @pytest.mark.xfail(
        reason="Betweenness centrality involves discrete shortest path selection"
    )
    def test_gradcheck(self):
        """Gradient check via finite differences."""
        # Use weighted graph for continuous gradients
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            return betweenness_centrality(adj).sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    @pytest.mark.xfail(
        reason="Betweenness centrality involves discrete shortest path selection"
    )
    def test_gradgradcheck(self):
        """Second-order gradient check."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            return betweenness_centrality(adj).sum()

        assert torch.autograd.gradgradcheck(func, (adj,), eps=1e-4, atol=1e-3)


class TestBetweennessCentralityBatched:
    """Tests for batched computation."""

    def test_batched_shape(self):
        """Batched adjacency returns batched centrality."""
        adj = torch.rand(3, 5, 5)
        adj = adj + adj.transpose(-1, -2)

        bc = betweenness_centrality(adj)

        assert bc.shape == (3, 5)


class TestBetweennessCentralityMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        bc = betweenness_centrality(adj)

        assert bc.shape == (5,)
        assert bc.device.type == "meta"


class TestBetweennessCentralityReference:
    """Tests comparing to reference implementations."""

    def test_matches_networkx(self):
        """Compare to NetworkX implementation."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        G = networkx.gnp_random_graph(8, 0.5, seed=42)
        G = networkx.Graph(G)

        adj_np = networkx.to_numpy_array(G)
        adj = torch.from_numpy(adj_np).float()

        bc_ours = betweenness_centrality(adj, normalized=True)

        bc_nx = networkx.betweenness_centrality(G, normalized=True)
        bc_nx_tensor = torch.tensor([bc_nx[i] for i in range(8)])

        assert torch.allclose(bc_ours, bc_nx_tensor, rtol=1e-3)
