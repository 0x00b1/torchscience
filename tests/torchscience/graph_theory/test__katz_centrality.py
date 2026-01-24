"""Tests for katz_centrality."""

import pytest
import torch

from torchscience.graph import katz_centrality


class TestKatzCentralityBasic:
    """Basic functionality tests."""

    def test_star_graph_center_highest(self):
        """Center of star graph has highest Katz centrality."""
        # Star: node 0 connected to all others
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        kc = katz_centrality(adj)

        assert kc.shape == (4,)
        assert kc[0] > kc[1]  # Center has highest centrality

    def test_chain_graph(self):
        """Chain graph: end nodes have lower centrality."""
        # Chain: 0 - 1 - 2 - 3
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        kc = katz_centrality(adj)

        # Middle nodes should have higher centrality
        assert kc[1] > kc[0]
        assert kc[2] > kc[3]

    def test_alpha_effect(self):
        """Higher alpha gives more weight to distant connections."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        kc_low = katz_centrality(adj, alpha=0.1)
        kc_high = katz_centrality(adj, alpha=0.4)

        # Higher alpha should increase relative importance of node 1 (center)
        ratio_low = kc_low[1] / kc_low[0]
        ratio_high = kc_high[1] / kc_high[0]

        assert ratio_high > ratio_low

    def test_beta_is_base_score(self):
        """Beta parameter adds base score to all nodes."""
        adj = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        # Use normalized=False to see the effect of beta on raw scores
        kc_beta1 = katz_centrality(adj, beta=1.0, normalized=False)
        kc_beta2 = katz_centrality(adj, beta=2.0, normalized=False)

        # All centralities should increase with higher beta
        assert (kc_beta2 > kc_beta1).all()


class TestKatzCentralityGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check via finite differences."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 1.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            return katz_centrality(adj, alpha=0.2).sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 1.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            return katz_centrality(adj, alpha=0.2).sum()

        assert torch.autograd.gradgradcheck(func, (adj,), eps=1e-4, atol=1e-3)


class TestKatzCentralityBatched:
    """Tests for batched computation."""

    def test_batched_shape(self):
        """Batched adjacency returns batched centrality."""
        adj = torch.rand(3, 5, 5)
        adj = adj + adj.transpose(-1, -2)

        kc = katz_centrality(adj, alpha=0.1)

        assert kc.shape == (3, 5)


class TestKatzCentralityMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        kc = katz_centrality(adj, alpha=0.1)

        assert kc.shape == (5,)
        assert kc.device.type == "meta"


class TestKatzCentralityReference:
    """Tests comparing to reference implementations."""

    def test_matches_networkx(self):
        """Compare to NetworkX implementation."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        # Create random graph
        G = networkx.gnp_random_graph(8, 0.4, seed=42)
        G = networkx.Graph(G)

        adj_np = networkx.to_numpy_array(G)
        adj = torch.from_numpy(adj_np).float()

        alpha = 0.1

        # Our implementation
        kc_ours = katz_centrality(adj, alpha=alpha, normalized=False)

        # NetworkX implementation
        kc_nx = networkx.katz_centrality_numpy(
            G, alpha=alpha, normalized=False
        )
        kc_nx_tensor = torch.tensor([kc_nx[i] for i in range(8)])

        assert torch.allclose(kc_ours, kc_nx_tensor, rtol=1e-4)
