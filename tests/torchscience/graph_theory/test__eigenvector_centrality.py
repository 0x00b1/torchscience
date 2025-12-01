"""Tests for eigenvector_centrality."""

import pytest
import torch

from torchscience.graph_theory import eigenvector_centrality


class TestEigenvectorCentralityBasic:
    """Basic functionality tests."""

    def test_star_graph_center_highest(self):
        """Center of star graph has highest eigenvector centrality."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        ec = eigenvector_centrality(adj)

        assert ec.shape == (4,)
        assert ec[0] > ec[1]

    def test_complete_graph_all_equal(self):
        """Complete graph: all nodes have equal centrality."""
        N = 5
        adj = torch.ones(N, N) - torch.eye(N)
        ec = eigenvector_centrality(adj)

        # All should be equal (within tolerance)
        assert torch.allclose(ec, ec[0].expand(N), rtol=1e-4)

    def test_positive_centrality(self):
        """All centralities should be non-negative."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        ec = eigenvector_centrality(adj)

        assert (ec >= 0).all()


class TestEigenvectorCentralityGradients:
    """Tests for gradient computation."""

    @pytest.mark.xfail(
        reason="Eigenvector centrality uses sign/clamp which have discontinuous gradients"
    )
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
            return eigenvector_centrality(adj).sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    @pytest.mark.xfail(
        reason="Second-order gradients through eigendecomposition may not be stable"
    )
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
            return eigenvector_centrality(adj).sum()

        assert torch.autograd.gradgradcheck(func, (adj,), eps=1e-4, atol=1e-3)


class TestEigenvectorCentralityBatched:
    """Tests for batched computation."""

    def test_batched_shape(self):
        """Batched adjacency returns batched centrality."""
        adj = torch.rand(3, 5, 5)
        adj = adj + adj.transpose(-1, -2)

        ec = eigenvector_centrality(adj)

        assert ec.shape == (3, 5)


class TestEigenvectorCentralityMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        ec = eigenvector_centrality(adj)

        assert ec.shape == (5,)
        assert ec.device.type == "meta"


class TestEigenvectorCentralityReference:
    """Tests comparing to reference implementations."""

    def test_matches_networkx(self):
        """Compare to NetworkX implementation."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        G = networkx.gnp_random_graph(8, 0.5, seed=42)
        G = networkx.Graph(G)

        adj_np = networkx.to_numpy_array(G)
        adj = torch.from_numpy(adj_np).float()

        ec_ours = eigenvector_centrality(adj)

        ec_nx = networkx.eigenvector_centrality_numpy(G)
        ec_nx_tensor = torch.tensor([ec_nx[i] for i in range(8)])

        # Compare up to sign (eigenvector can be negated)
        assert torch.allclose(ec_ours.abs(), ec_nx_tensor.abs(), rtol=1e-3)
