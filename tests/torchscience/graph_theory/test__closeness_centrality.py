"""Tests for closeness_centrality."""

import pytest
import torch

from torchscience.graph import closeness_centrality


class TestClosenessCentralityBasic:
    """Basic functionality tests."""

    def test_star_graph_center_highest(self):
        """Center of star graph has highest closeness."""
        inf = float("inf")
        # Star: node 0 connected to all others
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, inf, inf],
                [1.0, inf, 0.0, inf],
                [1.0, inf, inf, 0.0],
            ]
        )
        cc = closeness_centrality(adj)

        assert cc.shape == (4,)
        assert cc[0] > cc[1]  # Center has highest closeness
        assert cc[1] == cc[2] == cc[3]  # Leaves are equal

    def test_chain_graph(self):
        """Chain graph: middle nodes have higher closeness."""
        inf = float("inf")
        # Chain: 0 - 1 - 2 - 3
        adj = torch.tensor(
            [
                [0.0, 1.0, inf, inf],
                [1.0, 0.0, 1.0, inf],
                [inf, 1.0, 0.0, 1.0],
                [inf, inf, 1.0, 0.0],
            ]
        )
        cc = closeness_centrality(adj)

        # Middle nodes (1, 2) have higher closeness than endpoints
        assert cc[1] > cc[0]
        assert cc[2] > cc[3]
        assert torch.isclose(cc[1], cc[2])  # Symmetric
        assert torch.isclose(cc[0], cc[3])

    def test_complete_graph_all_equal(self):
        """Complete graph: all nodes have equal closeness."""
        N = 5
        adj = torch.ones(N, N) - torch.eye(N)
        cc = closeness_centrality(adj)

        assert torch.allclose(cc, cc[0].expand(N))

    def test_disconnected_node(self):
        """Disconnected node has zero closeness."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [1.0, 0.0, inf],
                [inf, inf, 0.0],  # Node 2 is isolated
            ]
        )
        cc = closeness_centrality(adj)

        assert cc[2] == 0.0  # Isolated node has 0 closeness


class TestClosenessCentralityNormalization:
    """Tests for normalization options."""

    def test_normalized_output_range(self):
        """Normalized closeness should be in [0, 1] for unit weight graphs."""
        # Create a random connected graph with unit edge weights
        # (normalized closeness in [0,1] only guaranteed for unit weights)
        inf = float("inf")
        adj = torch.where(
            torch.rand(10, 10) > 0.3,
            torch.ones(10, 10),
            torch.full((10, 10), inf),
        )
        adj = torch.minimum(adj, adj.T)  # Make symmetric
        adj.fill_diagonal_(0)

        cc = closeness_centrality(adj, normalized=True)

        assert (cc >= 0).all()
        assert (cc <= 1.0 + 1e-6).all()  # Allow small numerical tolerance


class TestClosenessCentralityGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check via finite differences."""
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
            return closeness_centrality(adj).sum()

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    @pytest.mark.xfail(
        reason="torch.minimum used in Floyd-Warshall is not twice differentiable"
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
            return closeness_centrality(adj).sum()

        assert torch.autograd.gradgradcheck(func, (adj,), eps=1e-4, atol=1e-3)


class TestClosenessCentralityBatched:
    """Tests for batched computation."""

    def test_batched_shape(self):
        """Batched adjacency returns batched centrality."""
        adj = torch.rand(3, 5, 5)
        adj = adj + adj.transpose(-1, -2)
        adj.diagonal(dim1=-2, dim2=-1).fill_(0)

        cc = closeness_centrality(adj)

        assert cc.shape == (3, 5)

    def test_batched_matches_loop(self):
        """Batched result matches sequential computation."""
        adj = torch.rand(4, 6, 6)
        adj = adj + adj.transpose(-1, -2)
        adj.diagonal(dim1=-2, dim2=-1).fill_(0)

        cc_batched = closeness_centrality(adj)
        cc_loop = torch.stack([closeness_centrality(adj[i]) for i in range(4)])

        assert torch.allclose(cc_batched, cc_loop)


class TestClosenessCentralityMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        cc = closeness_centrality(adj)

        assert cc.shape == (5,)
        assert cc.device.type == "meta"


class TestClosenessCentralitySparse:
    """Tests for sparse tensor support."""

    def test_sparse_matches_dense(self):
        """Sparse input gives same result as dense."""
        adj_dense = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.5],
                [2.0, 1.5, 0.0],
            ]
        )
        adj_sparse = adj_dense.to_sparse()

        cc_dense = closeness_centrality(adj_dense)
        cc_sparse = closeness_centrality(adj_sparse)

        assert torch.allclose(cc_dense, cc_sparse)


class TestClosenessCentralityReference:
    """Tests comparing to reference implementations."""

    def test_matches_networkx(self):
        """Compare to NetworkX implementation."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        # Create random graph
        G = networkx.gnp_random_graph(10, 0.5, seed=42)
        G = networkx.Graph(G)  # Ensure undirected

        # Convert to adjacency tensor (use edge weight 1)
        adj_np = networkx.to_numpy_array(G)
        adj_np[adj_np == 0] = numpy.inf
        numpy.fill_diagonal(adj_np, 0)
        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        cc_ours = closeness_centrality(adj, normalized=True)

        # NetworkX implementation
        cc_nx = networkx.closeness_centrality(G)
        cc_nx_tensor = torch.tensor([cc_nx[i] for i in range(10)])

        assert torch.allclose(cc_ours, cc_nx_tensor, rtol=1e-4)
