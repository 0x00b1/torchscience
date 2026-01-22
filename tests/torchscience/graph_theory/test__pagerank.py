"""Tests for pagerank."""

import pytest
import torch

from torchscience.graph import pagerank


class TestPageRankBasic:
    """Basic correctness tests."""

    def test_cycle_graph(self):
        """Cycle graph: all nodes should have equal PageRank."""
        # 0 -> 1 -> 2 -> 0
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        pr = pagerank(adj)
        # All nodes equally important in a cycle
        torch.testing.assert_close(pr, torch.ones(3) / 3, atol=1e-5, rtol=1e-5)

    def test_star_graph_hub_highest(self):
        """Star graph: hub should have highest PageRank."""
        # 0 is hub, 1,2,3 are leaves pointing to 0
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        pr = pagerank(adj)
        # Hub (node 0) should have highest score
        assert pr[0] == pr.max()
        # Leaves should have equal scores
        torch.testing.assert_close(pr[1], pr[2])
        torch.testing.assert_close(pr[2], pr[3])

    def test_sums_to_one(self):
        """PageRank scores should sum to 1."""
        adj = torch.rand(10, 10)
        pr = pagerank(adj)
        torch.testing.assert_close(
            pr.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5
        )

    def test_non_negative(self):
        """PageRank scores should be non-negative."""
        adj = torch.rand(10, 10)
        pr = pagerank(adj)
        assert (pr >= 0).all()

    def test_weighted_edges(self):
        """Higher weight edges should contribute more."""
        # Strong edge 0 -> 1, weak edge 0 -> 2
        adj = torch.tensor(
            [
                [0.0, 10.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        pr = pagerank(adj)
        # Node 1 should have higher score than node 2
        assert pr[1] > pr[2]


class TestPageRankDanglingNodes:
    """Tests for handling dangling nodes (no outgoing edges)."""

    def test_single_dangling_node(self):
        """Graph with one dangling node."""
        # Node 2 has no outgoing edges
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        pr = pagerank(adj)
        assert pr.sum().isclose(torch.tensor(1.0), atol=1e-5)
        assert (pr >= 0).all()
        # Dangling node should still have some PageRank
        assert pr[2] > 0

    def test_all_dangling_nodes(self):
        """Graph with no edges (all dangling)."""
        adj = torch.zeros(4, 4)
        pr = pagerank(adj)
        # Should be uniform when all nodes are dangling
        torch.testing.assert_close(pr, torch.ones(4) / 4, atol=1e-5, rtol=1e-5)


class TestPageRankPersonalization:
    """Tests for personalized PageRank."""

    def test_personalization_bias(self):
        """Personalization should bias toward specified nodes."""
        adj = torch.rand(5, 5)
        # Bias toward node 0
        pers = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        pr = pagerank(adj, personalization=pers)
        # Node 0 should have higher than uniform score
        assert pr[0] > 1 / 5

    def test_personalization_normalized(self):
        """Unnormalized personalization should work."""
        adj = torch.rand(4, 4)
        # Unnormalized personalization (will be normalized internally)
        pers = torch.tensor([10.0, 0.0, 0.0, 0.0])
        pr = pagerank(adj, personalization=pers)
        assert pr.sum().isclose(torch.tensor(1.0), atol=1e-5)

    def test_uniform_personalization_equals_default(self):
        """Uniform personalization should equal default."""
        adj = torch.rand(5, 5)
        pr_default = pagerank(adj)
        pr_uniform = pagerank(adj, personalization=torch.ones(5))
        torch.testing.assert_close(
            pr_default, pr_uniform, atol=1e-5, rtol=1e-5
        )


class TestPageRankAlpha:
    """Tests for different damping factors."""

    def test_high_alpha_follows_links(self):
        """High alpha should make PageRank follow link structure more."""
        # Linear chain: 0 -> 1 -> 2
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        pr_low = pagerank(adj, alpha=0.5)
        pr_high = pagerank(adj, alpha=0.99)
        # With high alpha, endpoint (node 2) should accumulate more
        assert pr_high[2] > pr_low[2]

    def test_low_alpha_more_uniform(self):
        """Low alpha should give more uniform distribution."""
        adj = torch.rand(5, 5)
        pr_low = pagerank(adj, alpha=0.1)
        pr_high = pagerank(adj, alpha=0.9)
        # Low alpha should have lower variance
        assert pr_low.var() < pr_high.var()


class TestPageRankGradients:
    """Tests for autograd support."""

    def test_gradcheck(self):
        """Gradcheck for PageRank."""
        adj = torch.rand(4, 4, dtype=torch.float64, requires_grad=True) + 0.1

        def func(x):
            return pagerank(x, max_iter=50)

        assert torch.autograd.gradcheck(
            func, adj, eps=1e-5, atol=1e-3, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Second-order gradient check."""
        adj = torch.rand(3, 3, dtype=torch.float64, requires_grad=True) + 0.1

        def func(x):
            return pagerank(x, max_iter=30)

        assert torch.autograd.gradgradcheck(
            func, adj, eps=1e-5, atol=1e-2, rtol=1e-2
        )

    def test_gradient_flows(self):
        """Verify gradients flow through PageRank."""
        adj = torch.rand(4, 4, requires_grad=True)
        pr = pagerank(adj)
        loss = pr.sum()
        loss.backward()
        assert adj.grad is not None
        assert not adj.grad.isnan().any()


class TestPageRankBatched:
    """Tests for batched computation."""

    def test_batch_2d(self):
        """Batched with single batch dimension."""
        adj = torch.rand(3, 5, 5)
        pr = pagerank(adj)
        assert pr.shape == (3, 5)
        # Each should sum to 1
        torch.testing.assert_close(
            pr.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
        )

    def test_batch_3d(self):
        """Batched with multiple batch dimensions."""
        adj = torch.rand(2, 3, 4, 4)
        pr = pagerank(adj)
        assert pr.shape == (2, 3, 4)

    def test_batch_consistency(self):
        """Batched result matches individual computations."""
        adj1 = torch.rand(4, 4)
        adj2 = torch.rand(4, 4)

        pr1 = pagerank(adj1)
        pr2 = pagerank(adj2)

        batch_adj = torch.stack([adj1, adj2])
        pr_batch = pagerank(batch_adj)

        torch.testing.assert_close(pr_batch[0], pr1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pr_batch[1], pr2, atol=1e-5, rtol=1e-5)


class TestPageRankDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        adj = torch.rand(4, 4, dtype=dtype)
        pr = pagerank(adj)
        assert pr.dtype == dtype


class TestPageRankValidation:
    """Tests for input validation."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            pagerank(torch.tensor([1.0, 2.0, 3.0]))

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            pagerank(torch.rand(3, 4))

    def test_rejects_invalid_alpha(self):
        """Should reject alpha outside (0, 1)."""
        adj = torch.rand(3, 3)
        with pytest.raises(ValueError, match="alpha"):
            pagerank(adj, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            pagerank(adj, alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            pagerank(adj, alpha=-0.5)

    def test_rejects_wrong_personalization_shape(self):
        """Should reject personalization with wrong shape."""
        adj = torch.rand(3, 3)
        with pytest.raises(ValueError, match="personalization"):
            pagerank(adj, personalization=torch.rand(4))


class TestPageRankReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_networkx(self, N):
        """Compare to NetworkX PageRank."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        # Random directed graph
        adj_np = numpy.random.rand(N, N)
        adj_np = (adj_np > 0.5).astype(float)  # Binary adjacency

        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        pr_ours = pagerank(adj, alpha=0.85, tol=1e-8, max_iter=200)

        # NetworkX implementation
        G = networkx.DiGraph(adj_np)
        pr_nx = networkx.pagerank(G, alpha=0.85, tol=1e-8, max_iter=200)
        pr_nx_tensor = torch.tensor([pr_nx[i] for i in range(N)])

        torch.testing.assert_close(pr_ours, pr_nx_tensor, atol=1e-4, rtol=1e-4)

    def test_matches_networkx_weighted(self):
        """Compare weighted graph to NetworkX."""
        networkx = pytest.importorskip("networkx")
        numpy = pytest.importorskip("numpy")

        N = 8
        adj_np = numpy.random.rand(N, N)

        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        pr_ours = pagerank(adj, alpha=0.85, tol=1e-8, max_iter=200)

        # NetworkX implementation
        G = networkx.DiGraph()
        for i in range(N):
            for j in range(N):
                if adj_np[i, j] > 0:
                    G.add_edge(i, j, weight=adj_np[i, j])
        pr_nx = networkx.pagerank(
            G, alpha=0.85, tol=1e-8, max_iter=200, weight="weight"
        )
        pr_nx_tensor = torch.tensor([pr_nx[i] for i in range(N)])

        torch.testing.assert_close(pr_ours, pr_nx_tensor, atol=1e-4, rtol=1e-4)


class TestPageRankConvergence:
    """Tests for convergence behavior."""

    def test_converges_within_max_iter(self):
        """Should converge for reasonable graphs."""
        adj = torch.rand(20, 20)
        pr1 = pagerank(adj, max_iter=50, tol=1e-6)
        pr2 = pagerank(adj, max_iter=100, tol=1e-6)
        # Should be the same if converged
        torch.testing.assert_close(pr1, pr2, atol=1e-5, rtol=1e-5)

    def test_empty_graph(self):
        """Empty graph (0 nodes) should return empty tensor."""
        adj = torch.rand(0, 0)
        pr = pagerank(adj)
        assert pr.shape == (0,)
