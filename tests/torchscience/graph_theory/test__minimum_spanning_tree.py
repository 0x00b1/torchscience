"""Tests for minimum_spanning_tree."""

import pytest
import torch

from torchscience.graph_theory import minimum_spanning_tree


class TestMinimumSpanningTreeBasic:
    """Basic functionality tests."""

    def test_simple_triangle(self):
        """Triangle graph: choose 2 smallest edges."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # MST should use edges (0,1) and (1,2) with weight 1+2=3
        assert weight.item() == 3.0
        assert edges.shape == (2, 2)

    def test_simple_chain(self):
        """Chain graph: all edges in MST."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [1.0, 0.0, 2.0],
                [inf, 2.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        assert weight.item() == 3.0  # 1 + 2
        assert edges.shape == (2, 2)

    def test_complete_graph_4(self):
        """Complete graph with 4 nodes."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 4.0, 3.0],
                [1.0, 0.0, 2.0, 5.0],
                [4.0, 2.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # MST: edges (0,1), (1,2), (0,3) with weights 1+2+3=6
        assert weight.item() == 6.0
        assert edges.shape == (3, 2)

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        weight, edges = minimum_spanning_tree(adj)

        assert weight.item() == 0.0
        assert edges.shape == (0, 2)

    def test_two_nodes(self):
        """Two connected nodes."""
        adj = torch.tensor(
            [
                [0.0, 5.0],
                [5.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        assert weight.item() == 5.0
        assert edges.shape == (1, 2)


class TestMinimumSpanningTreeDisconnected:
    """Tests for disconnected graphs."""

    def test_disconnected_graph(self):
        """Disconnected graph returns inf weight."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [1.0, 0.0, inf],
                [inf, inf, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        assert torch.isinf(weight)

    def test_isolated_nodes(self):
        """All isolated nodes."""
        inf = float("inf")
        adj = torch.full((3, 3), inf)
        adj.fill_diagonal_(0)
        weight, edges = minimum_spanning_tree(adj)

        assert torch.isinf(weight)


class TestMinimumSpanningTreeAsymmetric:
    """Tests for asymmetric adjacency matrices."""

    def test_uses_minimum_direction(self):
        """Uses minimum of forward/backward edge."""
        inf = float("inf")
        # Edge 0->1 has weight 5, edge 1->0 has weight 2
        adj = torch.tensor(
            [
                [0.0, 5.0, inf],
                [2.0, 0.0, 3.0],
                [inf, 3.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # Should use min(5, 2)=2 for edge 0-1, and 3 for edge 1-2
        assert weight.item() == 5.0  # 2 + 3


class TestMinimumSpanningTreeDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=dtype,
        )
        weight, edges = minimum_spanning_tree(adj)

        assert weight.dtype == dtype
        assert edges.dtype == torch.int64


class TestMinimumSpanningTreeValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="2D"):
            minimum_spanning_tree(torch.tensor([1.0, 2.0, 3.0]))

    def test_rejects_3d_input(self):
        """Should reject 3D input."""
        with pytest.raises(ValueError, match="2D"):
            minimum_spanning_tree(torch.rand(2, 3, 3))

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            minimum_spanning_tree(torch.rand(3, 4))


class TestMinimumSpanningTreeGradients:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Gradient check via finite differences."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 4.0],
                [1.0, 0.0, 2.0],
                [4.0, 2.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        def func(adj):
            weight, _ = minimum_spanning_tree(adj)
            return weight

        assert torch.autograd.gradcheck(func, (adj,), eps=1e-4, atol=1e-3)

    def test_gradient_only_on_mst_edges(self):
        """Gradient is nonzero only on MST edges."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 10.0],  # Edge (0,2) not in MST
                [1.0, 0.0, 2.0],
                [10.0, 2.0, 0.0],
            ],
            requires_grad=True,
        )

        weight, edges = minimum_spanning_tree(adj)
        weight.backward()

        # Edges in MST: (0,1) and (1,2)
        # For symmetric edges, gradient is split 0.5 to each direction
        assert adj.grad[0, 1] == 0.5
        assert adj.grad[1, 0] == 0.5
        assert adj.grad[1, 2] == 0.5
        assert adj.grad[2, 1] == 0.5
        # Edge (0,2) not in MST
        assert adj.grad[0, 2] == 0
        assert adj.grad[2, 0] == 0

    def test_gradient_symmetric(self):
        """Gradient is symmetric for undirected graph."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ],
            requires_grad=True,
        )

        weight, _ = minimum_spanning_tree(adj)
        weight.backward()

        # Gradient should be symmetric
        assert torch.allclose(adj.grad, adj.grad.T)


class TestMinimumSpanningTreeReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10, 15])
    def test_matches_scipy(self, N):
        """Compare to scipy.sparse.csgraph.minimum_spanning_tree."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random symmetric graph with ~50% edge density
        adj_np = numpy.random.rand(N, N) * 10
        adj_np = (adj_np + adj_np.T) / 2  # Symmetrize
        mask = numpy.random.rand(N, N) > 0.5
        mask = mask | mask.T  # Symmetric mask
        adj_np[mask] = numpy.inf
        numpy.fill_diagonal(adj_np, 0)

        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        weight_ours, edges_ours = minimum_spanning_tree(adj)

        # Scipy implementation
        # scipy uses 0 for missing edges, we use inf
        adj_scipy = numpy.where(numpy.isinf(adj_np), 0, adj_np)
        mst_scipy = scipy_sparse.csgraph.minimum_spanning_tree(adj_scipy)
        weight_scipy = mst_scipy.sum()

        # Compare weights (skip if graph is disconnected)
        if not numpy.isinf(weight_ours.item()):
            assert abs(weight_ours.item() - weight_scipy) < 1e-5, (
                f"Weight mismatch: ours={weight_ours.item()}, scipy={weight_scipy}"
            )


class TestMinimumSpanningTreeMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        weight, edges = minimum_spanning_tree(adj)

        assert weight.shape == ()
        assert edges.shape == (4, 2)
        assert weight.device.type == "meta"
        assert edges.device.type == "meta"

    def test_meta_single_node(self):
        """Meta tensor with single node."""
        adj = torch.rand(1, 1, device="meta")
        weight, edges = minimum_spanning_tree(adj)

        assert weight.shape == ()
        assert edges.shape == (0, 2)


class TestMinimumSpanningTreeEdgeCases:
    """Edge case tests."""

    def test_zero_weight_edges(self):
        """Graph with zero-weight edges."""
        adj = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        assert weight.item() == 0.0  # Two zero-weight edges

    def test_equal_weight_edges(self):
        """Graph where all edges have equal weight."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # MST has 2 edges, each with weight 1
        assert weight.item() == 2.0
        assert edges.shape == (2, 2)

    def test_negative_weights(self):
        """Graph with negative edge weights."""
        adj = torch.tensor(
            [
                [0.0, -1.0, 2.0],
                [-1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # MST uses edge (0,1) with weight -1 and edge (0,2) with weight 2
        assert weight.item() == 1.0  # -1 + 2

    def test_large_weights(self):
        """Graph with very large weights."""
        adj = torch.tensor(
            [
                [0.0, 1e10, 1e10],
                [1e10, 0.0, 1.0],
                [1e10, 1.0, 0.0],
            ]
        )
        weight, edges = minimum_spanning_tree(adj)

        # MST uses edges (1,2) with weight 1 and (0,1) with weight 1e10
        assert weight.item() == pytest.approx(1e10 + 1.0, rel=1e-9)
