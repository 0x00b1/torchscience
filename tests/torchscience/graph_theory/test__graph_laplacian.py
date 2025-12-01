"""Tests for graph_laplacian."""

import pytest
import torch

from torchscience.graph_theory import graph_laplacian


class TestGraphLaplacianCombinatorial:
    """Tests for combinatorial (unnormalized) Laplacian."""

    def test_simple_triangle(self):
        """Triangle graph: each node has degree 2."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj)
        expected = torch.tensor(
            [
                [2.0, -1.0, -1.0],
                [-1.0, 2.0, -1.0],
                [-1.0, -1.0, 2.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_path_graph(self):
        """Path graph: 0 -- 1 -- 2."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj)
        expected = torch.tensor(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 1.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_weighted_graph(self):
        """Graph with weighted edges."""
        adj = torch.tensor(
            [
                [0.0, 2.0, 3.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        L = graph_laplacian(adj)
        # Degrees: [5, 2, 3]
        expected = torch.tensor(
            [
                [5.0, -2.0, -3.0],
                [-2.0, 2.0, 0.0],
                [-3.0, 0.0, 3.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_disconnected_graph(self):
        """Two disconnected components."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj)
        # Should have two zero eigenvalues (two components)
        eigvals = torch.linalg.eigvalsh(L)
        zero_count = (eigvals.abs() < 1e-5).sum().item()
        assert zero_count == 2

    def test_self_loops_included_in_degree(self):
        """Self-loops contribute to degree but cancel on diagonal of L."""
        adj = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj)
        # Degrees: [2, 1] (self-loop counts)
        # L_ii = degree_i - A_ii, so self-loop cancels: L_00 = 2 - 1 = 1
        expected = torch.tensor(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )
        torch.testing.assert_close(L, expected)


class TestGraphLaplacianSymmetric:
    """Tests for symmetric normalized Laplacian."""

    def test_simple_triangle(self):
        """Triangle graph with symmetric normalization."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj, normalization="symmetric")
        # For regular graph, L_sym = I - (1/d) * A
        # All degrees are 2, so L_sym = I - 0.5 * A
        expected = torch.tensor(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, -0.5],
                [-0.5, -0.5, 1.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_eigenvalues_in_0_2(self):
        """Symmetric Laplacian eigenvalues should be in [0, 2]."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T  # Symmetrize
        adj.fill_diagonal_(0)  # No self-loops
        L = graph_laplacian(adj, normalization="symmetric")
        eigvals = torch.linalg.eigvalsh(L)
        assert eigvals.min() >= -1e-5  # Allow numerical error
        assert eigvals.max() <= 2.0 + 1e-5

    def test_isolated_node_handled(self):
        """Isolated node (degree 0) should not cause NaN."""
        adj = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj, normalization="symmetric")
        assert not L.isnan().any()
        # Isolated node row/col should be like identity
        assert L[0, 0] == 1.0
        assert L[0, 1] == 0.0
        assert L[0, 2] == 0.0


class TestGraphLaplacianRandomWalk:
    """Tests for random walk normalized Laplacian."""

    def test_simple_triangle(self):
        """Triangle graph with random walk normalization."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj, normalization="random_walk")
        # L_rw = I - D^{-1} A
        # All degrees are 2
        expected = torch.tensor(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, -0.5],
                [-0.5, -0.5, 1.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_irregular_graph(self):
        """Graph with different degrees."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        L = graph_laplacian(adj, normalization="random_walk")
        # Node 0 has degree 3, others have degree 1
        # Row 0: [1, -1/3, -1/3, -1/3]
        # Row 1: [-1, 1, 0, 0]
        expected = torch.tensor(
            [
                [1.0, -1 / 3, -1 / 3, -1 / 3],
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(L, expected)

    def test_isolated_node_handled(self):
        """Isolated node should not cause NaN."""
        adj = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        L = graph_laplacian(adj, normalization="random_walk")
        assert not L.isnan().any()


class TestGraphLaplacianGradients:
    """Tests for autograd support."""

    def test_combinatorial_gradcheck(self):
        """Gradcheck for combinatorial Laplacian."""
        adj = torch.rand(4, 4, dtype=torch.float64, requires_grad=True)
        adj = adj + adj.T  # Symmetrize

        def func(x):
            return graph_laplacian(x)

        assert torch.autograd.gradcheck(func, adj)

    def test_symmetric_gradcheck(self):
        """Gradcheck for symmetric normalized Laplacian."""
        # Ensure no isolated nodes (would cause gradient issues)
        adj = torch.rand(4, 4, dtype=torch.float64) + 0.1
        adj = adj + adj.T
        adj.requires_grad_(True)

        def func(x):
            return graph_laplacian(x, normalization="symmetric")

        assert torch.autograd.gradcheck(func, adj)

    def test_random_walk_gradcheck(self):
        """Gradcheck for random walk normalized Laplacian."""
        adj = torch.rand(4, 4, dtype=torch.float64) + 0.1
        adj = adj + adj.T
        adj.requires_grad_(True)

        def func(x):
            return graph_laplacian(x, normalization="random_walk")

        assert torch.autograd.gradcheck(func, adj)


class TestGraphLaplacianBatched:
    """Tests for batched computation."""

    def test_batch_2d(self):
        """Batched with 2D batch dimension."""
        adj = torch.rand(3, 5, 5)
        adj = adj + adj.transpose(-1, -2)  # Symmetrize
        L = graph_laplacian(adj)
        assert L.shape == (3, 5, 5)

    def test_batch_consistency(self):
        """Batched result matches individual computations."""
        adj1 = torch.rand(4, 4)
        adj1 = adj1 + adj1.T
        adj2 = torch.rand(4, 4)
        adj2 = adj2 + adj2.T

        L1 = graph_laplacian(adj1)
        L2 = graph_laplacian(adj2)

        batch_adj = torch.stack([adj1, adj2])
        L_batch = graph_laplacian(batch_adj)

        torch.testing.assert_close(L_batch[0], L1)
        torch.testing.assert_close(L_batch[1], L2)


class TestGraphLaplacianDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        adj = torch.rand(4, 4, dtype=dtype)
        adj = adj + adj.T
        L = graph_laplacian(adj)
        assert L.dtype == dtype

    def test_complex_dtype(self):
        """Complex adjacency matrices should work."""
        adj = torch.rand(4, 4, dtype=torch.complex64)
        adj = adj + adj.T.conj()  # Hermitian
        L = graph_laplacian(adj)
        assert L.dtype == torch.complex64


class TestGraphLaplacianValidation:
    """Tests for input validation."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            graph_laplacian(torch.tensor([1.0, 2.0, 3.0]))

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            graph_laplacian(torch.rand(3, 4))

    def test_rejects_invalid_normalization(self):
        """Should reject invalid normalization type."""
        with pytest.raises(ValueError, match="normalization"):
            graph_laplacian(torch.rand(3, 3), normalization="invalid")


class TestGraphLaplacianReference:
    """Tests comparing to scipy reference."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy_combinatorial(self, N):
        """Compare combinatorial Laplacian to scipy."""
        scipy_sparse = pytest.importorskip("scipy.sparse")

        # Random symmetric adjacency
        adj_np = torch.rand(N, N).numpy()
        adj_np = adj_np + adj_np.T
        adj_np[range(N), range(N)] = 0  # No self-loops

        adj = torch.from_numpy(adj_np)

        # Our implementation
        L_ours = graph_laplacian(adj)

        # Scipy implementation
        L_scipy = scipy_sparse.csgraph.laplacian(adj_np, normed=False)

        torch.testing.assert_close(
            L_ours, torch.from_numpy(L_scipy), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy_symmetric(self, N):
        """Compare symmetric normalized Laplacian to scipy."""
        scipy_sparse = pytest.importorskip("scipy.sparse")

        # Random symmetric adjacency with no isolated nodes
        adj_np = torch.rand(N, N).numpy() + 0.1
        adj_np = adj_np + adj_np.T
        adj_np[range(N), range(N)] = 0

        adj = torch.from_numpy(adj_np)

        # Our implementation
        L_ours = graph_laplacian(adj, normalization="symmetric")

        # Scipy implementation (normed=True gives symmetric normalized)
        L_scipy = scipy_sparse.csgraph.laplacian(adj_np, normed=True)

        torch.testing.assert_close(
            L_ours, torch.from_numpy(L_scipy), rtol=1e-5, atol=1e-5
        )
