"""Tests for connected_components."""

import pytest
import torch

from torchscience.graph import connected_components


class TestConnectedComponentsWeak:
    """Tests for weak connectivity."""

    def test_single_component(self):
        """Fully connected graph has one component."""
        adj = torch.ones(4, 4)
        n, labels = connected_components(adj, connection="weak")
        assert n == 1
        assert (labels == 0).all()

    def test_two_components(self):
        """Two disconnected edges form two components."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        n, labels = connected_components(adj, connection="weak")
        assert n == 2
        # Nodes 0,1 should have same label, nodes 2,3 should have same label
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_isolated_nodes(self):
        """Each isolated node is its own component."""
        adj = torch.zeros(3, 3)
        n, labels = connected_components(adj, connection="weak")
        assert n == 3
        # All labels should be different
        assert len(labels.unique()) == 3

    def test_single_node(self):
        """Single node graph."""
        adj = torch.tensor([[0.0]])
        n, labels = connected_components(adj, connection="weak")
        assert n == 1
        assert labels.shape == (1,)
        assert labels[0] == 0

    def test_empty_graph(self):
        """Empty graph (0 nodes)."""
        adj = torch.zeros(0, 0)
        n, labels = connected_components(adj, connection="weak")
        assert n == 0
        assert labels.shape == (0,)

    def test_chain_graph(self):
        """Chain graph: 0-1-2-3."""
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        n, labels = connected_components(adj, connection="weak")
        assert n == 1
        assert (labels == labels[0]).all()

    def test_directed_treated_as_undirected(self):
        """Weak connectivity ignores edge direction."""
        # Chain: 0 -> 1 -> 2
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        n, labels = connected_components(adj, directed=True, connection="weak")
        assert n == 1  # All connected when ignoring direction

    def test_weighted_edges(self):
        """Weighted edges are treated as present."""
        adj = torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 2.0],
                [0.0, 2.0, 0.0],
            ]
        )
        n, labels = connected_components(adj, connection="weak")
        assert n == 1


class TestConnectedComponentsStrong:
    """Tests for strong connectivity (directed graphs)."""

    def test_cycle_is_one_scc(self):
        """Directed cycle is one strongly connected component."""
        # 0 -> 1 -> 2 -> 0
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        n, labels = connected_components(
            adj, directed=True, connection="strong"
        )
        assert n == 1
        assert (labels == labels[0]).all()

    def test_chain_is_many_sccs(self):
        """Directed chain has each node as its own SCC."""
        # 0 -> 1 -> 2
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        n, labels = connected_components(
            adj, directed=True, connection="strong"
        )
        assert n == 3
        assert len(labels.unique()) == 3

    def test_two_sccs_with_bridge(self):
        """Two cycles connected by one-way bridge."""
        # SCC1: 0 <-> 1, SCC2: 2 <-> 3
        # Bridge: 1 -> 2 (one way)
        adj = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        n, labels = connected_components(
            adj, directed=True, connection="strong"
        )
        assert n == 2
        # Nodes 0,1 should be in same SCC
        assert labels[0] == labels[1]
        # Nodes 2,3 should be in same SCC
        assert labels[2] == labels[3]
        # Different SCCs
        assert labels[0] != labels[2]

    def test_self_loop(self):
        """Self-loop doesn't affect SCC count."""
        adj = torch.tensor(
            [
                [1.0, 0.0],  # Node 0 has self-loop
                [0.0, 0.0],
            ]
        )
        n, labels = connected_components(
            adj, directed=True, connection="strong"
        )
        assert n == 2  # Each node is its own SCC

    def test_complete_directed_graph(self):
        """Complete directed graph is one SCC."""
        adj = torch.ones(4, 4)
        n, labels = connected_components(
            adj, directed=True, connection="strong"
        )
        assert n == 1


class TestConnectedComponentsDtypes:
    """Tests for different dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        """Test different floating point types."""
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        n, labels = connected_components(adj, connection="weak")
        assert n == 1
        assert labels.dtype == torch.int64


class TestConnectedComponentsSparse:
    """Tests for sparse input."""

    def test_sparse_input(self):
        """Sparse COO tensor input."""
        indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        values = torch.tensor([1.0, 1.0, 1.0, 1.0])
        adj = torch.sparse_coo_tensor(indices, values, (3, 3))

        n, labels = connected_components(adj, connection="weak")
        assert n == 1


class TestConnectedComponentsValidation:
    """Tests for input validation."""

    def test_rejects_1d_input(self):
        """Should reject 1D input."""
        with pytest.raises(ValueError, match="at least 2D"):
            connected_components(torch.tensor([1.0, 2.0, 3.0]))

    def test_rejects_non_square(self):
        """Should reject non-square adjacency."""
        with pytest.raises(ValueError, match="square"):
            connected_components(torch.rand(3, 4))

    def test_rejects_invalid_connection(self):
        """Should reject invalid connection type."""
        with pytest.raises(ValueError, match="connection"):
            connected_components(torch.rand(3, 3), connection="invalid")

    def test_rejects_strong_with_undirected(self):
        """Should reject strong connectivity for undirected graphs."""
        with pytest.raises(ValueError, match="directed"):
            connected_components(
                torch.rand(3, 3), directed=False, connection="strong"
            )


class TestConnectedComponentsReference:
    """Tests comparing to reference implementations."""

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy_weak(self, N):
        """Compare weak connectivity to scipy."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random graph with ~50% edge density
        adj_np = (numpy.random.rand(N, N) > 0.5).astype(float)
        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        n_ours, labels_ours = connected_components(
            adj, directed=False, connection="weak"
        )

        # Scipy implementation
        n_scipy, labels_scipy = scipy_sparse.csgraph.connected_components(
            adj_np, directed=False, connection="weak"
        )

        assert n_ours == n_scipy

        # Labels may be in different order, but groupings should match
        for i in range(N):
            for j in range(N):
                same_ours = labels_ours[i] == labels_ours[j]
                same_scipy = labels_scipy[i] == labels_scipy[j]
                assert same_ours == same_scipy, f"Mismatch at ({i}, {j})"

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_matches_scipy_strong(self, N):
        """Compare strong connectivity to scipy."""
        scipy_sparse = pytest.importorskip("scipy.sparse")
        numpy = pytest.importorskip("numpy")

        # Random directed graph with ~30% edge density
        adj_np = (numpy.random.rand(N, N) > 0.7).astype(float)
        adj = torch.from_numpy(adj_np).float()

        # Our implementation
        n_ours, labels_ours = connected_components(
            adj, directed=True, connection="strong"
        )

        # Scipy implementation
        n_scipy, labels_scipy = scipy_sparse.csgraph.connected_components(
            adj_np, directed=True, connection="strong"
        )

        assert n_ours == n_scipy

        # Labels may be in different order, but groupings should match
        for i in range(N):
            for j in range(N):
                same_ours = labels_ours[i] == labels_ours[j]
                same_scipy = labels_scipy[i] == labels_scipy[j]
                assert same_ours == same_scipy, f"Mismatch at ({i}, {j})"


class TestConnectedComponentsMeta:
    """Tests for meta tensor support."""

    def test_meta_shape(self):
        """Meta tensor returns correct shape."""
        adj = torch.rand(5, 5, device="meta")
        n, labels = connected_components(adj, connection="weak")
        assert labels.shape == (5,)
        assert labels.device.type == "meta"
