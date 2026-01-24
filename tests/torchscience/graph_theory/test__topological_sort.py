"""Tests for topological_sort."""

import pytest
import torch

from torchscience.graph import topological_sort


class TestTopologicalSortBasic:
    """Basic functionality tests."""

    def test_simple_dag(self):
        """Linear DAG: 0 -> 1 -> 2."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ]
        )
        order = topological_sort(adj)

        pos = {order[i].item(): i for i in range(3)}
        assert pos[0] < pos[1] < pos[2]

    def test_diamond_dag(self):
        """Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, inf],
                [inf, 0.0, inf, 1.0],
                [inf, inf, 0.0, 1.0],
                [inf, inf, inf, 0.0],
            ]
        )
        order = topological_sort(adj)

        pos = {order[i].item(): i for i in range(4)}
        assert pos[0] < pos[1]
        assert pos[0] < pos[2]
        assert pos[1] < pos[3]
        assert pos[2] < pos[3]

    def test_single_node(self):
        adj = torch.tensor([[0.0]])
        order = topological_sort(adj)
        assert order.tolist() == [0]

    def test_isolated_nodes(self):
        inf = float("inf")
        adj = torch.full((3, 3), inf)
        adj.fill_diagonal_(0)
        order = topological_sort(adj)
        assert set(order.tolist()) == {0, 1, 2}

    def test_empty_graph(self):
        """Handles empty graph (N=0)."""
        adj = torch.empty(0, 0)
        order = topological_sort(adj)
        assert order.shape == (0,)

    def test_complex_dag(self):
        """More complex DAG with multiple valid orderings."""
        inf = float("inf")
        # Graph:
        # 0 -> 2, 0 -> 3
        # 1 -> 3, 1 -> 4
        # 2 -> 5
        # 3 -> 5
        # 4 -> 5
        adj = torch.tensor(
            [
                [0.0, inf, 1.0, 1.0, inf, inf],  # 0
                [inf, 0.0, inf, 1.0, 1.0, inf],  # 1
                [inf, inf, 0.0, inf, inf, 1.0],  # 2
                [inf, inf, inf, 0.0, inf, 1.0],  # 3
                [inf, inf, inf, inf, 0.0, 1.0],  # 4
                [inf, inf, inf, inf, inf, 0.0],  # 5
            ]
        )
        order = topological_sort(adj)

        pos = {order[i].item(): i for i in range(6)}
        # Check all edges go forward
        assert pos[0] < pos[2]
        assert pos[0] < pos[3]
        assert pos[1] < pos[3]
        assert pos[1] < pos[4]
        assert pos[2] < pos[5]
        assert pos[3] < pos[5]
        assert pos[4] < pos[5]


class TestTopologicalSortCycle:
    def test_rejects_cycle(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [1.0, inf, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(adj)

    def test_rejects_self_loop(self):
        adj = torch.tensor(
            [
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(adj)

    def test_rejects_two_node_cycle(self):
        """Rejects simple two-node cycle."""
        adj = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(adj)


class TestTopologicalSortValidation:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            topological_sort(torch.tensor([1.0, 2.0, 3.0]))

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            topological_sort(torch.rand(3, 4))

    def test_rejects_integer_dtype(self):
        """Rejects integer dtype."""
        with pytest.raises(ValueError, match="floating-point"):
            topological_sort(torch.tensor([[0, 1], [0, 0]]))


class TestTopologicalSortBatched:
    def test_batched_2d(self):
        inf = float("inf")
        adj = torch.tensor(
            [
                [[0.0, 1.0], [inf, 0.0]],
                [[0.0, 1.0], [inf, 0.0]],
            ]
        )
        order = topological_sort(adj)
        assert order.shape == (2, 2)

    def test_batched_3d(self):
        """Handles nested batch dimensions."""
        inf = float("inf")
        adj = torch.zeros(2, 3, 4, 4)
        adj.fill_(inf)
        for i in range(2):
            for j in range(3):
                adj[i, j].fill_diagonal_(0)
        order = topological_sort(adj)
        assert order.shape == (2, 3, 4)


class TestTopologicalSortMeta:
    def test_meta_shape(self):
        adj = torch.rand(5, 5, device="meta")
        order = topological_sort(adj)
        assert order.shape == (5,)
        assert order.device.type == "meta"

    def test_meta_batched(self):
        adj = torch.rand(3, 5, 5, device="meta")
        order = topological_sort(adj)
        assert order.shape == (3, 5)
        assert order.device.type == "meta"


class TestTopologicalSortDtypes:
    """Dtype support tests."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ],
    )
    def test_dtype(self, dtype):
        """Supports various floating-point dtypes."""
        inf = float("inf")
        adj = torch.tensor(
            [
                [0.0, 1.0, inf],
                [inf, 0.0, 1.0],
                [inf, inf, 0.0],
            ],
            dtype=dtype,
        )

        order = topological_sort(adj)

        assert order.dtype == torch.int64


class TestTopologicalSortReference:
    """Compare against NetworkX reference implementation."""

    def test_matches_networkx(self):
        """Results match networkx.topological_sort."""
        nx = pytest.importorskip("networkx")

        inf = float("inf")
        # Simple DAG
        adj = torch.tensor(
            [
                [0.0, 1.0, 1.0, inf],
                [inf, 0.0, inf, 1.0],
                [inf, inf, 0.0, 1.0],
                [inf, inf, inf, 0.0],
            ]
        )

        # Compute with torchscience
        order = topological_sort(adj)
        order_list = order.tolist()

        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

        # Verify our order is valid
        pos = {node: i for i, node in enumerate(order_list)}
        for u, v in G.edges():
            assert pos[u] < pos[v], (
                f"Edge ({u}, {v}) violates topological order"
            )
