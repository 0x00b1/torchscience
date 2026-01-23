"""Tests for push_relabel maximum flow operator."""

import pytest
import torch

from torchscience.graph import push_relabel


class TestPushRelabelBasic:
    """Basic functionality tests."""

    def test_simple_flow(self):
        """Simple 2-node flow."""
        capacity = torch.tensor(
            [
                [0.0, 5.0],
                [0.0, 0.0],
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=1)
        assert max_flow.item() == 5.0
        assert flow[0, 1] == 5.0

    def test_bottleneck(self):
        """Flow limited by bottleneck edge."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],  # Bottleneck
                [0.0, 0.0, 0.0],
            ]
        )
        max_flow, _ = push_relabel(capacity, source=0, sink=2)
        assert max_flow.item() == 3.0

    def test_parallel_paths(self):
        """Multiple augmenting paths."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, _ = push_relabel(capacity, source=0, sink=3)
        assert max_flow.item() == 10.0

    def test_no_path(self):
        """No path from source to sink."""
        capacity = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=1)
        assert max_flow.item() == 0.0
        assert flow.sum().item() == 0.0

    def test_single_node(self):
        """Single node graph - source equals sink."""
        capacity = torch.tensor([[0.0]])
        # For single node, max flow is infinite if source == sink
        # But since source != sink constraint, should handle gracefully
        max_flow, flow = push_relabel(capacity, source=0, sink=0)
        assert flow.sum().item() == 0.0

    def test_matches_edmonds_karp(self):
        """Should give same result as edmonds_karp."""
        from torchscience.graph import edmonds_karp

        capacity = torch.tensor(
            [
                [0.0, 16.0, 13.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 12.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 14.0, 0.0],
                [0.0, 0.0, 9.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 0.0, 7.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        max_flow_pr, _ = push_relabel(capacity, source=0, sink=5)
        max_flow_ek, _ = edmonds_karp(capacity, source=0, sink=5)

        assert abs(max_flow_pr.item() - max_flow_ek.item()) < 1e-5


class TestPushRelabelComplex:
    """More complex test cases."""

    def test_diamond_graph(self):
        """Diamond shape with multiple paths."""
        #      1
        #     /|\
        #    / | \
        #   3  |  4
        #    \ | /
        #     \|/
        #      2
        # Node 0 = source, Node 3 = sink
        capacity = torch.tensor(
            [
                [0.0, 3.0, 2.0, 0.0],  # source -> 1: 3, source -> 2: 2
                [0.0, 0.0, 0.0, 2.0],  # 1 -> sink: 2
                [0.0, 0.0, 0.0, 3.0],  # 2 -> sink: 3
                [0.0, 0.0, 0.0, 0.0],  # sink
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=3)
        # Max flow is min(3+2, 2+3) = 5, but bottleneck at 1->sink is 2
        # Path 0->1->3: capacity 2
        # Path 0->2->3: capacity 2 (limited by 0->2)
        # Total: 2 + 2 = 4
        assert max_flow.item() == pytest.approx(4.0)

    def test_reverse_edges(self):
        """Flow network with reverse edges (bidirectional capacity)."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [2.0, 0.0, 5.0],  # Reverse edge 1->0 with capacity 2
                [0.0, 0.0, 0.0],
            ]
        )
        max_flow, _ = push_relabel(capacity, source=0, sink=2)
        # Forward path: 0->1->2 with capacity 5
        # The reverse edge 1->0 doesn't help for 0->2 flow
        assert max_flow.item() == 5.0

    def test_multiple_augmenting_paths(self):
        """Classic max-flow example with multiple augmenting paths."""
        # Ford-Fulkerson example graph
        capacity = torch.tensor(
            [
                [0.0, 16.0, 13.0, 0.0, 0.0, 0.0],  # source
                [0.0, 0.0, 10.0, 12.0, 0.0, 0.0],  # node 1
                [0.0, 4.0, 0.0, 0.0, 14.0, 0.0],  # node 2
                [0.0, 0.0, 9.0, 0.0, 0.0, 20.0],  # node 3
                [0.0, 0.0, 0.0, 7.0, 0.0, 4.0],  # node 4
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sink
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=5)
        # Known max flow for this graph is 23
        assert max_flow.item() == pytest.approx(23.0)

        # Verify flow conservation
        for i in range(1, 5):  # Intermediate nodes
            inflow = flow[:, i].sum()
            outflow = flow[i, :].sum()
            assert inflow.item() == pytest.approx(outflow.item(), abs=1e-5)

    def test_flow_conservation(self):
        """Test that flow conservation holds at intermediate nodes."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 2.0, 6.0],
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=3)

        # Check flow conservation at intermediate nodes (1 and 2)
        for i in [1, 2]:
            inflow = flow[:, i].sum()
            outflow = flow[i, :].sum()
            assert inflow.item() == pytest.approx(outflow.item(), abs=1e-5)

        # Check that total outflow from source equals max_flow
        assert flow[0, :].sum().item() == pytest.approx(
            max_flow.item(), abs=1e-5
        )

        # Check that total inflow to sink equals max_flow
        assert flow[:, 3].sum().item() == pytest.approx(
            max_flow.item(), abs=1e-5
        )


class TestPushRelabelValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            push_relabel(torch.tensor([1.0]), source=0, sink=0)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2D"):
            push_relabel(torch.rand(2, 3, 3), source=0, sink=1)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            push_relabel(torch.rand(3, 4), source=0, sink=1)

    def test_rejects_negative_source(self):
        with pytest.raises(ValueError, match="source"):
            push_relabel(torch.rand(3, 3), source=-1, sink=1)

    def test_rejects_source_out_of_range(self):
        with pytest.raises(ValueError, match="source"):
            push_relabel(torch.rand(3, 3), source=5, sink=1)

    def test_rejects_negative_sink(self):
        with pytest.raises(ValueError, match="sink"):
            push_relabel(torch.rand(3, 3), source=0, sink=-1)

    def test_rejects_sink_out_of_range(self):
        with pytest.raises(ValueError, match="sink"):
            push_relabel(torch.rand(3, 3), source=0, sink=5)


class TestPushRelabelNetworkX:
    """Comparison tests with NetworkX."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_matches_networkx_random(self, seed):
        """Compare with NetworkX on random graphs."""
        pytest.importorskip("networkx")
        import networkx as nx

        torch.manual_seed(seed)
        N = 6

        # Create random capacity matrix with some zeros
        capacity = torch.rand(N, N) * 10
        capacity = capacity * (torch.rand(N, N) > 0.3).float()
        capacity.fill_diagonal_(0.0)

        # Ensure connectivity from source to sink
        capacity[0, 1] = 5.0  # source has outgoing edge
        capacity[N - 2, N - 1] = 5.0  # sink has incoming edge

        # Create NetworkX graph
        G = nx.DiGraph()
        for i in range(N):
            for j in range(N):
                if capacity[i, j] > 0:
                    G.add_edge(i, j, capacity=capacity[i, j].item())

        # Compute max flow with both
        max_flow_ts, _ = push_relabel(capacity, source=0, sink=N - 1)

        try:
            flow_value_nx, _ = nx.maximum_flow(G, 0, N - 1)
        except nx.NetworkXError:
            # No path exists
            flow_value_nx = 0.0

        assert max_flow_ts.item() == pytest.approx(flow_value_nx, abs=1e-4)


class TestPushRelabelGradients:
    """Gradient tests (if autograd is supported)."""

    def test_gradient_through_bottleneck(self):
        """Gradient should flow through bottleneck edge."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],  # Bottleneck
                [0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )

        max_flow, _ = push_relabel(capacity, source=0, sink=2)
        max_flow.backward()

        # Gradient should be non-zero for bottleneck edge
        assert capacity.grad[1, 2] != 0.0

    def test_gradient_parallel_paths(self):
        """Test gradient with parallel paths."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        max_flow, _ = push_relabel(capacity, source=0, sink=3)
        max_flow.backward()

        # Both paths should contribute to gradient
        # The min-cut edges should have non-zero gradients
        assert capacity.grad is not None


class TestPushRelabelMeta:
    """Meta tensor tests for shape inference."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        capacity = torch.rand(5, 5, device="meta")
        max_flow, flow = push_relabel(capacity, source=0, sink=4)

        assert max_flow.shape == ()
        assert flow.shape == (5, 5)
        assert max_flow.device.type == "meta"
        assert flow.device.type == "meta"


class TestPushRelabelDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.half, torch.bfloat16]
    )
    def test_dtype(self, dtype):
        """Test different floating point types."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=dtype,
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=2)

        assert max_flow.dtype == dtype
        assert flow.dtype == dtype
        assert max_flow.item() == pytest.approx(3.0, rel=1e-2)


class TestPushRelabelCapacityConstraints:
    """Test that flow respects capacity constraints."""

    def test_flow_respects_capacity(self):
        """Flow should never exceed capacity on any edge."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 2.0, 6.0],
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, flow = push_relabel(capacity, source=0, sink=3)

        # Flow should be <= capacity everywhere
        assert (flow <= capacity + 1e-6).all()

        # Flow should be >= 0 everywhere
        assert (flow >= -1e-6).all()

    def test_zero_capacity_zero_flow(self):
        """Edges with zero capacity should have zero flow."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        _, flow = push_relabel(capacity, source=0, sink=2)

        # Check that zero-capacity edges have zero flow
        zero_capacity_mask = capacity == 0
        assert (flow[zero_capacity_mask].abs() < 1e-6).all()
