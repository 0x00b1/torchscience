"""Tests for min_cost_max_flow operator."""

import pytest
import torch

from torchscience.graph import min_cost_max_flow


class TestMinCostMaxFlowBasic:
    """Basic functionality tests."""

    def test_simple_flow(self):
        """Simple 2-node flow."""
        capacity = torch.tensor(
            [
                [0.0, 5.0],
                [0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=1
        )

        assert max_flow.item() == 5.0
        assert total_cost.item() == 10.0  # 5 * 2 = 10
        assert flow[0, 1] == 5.0

    def test_cheaper_path_chosen(self):
        """Should choose cheaper path when capacities allow."""
        # Two paths: 0->1->2 (cost 3) or 0->2 (cost 10)
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0, 10.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )

        # Should use path 0->1->2 first (cost 3) until bottleneck, then 0->2
        assert max_flow.item() == 10.0  # Total capacity to sink
        # First 5 units via cheap path (cost 3*5=15), then 5 via expensive (cost 10*5=50)
        assert total_cost.item() == 65.0

    def test_no_path(self):
        """No path from source to sink."""
        capacity = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        max_flow, total_cost, _ = min_cost_max_flow(
            capacity, cost, source=0, sink=1
        )

        assert max_flow.item() == 0.0
        assert total_cost.item() == 0.0

    def test_single_edge(self):
        """Single edge with unit cost."""
        capacity = torch.tensor(
            [
                [0.0, 10.0],
                [0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=1
        )

        assert max_flow.item() == 10.0
        assert total_cost.item() == 10.0  # 10 * 1 = 10
        assert flow[0, 1] == 10.0

    def test_source_equals_sink(self):
        """Source equals sink - should have zero flow."""
        capacity = torch.tensor(
            [
                [0.0, 5.0],
                [0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=0
        )

        assert max_flow.item() == 0.0
        assert total_cost.item() == 0.0
        assert flow.sum().item() == 0.0


class TestMinCostMaxFlowComplex:
    """More complex test cases."""

    def test_diamond_graph_cost_selection(self):
        """Diamond graph where cost determines path selection."""
        #        1 (cheap)
        #       / \
        #      /   \
        #     0     3
        #      \   /
        #       \ /
        #        2 (expensive)
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0, 0.0],  # 0 -> 1, 0 -> 2
                [0.0, 0.0, 0.0, 5.0],  # 1 -> 3
                [0.0, 0.0, 0.0, 5.0],  # 2 -> 3
                [0.0, 0.0, 0.0, 0.0],  # 3 (sink)
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0, 10.0, 0.0],  # cheap to 1, expensive to 2
                [0.0, 0.0, 0.0, 1.0],  # cheap from 1
                [0.0, 0.0, 0.0, 10.0],  # expensive from 2
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=3
        )

        # Max flow is 10 (5 through each path)
        assert max_flow.item() == 10.0
        # Cheap path 0->1->3: 5 * (1+1) = 10
        # Expensive path 0->2->3: 5 * (10+10) = 100
        # Total: 110
        assert total_cost.item() == 110.0

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
        cost = torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=3
        )

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

    def test_zero_cost_flow(self):
        """Flow with zero costs should just be max flow."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cost = torch.zeros(3, 3)
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )

        assert max_flow.item() == 5.0
        assert total_cost.item() == 0.0


class TestMinCostMaxFlowValidation:
    """Input validation tests."""

    def test_rejects_1d_capacity(self):
        with pytest.raises(ValueError, match="2D"):
            min_cost_max_flow(
                torch.tensor([1.0]),
                torch.tensor([1.0]),
                source=0,
                sink=0,
            )

    def test_rejects_3d_capacity(self):
        with pytest.raises(ValueError, match="2D"):
            min_cost_max_flow(
                torch.rand(2, 3, 3),
                torch.rand(2, 3, 3),
                source=0,
                sink=1,
            )

    def test_rejects_non_square_capacity(self):
        with pytest.raises(ValueError, match="square"):
            min_cost_max_flow(
                torch.rand(3, 4),
                torch.rand(3, 4),
                source=0,
                sink=1,
            )

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            min_cost_max_flow(
                torch.rand(3, 3),
                torch.rand(4, 4),
                source=0,
                sink=1,
            )

    def test_rejects_negative_source(self):
        with pytest.raises(ValueError, match="source"):
            min_cost_max_flow(
                torch.rand(3, 3),
                torch.rand(3, 3),
                source=-1,
                sink=1,
            )

    def test_rejects_source_out_of_range(self):
        with pytest.raises(ValueError, match="source"):
            min_cost_max_flow(
                torch.rand(3, 3),
                torch.rand(3, 3),
                source=5,
                sink=1,
            )

    def test_rejects_negative_sink(self):
        with pytest.raises(ValueError, match="sink"):
            min_cost_max_flow(
                torch.rand(3, 3),
                torch.rand(3, 3),
                source=0,
                sink=-1,
            )

    def test_rejects_sink_out_of_range(self):
        with pytest.raises(ValueError, match="sink"):
            min_cost_max_flow(
                torch.rand(3, 3),
                torch.rand(3, 3),
                source=0,
                sink=5,
            )


class TestMinCostMaxFlowNetworkX:
    """Comparison tests with NetworkX."""

    def test_matches_networkx_simple(self):
        """Compare with NetworkX on simple graph."""
        networkx = pytest.importorskip("networkx")

        capacity = torch.tensor(
            [
                [0.0, 4.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0, 5.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        max_flow_ours, cost_ours, _ = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )

        # NetworkX
        G = networkx.DiGraph()
        N = 3
        for i in range(N):
            for j in range(N):
                if capacity[i, j] > 0:
                    G.add_edge(
                        i,
                        j,
                        capacity=capacity[i, j].item(),
                        weight=cost[i, j].item(),
                    )

        flow_dict = networkx.max_flow_min_cost(G, 0, 2)
        flow_cost = networkx.cost_of_flow(G, flow_dict)
        flow_value = sum(flow_dict[0][j] for j in flow_dict[0])

        assert abs(max_flow_ours.item() - flow_value) < 1e-5
        assert abs(cost_ours.item() - flow_cost) < 1e-5

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_matches_networkx_random(self, seed):
        """Compare with NetworkX on random graphs."""
        networkx = pytest.importorskip("networkx")

        torch.manual_seed(seed)
        N = 5

        # Create random capacity matrix with some zeros
        capacity = torch.rand(N, N) * 10
        capacity = capacity * (torch.rand(N, N) > 0.4).float()
        capacity.fill_diagonal_(0.0)

        # Random non-negative costs
        cost = torch.rand(N, N) * 5
        cost = cost * (capacity > 0).float()  # Only costs for existing edges

        # Ensure connectivity from source to sink
        capacity[0, 1] = 5.0
        cost[0, 1] = 1.0
        capacity[N - 2, N - 1] = 5.0
        cost[N - 2, N - 1] = 1.0

        max_flow_ours, cost_ours, _ = min_cost_max_flow(
            capacity, cost, source=0, sink=N - 1
        )

        # NetworkX
        G = networkx.DiGraph()
        for i in range(N):
            for j in range(N):
                if capacity[i, j] > 0:
                    G.add_edge(
                        i,
                        j,
                        capacity=capacity[i, j].item(),
                        weight=cost[i, j].item(),
                    )

        try:
            flow_dict = networkx.max_flow_min_cost(G, 0, N - 1)
            flow_cost = networkx.cost_of_flow(G, flow_dict)
            flow_value = sum(flow_dict[0].get(j, 0) for j in range(N))
        except networkx.NetworkXError:
            flow_value = 0.0
            flow_cost = 0.0

        assert max_flow_ours.item() == pytest.approx(flow_value, abs=1e-4)
        assert cost_ours.item() == pytest.approx(flow_cost, abs=1e-4)


class TestMinCostMaxFlowGradients:
    """Gradient tests."""

    def test_gradient_through_cost(self):
        """Gradient should flow through cost matrix."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cost = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )

        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )
        total_cost.backward()

        # Gradient of cost w.r.t. edge cost should be the flow on that edge
        # Edge (0,1) has flow 5, edge (1,2) has flow 5
        assert cost.grad[0, 1] == pytest.approx(5.0, abs=1e-5)
        assert cost.grad[1, 2] == pytest.approx(5.0, abs=1e-5)

    def test_gradient_through_capacity(self):
        """Gradient should flow through capacity matrix."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],  # Bottleneck
                [0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )
        cost = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )

        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )
        max_flow.backward()

        # Bottleneck edge should have gradient
        assert capacity.grad[1, 2] != 0.0


class TestMinCostMaxFlowMeta:
    """Meta tensor tests for shape inference."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        capacity = torch.rand(5, 5, device="meta")
        cost = torch.rand(5, 5, device="meta")
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=4
        )

        assert max_flow.shape == ()
        assert total_cost.shape == ()
        assert flow.shape == (5, 5)
        assert max_flow.device.type == "meta"
        assert total_cost.device.type == "meta"
        assert flow.device.type == "meta"


class TestMinCostMaxFlowDtypes:
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
        cost = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=dtype,
        )

        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=2
        )

        assert max_flow.dtype == dtype
        assert total_cost.dtype == dtype
        assert flow.dtype == dtype
        assert max_flow.item() == pytest.approx(3.0, rel=1e-2)
        # Cost = 3*1 + 3*2 = 9
        assert total_cost.item() == pytest.approx(9.0, rel=1e-2)


class TestMinCostMaxFlowCapacityConstraints:
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
        cost = torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        max_flow, total_cost, flow = min_cost_max_flow(
            capacity, cost, source=0, sink=3
        )

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
        cost = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )
        _, _, flow = min_cost_max_flow(capacity, cost, source=0, sink=2)

        # Check that zero-capacity edges have zero flow
        zero_capacity_mask = capacity == 0
        assert (flow[zero_capacity_mask].abs() < 1e-6).all()
