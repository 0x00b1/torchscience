"""Tests for minimum_cut operator."""

import pytest
import torch

from torchscience.graph import minimum_cut


class TestMinimumCutBasic:
    """Basic functionality tests."""

    def test_simple_cut(self):
        """Simple 2-node cut."""
        capacity = torch.tensor(
            [
                [0.0, 5.0],
                [0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=1
        )

        assert cut_value.item() == 5.0
        assert reachable[0] == True
        assert reachable[1] == False
        assert cut_edges.shape[0] == 1
        assert cut_edges[0].tolist() == [0, 1]

    def test_bottleneck_cut(self):
        """Cut at bottleneck edge."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=2
        )
        assert cut_value.item() == 3.0
        # Reachable should include 0 and 1
        assert reachable[0] == True
        assert reachable[1] == True
        assert reachable[2] == False
        # Cut edge is (1, 2)
        assert cut_edges.shape[0] == 1
        assert cut_edges[0].tolist() == [1, 2]

    def test_parallel_paths_cut(self):
        """Multiple edges in cut."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=3
        )
        assert cut_value.item() == 10.0
        # Reachable includes source and intermediate nodes
        assert reachable[0] == True
        assert reachable[3] == False  # Sink not reachable

    def test_no_path(self):
        """No path from source to sink means zero cut value."""
        capacity = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=1
        )
        assert cut_value.item() == 0.0
        assert reachable[0] == True
        assert reachable[1] == False
        assert cut_edges.shape[0] == 0

    def test_source_equals_sink(self):
        """Source equals sink - edge case."""
        capacity = torch.tensor([[0.0]])
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=0
        )
        assert cut_value.item() == 0.0
        assert reachable[0] == True
        assert cut_edges.shape[0] == 0


class TestMinimumCutComplex:
    """More complex test cases."""

    def test_diamond_graph(self):
        """Diamond shape with multiple paths."""
        capacity = torch.tensor(
            [
                [0.0, 3.0, 2.0, 0.0],  # source -> 1: 3, source -> 2: 2
                [0.0, 0.0, 0.0, 2.0],  # 1 -> sink: 2
                [0.0, 0.0, 0.0, 3.0],  # 2 -> sink: 3
                [0.0, 0.0, 0.0, 0.0],  # sink
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=3
        )
        # Max flow = 4 (path 0->1->3: 2, path 0->2->3: 2)
        assert cut_value.item() == pytest.approx(4.0)

    def test_classic_example(self):
        """Classic Ford-Fulkerson example graph."""
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
        cut_value, _, _ = minimum_cut(capacity, source=0, sink=5)
        # Known max flow for this graph is 23
        assert cut_value.item() == pytest.approx(23.0)

    def test_cut_edges_sum_to_cut_value(self):
        """Sum of capacities of cut edges equals cut value."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 2.0, 6.0],
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=3
        )

        # Sum of capacities of cut edges should equal cut value
        total_cut_capacity = 0.0
        for i in range(cut_edges.shape[0]):
            u, v = cut_edges[i].tolist()
            total_cut_capacity += capacity[u, v].item()

        assert total_cut_capacity == pytest.approx(cut_value.item(), abs=1e-5)


class TestMinimumCutValidation:
    """Input validation tests."""

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            minimum_cut(torch.tensor([1.0]), source=0, sink=0)

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2D"):
            minimum_cut(torch.rand(2, 3, 3), source=0, sink=1)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            minimum_cut(torch.rand(3, 4), source=0, sink=1)

    def test_rejects_negative_source(self):
        with pytest.raises(ValueError, match="source"):
            minimum_cut(torch.rand(3, 3), source=-1, sink=1)

    def test_rejects_source_out_of_range(self):
        with pytest.raises(ValueError, match="source"):
            minimum_cut(torch.rand(3, 3), source=5, sink=1)

    def test_rejects_negative_sink(self):
        with pytest.raises(ValueError, match="sink"):
            minimum_cut(torch.rand(3, 3), source=0, sink=-1)

    def test_rejects_sink_out_of_range(self):
        with pytest.raises(ValueError, match="sink"):
            minimum_cut(torch.rand(3, 3), source=0, sink=5)


class TestMinimumCutNetworkX:
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

        # Compute minimum cut with both
        cut_value_ts, _, _ = minimum_cut(capacity, source=0, sink=N - 1)

        try:
            cut_value_nx, _ = nx.minimum_cut(G, 0, N - 1)
        except nx.NetworkXError:
            # No path exists
            cut_value_nx = 0.0

        assert cut_value_ts.item() == pytest.approx(cut_value_nx, abs=1e-4)

    def test_matches_networkx_cut_partition(self):
        """Verify cut partition is valid (source reachable, sink unreachable)."""
        pytest.importorskip("networkx")
        import networkx as nx

        # Use a graph where the minimum cut is unique
        # Bottleneck graph: 0 --10--> 1 --3--> 2
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 0.0],
            ]
        )

        # Create NetworkX graph
        G = nx.DiGraph()
        N = capacity.shape[0]
        for i in range(N):
            for j in range(N):
                if capacity[i, j] > 0:
                    G.add_edge(i, j, capacity=capacity[i, j].item())

        cut_value_ts, reachable_ts, _ = minimum_cut(capacity, source=0, sink=2)
        cut_value_nx, (reachable_nx, unreachable_nx) = nx.minimum_cut(G, 0, 2)

        assert cut_value_ts.item() == pytest.approx(cut_value_nx, abs=1e-4)

        # Check partition - for unique minimum cut, should match NetworkX
        reachable_set = {i for i in range(N) if reachable_ts[i].item()}
        assert reachable_set == reachable_nx

    def test_partition_validity(self):
        """Verify partition is always valid (source in S, sink in T)."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=3
        )

        # Source should always be reachable
        assert reachable[0] == True
        # Sink should never be reachable
        assert reachable[3] == False
        # Cut value should be correct
        assert cut_value.item() == pytest.approx(10.0)


class TestMinimumCutGradients:
    """Gradient tests."""

    def test_gradient_on_cut_edges(self):
        """Gradient is 1 on cut edges, 0 elsewhere."""
        capacity = torch.tensor(
            [
                [0.0, 5.0],
                [0.0, 0.0],
            ],
            requires_grad=True,
        )
        cut_value, _, _ = minimum_cut(capacity, source=0, sink=1)
        cut_value.backward()

        assert capacity.grad[0, 1] == 1.0  # Cut edge
        assert capacity.grad[1, 0] == 0.0  # Not a cut edge

    def test_gradient_bottleneck(self):
        """Gradient should be 1 for bottleneck edge."""
        capacity = torch.tensor(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 3.0],  # Bottleneck
                [0.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )

        cut_value, _, _ = minimum_cut(capacity, source=0, sink=2)
        cut_value.backward()

        # Gradient should be 1 for bottleneck edge
        assert capacity.grad[1, 2] == 1.0

    def test_gradient_parallel_paths(self):
        """Test gradient with parallel paths - cut edges at sink."""
        capacity = torch.tensor(
            [
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        cut_value, _, _ = minimum_cut(capacity, source=0, sink=3)
        cut_value.backward()

        # Edges from source to intermediates are the min-cut
        # Gradient should be 1 for (0, 1) and (0, 2)
        assert capacity.grad[0, 1] == 1.0
        assert capacity.grad[0, 2] == 1.0


class TestMinimumCutMeta:
    """Meta tensor tests for shape inference."""

    def test_meta_shape(self):
        """Test shape inference with meta tensors."""
        capacity = torch.rand(5, 5, device="meta")
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=4
        )

        assert cut_value.shape == ()
        assert reachable.shape == (5,)
        assert cut_edges.dim() == 2
        assert cut_edges.size(1) == 2
        assert cut_value.device.type == "meta"
        assert reachable.device.type == "meta"
        assert cut_edges.device.type == "meta"


class TestMinimumCutDtypes:
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
        cut_value, reachable, cut_edges = minimum_cut(
            capacity, source=0, sink=2
        )

        assert cut_value.dtype == dtype
        assert reachable.dtype == torch.bool
        assert cut_edges.dtype == torch.int64
        assert cut_value.item() == pytest.approx(3.0, rel=1e-2)
