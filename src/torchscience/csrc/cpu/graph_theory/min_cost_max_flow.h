#pragma once

#include <limits>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

// Bellman-Ford to find shortest path from source to sink in residual graph
// with edge costs. Returns true if a path exists, and fills parent array.
template <typename scalar_t>
bool bellman_ford_shortest_path(
    const std::vector<std::vector<scalar_t>>& residual,
    const std::vector<std::vector<scalar_t>>& cost_graph,
    int64_t N,
    int64_t source,
    int64_t sink,
    std::vector<int64_t>& parent,
    std::vector<scalar_t>& dist
) {
  const scalar_t inf = std::numeric_limits<scalar_t>::max() / scalar_t(2);

  // Initialize distances
  std::fill(dist.begin(), dist.end(), inf);
  std::fill(parent.begin(), parent.end(), -1);
  dist[source] = scalar_t(0);

  // Relax edges N-1 times
  for (int64_t iter = 0; iter < N - 1; ++iter) {
    bool changed = false;
    for (int64_t u = 0; u < N; ++u) {
      if (dist[u] >= inf) continue;
      for (int64_t v = 0; v < N; ++v) {
        // Check if edge (u, v) has residual capacity
        if (residual[u][v] > scalar_t(0)) {
          scalar_t new_dist = dist[u] + cost_graph[u][v];
          if (new_dist < dist[v]) {
            dist[v] = new_dist;
            parent[v] = u;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;  // Early termination if no changes
  }

  return parent[sink] != -1;  // Path exists if sink is reachable
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> min_cost_max_flow_impl(
    const scalar_t* capacity_ptr,
    const scalar_t* cost_ptr,
    int64_t N,
    int64_t source,
    int64_t sink,
    const at::TensorOptions& options
) {
  // Handle source == sink case
  if (source == sink) {
    at::Tensor max_flow = at::zeros({}, options);
    at::Tensor total_cost = at::zeros({}, options);
    at::Tensor flow = at::zeros({N, N}, options);
    return std::make_tuple(max_flow, total_cost, flow);
  }

  // Initialize residual graph (copy of capacity)
  std::vector<std::vector<scalar_t>> residual(N, std::vector<scalar_t>(N, scalar_t(0)));
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      residual[i][j] = capacity_ptr[i * N + j];
    }
  }

  // Cost graph for residual network
  // Forward edge (i,j): cost[i,j]
  // Reverse edge (j,i): -cost[i,j] (to cancel out cost when flow is pushed back)
  std::vector<std::vector<scalar_t>> cost_graph(N, std::vector<scalar_t>(N, scalar_t(0)));
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      cost_graph[i][j] = cost_ptr[i * N + j];
    }
  }

  // Parent array for path reconstruction
  std::vector<int64_t> parent(N, -1);
  // Distance array for Bellman-Ford
  std::vector<scalar_t> dist(N);

  // Total max flow and cost
  scalar_t total_flow = scalar_t(0);
  scalar_t total_flow_cost = scalar_t(0);

  // Flow matrix to track actual flow on each edge
  std::vector<std::vector<scalar_t>> flow_values(N, std::vector<scalar_t>(N, scalar_t(0)));

  // Successive shortest paths algorithm
  while (bellman_ford_shortest_path<scalar_t>(
             residual, cost_graph, N, source, sink, parent, dist)) {
    // Find bottleneck capacity along the shortest path
    scalar_t path_flow = std::numeric_limits<scalar_t>::max();
    int64_t v = sink;
    while (v != source) {
      int64_t u = parent[v];
      path_flow = std::min(path_flow, residual[u][v]);
      v = u;
    }

    // Update residual capacities and flow along the path
    v = sink;
    while (v != source) {
      int64_t u = parent[v];
      residual[u][v] -= path_flow;
      residual[v][u] += path_flow;

      // Update cost graph for residual edges
      // When we use edge (u,v), we need reverse edge (v,u) with negative cost
      cost_graph[v][u] = -cost_graph[u][v];

      v = u;
    }

    total_flow += path_flow;
    total_flow_cost += path_flow * dist[sink];
  }

  // Compute actual flow from residual graph
  // flow[u][v] = capacity[u][v] - residual[u][v] (but only if positive)
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      scalar_t cap = capacity_ptr[i * N + j];
      scalar_t res = residual[i][j];
      scalar_t f = cap - res;
      if (f > scalar_t(0)) {
        flow_values[i][j] = f;
      }
    }
  }

  // Convert to tensors
  at::Tensor max_flow_tensor = at::empty({}, options);
  at::Tensor total_cost_tensor = at::empty({}, options);
  at::Tensor flow_tensor = at::empty({N, N}, options);

  max_flow_tensor.fill_(total_flow);
  total_cost_tensor.fill_(total_flow_cost);

  auto flow_ptr_out = flow_tensor.data_ptr<scalar_t>();
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      flow_ptr_out[i * N + j] = flow_values[i][j];
    }
  }

  return std::make_tuple(max_flow_tensor, total_cost_tensor, flow_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> min_cost_max_flow(
    const at::Tensor& capacity,
    const at::Tensor& cost,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "min_cost_max_flow: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "min_cost_max_flow: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );
  TORCH_CHECK(
      cost.dim() == 2,
      "min_cost_max_flow: cost must be 2D, got ", cost.dim(), "D"
  );
  TORCH_CHECK(
      cost.size(0) == cost.size(1),
      "min_cost_max_flow: cost must be square, got ",
      cost.size(0), " x ", cost.size(1)
  );
  TORCH_CHECK(
      capacity.size(0) == cost.size(0) && capacity.size(1) == cost.size(1),
      "min_cost_max_flow: capacity and cost must have same shape, got ",
      capacity.size(0), " x ", capacity.size(1), " vs ",
      cost.size(0), " x ", cost.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "min_cost_max_flow: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "min_cost_max_flow: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Handle sparse input
  at::Tensor dense_capacity = capacity.is_sparse() ? capacity.to_dense() : capacity;
  at::Tensor dense_cost = cost.is_sparse() ? cost.to_dense() : cost;
  dense_capacity = dense_capacity.contiguous();
  dense_cost = dense_cost.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::zeros({}, dense_capacity.options()),
        at::zeros({}, dense_capacity.options()),
        at::empty({0, 0}, dense_capacity.options())
    );
  }

  at::Tensor max_flow, total_cost_out, flow;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_capacity.scalar_type(),
      "min_cost_max_flow_cpu",
      [&] {
        const scalar_t* cap_ptr = dense_capacity.data_ptr<scalar_t>();
        const scalar_t* cost_ptr_in = dense_cost.data_ptr<scalar_t>();
        std::tie(max_flow, total_cost_out, flow) = min_cost_max_flow_impl<scalar_t>(
            cap_ptr, cost_ptr_in, N, source, sink, dense_capacity.options()
        );
      }
  );

  return std::make_tuple(max_flow, total_cost_out, flow);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("min_cost_max_flow", &torchscience::cpu::graph_theory::min_cost_max_flow);
}
