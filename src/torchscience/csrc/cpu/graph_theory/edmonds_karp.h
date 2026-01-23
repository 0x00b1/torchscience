#pragma once

#include <limits>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
bool bfs_find_path(
    const std::vector<std::vector<scalar_t>>& residual,
    int64_t N,
    int64_t source,
    int64_t sink,
    std::vector<int64_t>& parent
) {
  // BFS to find augmenting path in residual graph
  std::fill(parent.begin(), parent.end(), -1);
  std::vector<bool> visited(N, false);

  std::queue<int64_t> q;
  q.push(source);
  visited[source] = true;

  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();

    for (int64_t v = 0; v < N; ++v) {
      // Check if edge has residual capacity and v is not visited
      if (!visited[v] && residual[u][v] > scalar_t(0)) {
        visited[v] = true;
        parent[v] = u;
        if (v == sink) {
          return true;  // Found path to sink
        }
        q.push(v);
      }
    }
  }

  return false;  // No augmenting path found
}

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor> edmonds_karp_impl(
    const scalar_t* capacity_ptr,
    int64_t N,
    int64_t source,
    int64_t sink,
    const at::TensorOptions& options
) {
  // Handle source == sink case
  if (source == sink) {
    at::Tensor max_flow = at::zeros({}, options);
    at::Tensor flow = at::zeros({N, N}, options);
    return std::make_tuple(max_flow, flow);
  }

  // Initialize residual graph (copy of capacity)
  std::vector<std::vector<scalar_t>> residual(N, std::vector<scalar_t>(N, scalar_t(0)));
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      residual[i][j] = capacity_ptr[i * N + j];
    }
  }

  // Parent array for BFS path reconstruction
  std::vector<int64_t> parent(N, -1);

  // Total max flow
  scalar_t total_flow = scalar_t(0);

  // Edmonds-Karp: use BFS to find augmenting paths
  while (bfs_find_path<scalar_t>(residual, N, source, sink, parent)) {
    // Find bottleneck capacity along the path
    scalar_t path_flow = std::numeric_limits<scalar_t>::max();
    int64_t v = sink;
    while (v != source) {
      int64_t u = parent[v];
      path_flow = std::min(path_flow, residual[u][v]);
      v = u;
    }

    // Update residual capacities along the path
    v = sink;
    while (v != source) {
      int64_t u = parent[v];
      residual[u][v] -= path_flow;
      residual[v][u] += path_flow;  // Add reverse edge for flow cancellation
      v = u;
    }

    total_flow += path_flow;
  }

  // Compute actual flow from residual graph
  // flow[u][v] = capacity[u][v] - residual[u][v]
  std::vector<std::vector<scalar_t>> flow_values(N, std::vector<scalar_t>(N, scalar_t(0)));
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      scalar_t cap = capacity_ptr[i * N + j];
      scalar_t res = residual[i][j];
      // Flow on edge (i,j) is capacity - residual (but only if positive)
      scalar_t f = cap - res;
      if (f > scalar_t(0)) {
        flow_values[i][j] = f;
      }
    }
  }

  // Convert to tensors
  at::Tensor max_flow_tensor = at::empty({}, options);
  at::Tensor flow_tensor = at::empty({N, N}, options);

  max_flow_tensor.fill_(total_flow);

  auto flow_ptr = flow_tensor.data_ptr<scalar_t>();
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      flow_ptr[i * N + j] = flow_values[i][j];
    }
  }

  return std::make_tuple(max_flow_tensor, flow_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> edmonds_karp(
    const at::Tensor& capacity,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "edmonds_karp: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "edmonds_karp: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "edmonds_karp: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "edmonds_karp: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Handle sparse input
  at::Tensor dense_capacity = capacity.is_sparse() ? capacity.to_dense() : capacity;
  dense_capacity = dense_capacity.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::zeros({}, dense_capacity.options()),
        at::empty({0, 0}, dense_capacity.options())
    );
  }

  at::Tensor max_flow, flow;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_capacity.scalar_type(),
      "edmonds_karp_cpu",
      [&] {
        const scalar_t* cap_ptr = dense_capacity.data_ptr<scalar_t>();
        std::tie(max_flow, flow) = edmonds_karp_impl<scalar_t>(
            cap_ptr, N, source, sink, dense_capacity.options()
        );
      }
  );

  return std::make_tuple(max_flow, flow);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("edmonds_karp", &torchscience::cpu::graph_theory::edmonds_karp);
}
