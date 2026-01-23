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
std::tuple<at::Tensor, at::Tensor> dag_shortest_paths_impl(
    const scalar_t* adj,
    int64_t N,
    int64_t source,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Step 1: Compute topological sort using Kahn's algorithm
  // First, compute in-degrees (only count non-diagonal edges)
  // Self-loops are ignored - they don't affect shortest path computation
  // since dist[u] + w[u,u] >= dist[u] for non-negative weights
  std::vector<int64_t> in_degree(N, 0);
  for (int64_t j = 0; j < N; ++j) {
    for (int64_t i = 0; i < N; ++i) {
      if (i == j) continue;  // Skip diagonal (self-loops are harmless)
      scalar_t w = adj[i * N + j];
      // Edge from i to j exists if weight is finite and positive
      if (w < inf && w > scalar_t(0)) {
        in_degree[j]++;
      }
    }
  }

  // Initialize queue with nodes having in-degree 0
  std::queue<int64_t> q;
  for (int64_t i = 0; i < N; ++i) {
    if (in_degree[i] == 0) {
      q.push(i);
    }
  }

  // Kahn's algorithm to get topological order
  std::vector<int64_t> topo_order;
  topo_order.reserve(N);

  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();
    topo_order.push_back(u);

    // For each neighbor v of u (outgoing edges)
    for (int64_t v = 0; v < N; ++v) {
      if (v == u) continue;
      scalar_t w = adj[u * N + v];
      if (w < inf && w > scalar_t(0)) {
        in_degree[v]--;
        if (in_degree[v] == 0) {
          q.push(v);
        }
      }
    }
  }

  // Check for cycle
  if (static_cast<int64_t>(topo_order.size()) != N) {
    TORCH_CHECK(false, "dag_shortest_paths: graph contains a cycle");
  }

  // Step 2: Initialize distances and predecessors
  std::vector<scalar_t> dist(N, inf);
  std::vector<int64_t> pred(N, -1);
  dist[source] = scalar_t(0);

  // Step 3: Relax edges in topological order
  for (int64_t u : topo_order) {
    if (std::isinf(dist[u])) continue;  // Skip unreachable nodes

    // For each neighbor v of u
    for (int64_t v = 0; v < N; ++v) {
      if (v == u) continue;
      scalar_t w = adj[u * N + v];
      if (w < inf && w > scalar_t(0)) {
        scalar_t new_dist = dist[u] + w;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pred[v] = u;
        }
      }
    }
  }

  // Convert to tensors
  at::Tensor distances = at::empty({N}, options);
  at::Tensor predecessors = at::empty({N}, options.dtype(at::kLong));

  auto dist_ptr = distances.data_ptr<scalar_t>();
  auto pred_ptr = predecessors.data_ptr<int64_t>();

  for (int64_t i = 0; i < N; ++i) {
    dist_ptr[i] = dist[i];
    pred_ptr[i] = pred[i];
  }

  return std::make_tuple(distances, predecessors);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> dag_shortest_paths(
    const at::Tensor& adjacency,
    int64_t source
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "dag_shortest_paths: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "dag_shortest_paths: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);
  TORCH_CHECK(
      source >= 0 && source < N,
      "dag_shortest_paths: source must be in [0, ", N - 1, "], got ", source
  );

  // Handle sparse input
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::empty({0}, dense_adj.options()),
        at::empty({0}, dense_adj.options().dtype(at::kLong))
    );
  }

  at::Tensor distances, predecessors;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "dag_shortest_paths_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(distances, predecessors) = dag_shortest_paths_impl<scalar_t>(
            adj_ptr, N, source, dense_adj.options()
        );
      }
  );

  return std::make_tuple(distances, predecessors);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("dag_shortest_paths", &torchscience::cpu::graph_theory::dag_shortest_paths);
}
