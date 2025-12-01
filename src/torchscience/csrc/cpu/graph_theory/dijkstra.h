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
std::tuple<at::Tensor, at::Tensor> dijkstra_impl(
    const scalar_t* adj,
    int64_t N,
    int64_t source,
    bool directed,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Initialize distances and predecessors
  std::vector<scalar_t> dist(N, inf);
  std::vector<int64_t> pred(N, -1);

  dist[source] = scalar_t(0);

  // Priority queue: (distance, node)
  using PQElement = std::pair<scalar_t, int64_t>;
  std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
  pq.push({scalar_t(0), source});

  std::vector<bool> visited(N, false);

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();

    if (visited[u]) continue;
    visited[u] = true;

    // Explore neighbors
    for (int64_t v = 0; v < N; ++v) {
      scalar_t w = adj[u * N + v];

      // For undirected graphs, also check reverse edge
      if (!directed) {
        scalar_t w_rev = adj[v * N + u];
        if (w_rev < w) w = w_rev;
      }

      if (w < inf && !visited[v]) {
        scalar_t new_dist = dist[u] + w;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pred[v] = u;
          pq.push({new_dist, v});
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

inline std::tuple<at::Tensor, at::Tensor> dijkstra(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "dijkstra: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "dijkstra: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);
  TORCH_CHECK(
      source >= 0 && source < N,
      "dijkstra: source must be in [0, ", N - 1, "], got ", source
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
      "dijkstra_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(distances, predecessors) = dijkstra_impl<scalar_t>(
            adj_ptr, N, source, directed, dense_adj.options()
        );
      }
  );

  return std::make_tuple(distances, predecessors);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("dijkstra", &torchscience::cpu::graph_theory::dijkstra);
}
