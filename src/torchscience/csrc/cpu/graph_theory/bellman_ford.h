#pragma once

#include <limits>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, bool> bellman_ford_impl(
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

  // Build edge list for efficiency
  std::vector<std::tuple<int64_t, int64_t, scalar_t>> edges;
  for (int64_t u = 0; u < N; ++u) {
    for (int64_t v = 0; v < N; ++v) {
      scalar_t w = adj[u * N + v];
      if (w < inf) {
        edges.push_back({u, v, w});
      }
      // For undirected graphs, add reverse edge if different
      if (!directed && adj[v * N + u] < adj[u * N + v]) {
        // We'll handle this by checking both directions
      }
    }
  }

  // Relax edges N-1 times
  for (int64_t iter = 0; iter < N - 1; ++iter) {
    bool changed = false;
    for (const auto& [u, v, w] : edges) {
      if (dist[u] < inf) {
        scalar_t new_dist = dist[u] + w;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pred[v] = u;
          changed = true;
        }
      }
    }
    // For undirected graphs, also relax in reverse direction
    if (!directed) {
      for (const auto& [u, v, w] : edges) {
        if (dist[v] < inf) {
          scalar_t new_dist = dist[v] + w;
          if (new_dist < dist[u]) {
            dist[u] = new_dist;
            pred[u] = v;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;  // Early termination
  }

  // Check for negative cycles (one more iteration)
  bool has_negative_cycle = false;
  for (const auto& [u, v, w] : edges) {
    if (dist[u] < inf && dist[u] + w < dist[v]) {
      has_negative_cycle = true;
      break;
    }
  }
  if (!has_negative_cycle && !directed) {
    for (const auto& [u, v, w] : edges) {
      if (dist[v] < inf && dist[v] + w < dist[u]) {
        has_negative_cycle = true;
        break;
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

  return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, bool> bellman_ford(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "bellman_ford: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "bellman_ford: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);
  TORCH_CHECK(
      source >= 0 && source < N,
      "bellman_ford: source must be in [0, ", N - 1, "], got ", source
  );

  // Handle sparse input
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::empty({0}, dense_adj.options()),
        at::empty({0}, dense_adj.options().dtype(at::kLong)),
        false
    );
  }

  at::Tensor distances, predecessors;
  bool has_negative_cycle = false;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "bellman_ford_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(distances, predecessors, has_negative_cycle) =
            bellman_ford_impl<scalar_t>(
                adj_ptr, N, source, directed, dense_adj.options()
            );
      }
  );

  return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("bellman_ford", &torchscience::cpu::graph_theory::bellman_ford);
}
