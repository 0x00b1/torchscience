#pragma once

#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
bool bfs_find_path_mincut(
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
std::tuple<at::Tensor, at::Tensor, at::Tensor> minimum_cut_impl(
    const scalar_t* capacity_ptr,
    int64_t N,
    int64_t source,
    int64_t sink,
    const at::TensorOptions& options
) {
  // Handle source == sink case
  if (source == sink) {
    at::Tensor cut_value = at::zeros({}, options);
    at::Tensor reachable = at::zeros({N}, options.dtype(at::kBool));
    reachable[source] = true;
    at::Tensor cut_edges = at::empty({0, 2}, options.dtype(at::kLong));
    return std::make_tuple(cut_value, reachable, cut_edges);
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

  // Total max flow (will equal cut value)
  scalar_t total_flow = scalar_t(0);

  // Edmonds-Karp: use BFS to find augmenting paths
  while (bfs_find_path_mincut<scalar_t>(residual, N, source, sink, parent)) {
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

  // Find nodes reachable from source in residual graph using BFS
  std::vector<bool> reachable_vec(N, false);
  {
    std::queue<int64_t> q;
    q.push(source);
    reachable_vec[source] = true;

    while (!q.empty()) {
      int64_t u = q.front();
      q.pop();

      for (int64_t v = 0; v < N; ++v) {
        if (!reachable_vec[v] && residual[u][v] > scalar_t(0)) {
          reachable_vec[v] = true;
          q.push(v);
        }
      }
    }
  }

  // Find cut edges: from reachable to non-reachable with positive capacity
  std::vector<std::pair<int64_t, int64_t>> cut_edges_vec;
  for (int64_t u = 0; u < N; ++u) {
    if (reachable_vec[u]) {
      for (int64_t v = 0; v < N; ++v) {
        if (!reachable_vec[v] && capacity_ptr[u * N + v] > scalar_t(0)) {
          cut_edges_vec.push_back({u, v});
        }
      }
    }
  }

  // Convert to tensors
  at::Tensor cut_value_tensor = at::empty({}, options);
  cut_value_tensor.fill_(total_flow);

  at::Tensor reachable_tensor = at::empty({N}, options.dtype(at::kBool));
  auto reachable_ptr = reachable_tensor.data_ptr<bool>();
  for (int64_t i = 0; i < N; ++i) {
    reachable_ptr[i] = reachable_vec[i];
  }

  int64_t num_cut_edges = static_cast<int64_t>(cut_edges_vec.size());
  at::Tensor cut_edges_tensor = at::empty({num_cut_edges, 2}, options.dtype(at::kLong));
  auto cut_edges_ptr = cut_edges_tensor.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_cut_edges; ++i) {
    cut_edges_ptr[i * 2] = cut_edges_vec[i].first;
    cut_edges_ptr[i * 2 + 1] = cut_edges_vec[i].second;
  }

  return std::make_tuple(cut_value_tensor, reachable_tensor, cut_edges_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> minimum_cut(
    const at::Tensor& capacity,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "minimum_cut: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "minimum_cut: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "minimum_cut: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "minimum_cut: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Handle sparse input
  at::Tensor dense_capacity = capacity.is_sparse() ? capacity.to_dense() : capacity;
  dense_capacity = dense_capacity.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::zeros({}, dense_capacity.options()),
        at::empty({0}, dense_capacity.options().dtype(at::kBool)),
        at::empty({0, 2}, dense_capacity.options().dtype(at::kLong))
    );
  }

  at::Tensor cut_value, reachable, cut_edges;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_capacity.scalar_type(),
      "minimum_cut_cpu",
      [&] {
        const scalar_t* cap_ptr = dense_capacity.data_ptr<scalar_t>();
        std::tie(cut_value, reachable, cut_edges) = minimum_cut_impl<scalar_t>(
            cap_ptr, N, source, sink, dense_capacity.options()
        );
      }
  );

  return std::make_tuple(cut_value, reachable, cut_edges);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("minimum_cut", &torchscience::cpu::graph_theory::minimum_cut);
}
