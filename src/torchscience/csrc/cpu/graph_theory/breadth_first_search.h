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
std::tuple<at::Tensor, at::Tensor> breadth_first_search_impl(
    const scalar_t* adj,
    int64_t N,
    int64_t source,
    bool directed,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Initialize distances and predecessors
  std::vector<int64_t> dist(N, -1);  // -1 means unreachable
  std::vector<int64_t> pred(N, -1);  // -1 means no predecessor

  // BFS queue
  std::queue<int64_t> q;
  q.push(source);
  dist[source] = 0;

  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();

    // For each neighbor v of u
    for (int64_t v = 0; v < N; ++v) {
      if (v == u) continue;

      // Check if edge exists
      bool has_edge = false;
      if (directed) {
        // Directed: only check u -> v
        scalar_t w = adj[u * N + v];
        if (w < inf && w > scalar_t(0)) {
          has_edge = true;
        }
      } else {
        // Undirected: check both u -> v and v -> u
        scalar_t w_uv = adj[u * N + v];
        scalar_t w_vu = adj[v * N + u];
        if ((w_uv < inf && w_uv > scalar_t(0)) ||
            (w_vu < inf && w_vu > scalar_t(0))) {
          has_edge = true;
        }
      }

      if (has_edge && dist[v] == -1) {
        dist[v] = dist[u] + 1;
        pred[v] = u;
        q.push(v);
      }
    }
  }

  // Convert to tensors
  at::Tensor dist_tensor = at::empty({N}, options.dtype(at::kLong));
  at::Tensor pred_tensor = at::empty({N}, options.dtype(at::kLong));
  auto dist_ptr = dist_tensor.data_ptr<int64_t>();
  auto pred_ptr = pred_tensor.data_ptr<int64_t>();

  for (int64_t i = 0; i < N; ++i) {
    dist_ptr[i] = dist[i];
    pred_ptr[i] = pred[i];
  }

  return std::make_tuple(dist_tensor, pred_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> breadth_first_search(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "breadth_first_search: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "breadth_first_search: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );
  TORCH_CHECK(
      adjacency.is_floating_point(),
      "breadth_first_search: adjacency must be floating-point, got ", adjacency.dtype()
  );

  int64_t N = adjacency.size(-1);

  TORCH_CHECK(
      source >= 0 && source < N,
      "breadth_first_search: source must be in range [0, ", N, "), got ", source
  );

  // Handle empty graph
  if (N == 0) {
    auto batch_sizes = adjacency.sizes().slice(0, adjacency.dim() - 2);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(0);
    auto dist = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
    auto pred = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
    return std::make_tuple(dist, pred);
  }

  // Handle batched input
  if (adjacency.dim() > 2) {
    auto batch_sizes = adjacency.sizes().slice(0, adjacency.dim() - 2);
    int64_t batch_numel = 1;
    for (auto s : batch_sizes) {
      batch_numel *= s;
    }

    // Flatten batch dimensions
    at::Tensor flat_adj = adjacency.reshape({batch_numel, N, N});

    // Process each graph in the batch
    std::vector<at::Tensor> dist_results;
    std::vector<at::Tensor> pred_results;
    dist_results.reserve(batch_numel);
    pred_results.reserve(batch_numel);

    for (int64_t b = 0; b < batch_numel; ++b) {
      at::Tensor single_adj = flat_adj[b].contiguous();
      at::Tensor single_dist, single_pred;

      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf, at::kBFloat16,
          single_adj.scalar_type(),
          "breadth_first_search_cpu",
          [&] {
            const scalar_t* adj_ptr = single_adj.data_ptr<scalar_t>();
            std::tie(single_dist, single_pred) = breadth_first_search_impl<scalar_t>(
                adj_ptr, N, source, directed, single_adj.options()
            );
          }
      );
      dist_results.push_back(single_dist);
      pred_results.push_back(single_pred);
    }

    // Stack and reshape to match batch dimensions
    at::Tensor stacked_dist = at::stack(dist_results, 0);
    at::Tensor stacked_pred = at::stack(pred_results, 0);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(N);
    return std::make_tuple(
        stacked_dist.reshape(result_sizes),
        stacked_pred.reshape(result_sizes)
    );
  }

  // Single graph case
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  at::Tensor dist, pred;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "breadth_first_search_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(dist, pred) = breadth_first_search_impl<scalar_t>(
            adj_ptr, N, source, directed, dense_adj.options()
        );
      }
  );

  return std::make_tuple(dist, pred);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("breadth_first_search", &torchscience::cpu::graph_theory::breadth_first_search);
}
