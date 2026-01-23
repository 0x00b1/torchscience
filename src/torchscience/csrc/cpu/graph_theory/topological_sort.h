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
at::Tensor topological_sort_impl(
    const scalar_t* adj,
    int64_t N,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Compute in-degrees
  std::vector<int64_t> in_degree(N, 0);
  for (int64_t j = 0; j < N; ++j) {
    for (int64_t i = 0; i < N; ++i) {
      scalar_t w = adj[i * N + j];
      // Edge exists if weight is finite and non-zero (for non-diagonal)
      // or if it's a self-loop (diagonal with non-zero weight)
      if (i == j) {
        // Self-loop check: non-zero diagonal indicates a cycle
        if (w != scalar_t(0) && !std::isinf(w)) {
          TORCH_CHECK(false, "topological_sort: graph contains a cycle (self-loop at node ", j, ")");
        }
      } else {
        // Edge from i to j exists if weight is finite and positive
        if (w < inf && w > scalar_t(0)) {
          in_degree[j]++;
        }
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

  // Kahn's algorithm
  std::vector<int64_t> order;
  order.reserve(N);

  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();
    order.push_back(u);

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
  if (static_cast<int64_t>(order.size()) != N) {
    TORCH_CHECK(false, "topological_sort: graph contains a cycle");
  }

  // Convert to tensor
  at::Tensor result = at::empty({N}, options.dtype(at::kLong));
  auto result_ptr = result.data_ptr<int64_t>();

  for (int64_t i = 0; i < N; ++i) {
    result_ptr[i] = order[i];
  }

  return result;
}

}  // anonymous namespace

inline at::Tensor topological_sort(const at::Tensor& adjacency) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "topological_sort: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "topological_sort: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );
  TORCH_CHECK(
      adjacency.is_floating_point(),
      "topological_sort: adjacency must be floating-point, got ", adjacency.dtype()
  );

  int64_t N = adjacency.size(-1);

  // Handle empty graph
  if (N == 0) {
    // Return tensor with shape (..., 0) matching batch dims
    auto batch_sizes = adjacency.sizes().slice(0, adjacency.dim() - 2);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(0);
    return at::empty(result_sizes, adjacency.options().dtype(at::kLong));
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
    std::vector<at::Tensor> results;
    results.reserve(batch_numel);

    for (int64_t b = 0; b < batch_numel; ++b) {
      at::Tensor single_adj = flat_adj[b].contiguous();
      at::Tensor single_result;

      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf, at::kBFloat16,
          single_adj.scalar_type(),
          "topological_sort_cpu",
          [&] {
            const scalar_t* adj_ptr = single_adj.data_ptr<scalar_t>();
            single_result = topological_sort_impl<scalar_t>(
                adj_ptr, N, single_adj.options()
            );
          }
      );
      results.push_back(single_result);
    }

    // Stack and reshape to match batch dimensions
    at::Tensor stacked = at::stack(results, 0);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(N);
    return stacked.reshape(result_sizes);
  }

  // Single graph case
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  at::Tensor result;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "topological_sort_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        result = topological_sort_impl<scalar_t>(
            adj_ptr, N, dense_adj.options()
        );
      }
  );

  return result;
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("topological_sort", &torchscience::cpu::graph_theory::topological_sort);
}
