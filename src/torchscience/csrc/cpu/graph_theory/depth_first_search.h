#pragma once

#include <limits>
#include <stack>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> depth_first_search_impl(
    const scalar_t* adj,
    int64_t N,
    int64_t source,
    bool directed,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Initialize discovery time, finish time, and predecessors
  std::vector<int64_t> disc(N, -1);    // -1 means not yet discovered
  std::vector<int64_t> finish(N, -1);  // -1 means not yet finished
  std::vector<int64_t> pred(N, -1);    // -1 means no predecessor
  std::vector<bool> visited(N, false);

  // DFS using an iterative approach with explicit stack
  // Stack stores (node, next_neighbor_to_explore)
  std::stack<std::pair<int64_t, int64_t>> stk;

  int64_t time = 0;

  // Start DFS from source
  disc[source] = time++;
  visited[source] = true;
  stk.push({source, 0});

  while (!stk.empty()) {
    auto& [u, next_v] = stk.top();

    // Find the next unvisited neighbor
    bool found_neighbor = false;
    for (int64_t v = next_v; v < N; ++v) {
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

      if (has_edge && !visited[v]) {
        // Update the next neighbor to explore for current node
        stk.top().second = v + 1;

        // Discover v
        disc[v] = time++;
        visited[v] = true;
        pred[v] = u;

        // Push v onto stack to explore its neighbors
        stk.push({v, 0});
        found_neighbor = true;
        break;
      }
    }

    if (!found_neighbor) {
      // All neighbors explored, finish this node
      finish[u] = time++;
      stk.pop();
    }
  }

  // Convert to tensors
  at::Tensor disc_tensor = at::empty({N}, options.dtype(at::kLong));
  at::Tensor finish_tensor = at::empty({N}, options.dtype(at::kLong));
  at::Tensor pred_tensor = at::empty({N}, options.dtype(at::kLong));
  auto disc_ptr = disc_tensor.data_ptr<int64_t>();
  auto finish_ptr = finish_tensor.data_ptr<int64_t>();
  auto pred_ptr = pred_tensor.data_ptr<int64_t>();

  for (int64_t i = 0; i < N; ++i) {
    disc_ptr[i] = disc[i];
    finish_ptr[i] = finish[i];
    pred_ptr[i] = pred[i];
  }

  return std::make_tuple(disc_tensor, finish_tensor, pred_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> depth_first_search(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "depth_first_search: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "depth_first_search: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );
  TORCH_CHECK(
      adjacency.is_floating_point(),
      "depth_first_search: adjacency must be floating-point, got ", adjacency.dtype()
  );

  int64_t N = adjacency.size(-1);

  TORCH_CHECK(
      source >= 0 && source < N,
      "depth_first_search: source must be in range [0, ", N, "), got ", source
  );

  // Handle empty graph
  if (N == 0) {
    auto batch_sizes = adjacency.sizes().slice(0, adjacency.dim() - 2);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(0);
    auto disc = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
    auto finish = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
    auto pred = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
    return std::make_tuple(disc, finish, pred);
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
    std::vector<at::Tensor> disc_results;
    std::vector<at::Tensor> finish_results;
    std::vector<at::Tensor> pred_results;
    disc_results.reserve(batch_numel);
    finish_results.reserve(batch_numel);
    pred_results.reserve(batch_numel);

    for (int64_t b = 0; b < batch_numel; ++b) {
      at::Tensor single_adj = flat_adj[b].contiguous();
      at::Tensor single_disc, single_finish, single_pred;

      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf, at::kBFloat16,
          single_adj.scalar_type(),
          "depth_first_search_cpu",
          [&] {
            const scalar_t* adj_ptr = single_adj.data_ptr<scalar_t>();
            std::tie(single_disc, single_finish, single_pred) = depth_first_search_impl<scalar_t>(
                adj_ptr, N, source, directed, single_adj.options()
            );
          }
      );
      disc_results.push_back(single_disc);
      finish_results.push_back(single_finish);
      pred_results.push_back(single_pred);
    }

    // Stack and reshape to match batch dimensions
    at::Tensor stacked_disc = at::stack(disc_results, 0);
    at::Tensor stacked_finish = at::stack(finish_results, 0);
    at::Tensor stacked_pred = at::stack(pred_results, 0);
    std::vector<int64_t> result_sizes(batch_sizes.begin(), batch_sizes.end());
    result_sizes.push_back(N);
    return std::make_tuple(
        stacked_disc.reshape(result_sizes),
        stacked_finish.reshape(result_sizes),
        stacked_pred.reshape(result_sizes)
    );
  }

  // Single graph case
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  at::Tensor disc, finish, pred;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "depth_first_search_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(disc, finish, pred) = depth_first_search_impl<scalar_t>(
            adj_ptr, N, source, directed, dense_adj.options()
        );
      }
  );

  return std::make_tuple(disc, finish, pred);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("depth_first_search", &torchscience::cpu::graph_theory::depth_first_search);
}
