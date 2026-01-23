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
std::tuple<at::Tensor, at::Tensor> minimum_spanning_tree_impl(
    const scalar_t* adj,
    int64_t N,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Prim's algorithm
  // Priority queue: (weight, from_node, to_node)
  using PQElement = std::tuple<scalar_t, int64_t, int64_t>;
  std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;

  std::vector<bool> in_mst(N, false);
  std::vector<std::pair<int64_t, int64_t>> mst_edges;
  scalar_t total_weight = scalar_t(0);

  // Start from node 0
  in_mst[0] = true;
  int64_t nodes_in_mst = 1;

  // Add edges from node 0 to the priority queue
  for (int64_t v = 1; v < N; ++v) {
    // Use minimum of both directions (undirected)
    scalar_t w = adj[0 * N + v];
    scalar_t w_rev = adj[v * N + 0];
    if (w_rev < w) w = w_rev;

    if (w < inf) {
      pq.push({w, 0, v});
    }
  }

  // Build MST
  while (!pq.empty() && nodes_in_mst < N) {
    auto [weight, u, v] = pq.top();
    pq.pop();

    // Skip if target already in MST
    if (in_mst[v]) continue;

    // Add edge to MST
    in_mst[v] = true;
    nodes_in_mst++;
    total_weight += weight;
    mst_edges.push_back({u, v});

    // Add edges from v to non-MST nodes
    for (int64_t w_node = 0; w_node < N; ++w_node) {
      if (!in_mst[w_node]) {
        // Use minimum of both directions (undirected)
        scalar_t edge_w = adj[v * N + w_node];
        scalar_t edge_w_rev = adj[w_node * N + v];
        if (edge_w_rev < edge_w) edge_w = edge_w_rev;

        if (edge_w < inf) {
          pq.push({edge_w, v, w_node});
        }
      }
    }
  }

  // Check if graph is connected
  if (nodes_in_mst < N) {
    total_weight = inf;
  }

  // Create output tensors
  at::Tensor weight_tensor = at::empty({}, options);
  weight_tensor.fill_(total_weight);

  // Create edges tensor of shape (N-1, 2), padded with -1 if disconnected
  int64_t num_edges = N > 0 ? N - 1 : 0;
  at::Tensor edges_tensor = at::full({num_edges, 2}, -1, options.dtype(at::kLong));

  if (num_edges > 0) {
    auto edges_ptr = edges_tensor.data_ptr<int64_t>();
    for (size_t i = 0; i < mst_edges.size(); ++i) {
      edges_ptr[i * 2] = mst_edges[i].first;
      edges_ptr[i * 2 + 1] = mst_edges[i].second;
    }
  }

  return std::make_tuple(weight_tensor, edges_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> minimum_spanning_tree(
    const at::Tensor& adjacency
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "minimum_spanning_tree: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "minimum_spanning_tree: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );

  int64_t N = adjacency.size(-1);

  // Handle empty graph
  if (N == 0) {
    auto batch_sizes = adjacency.sizes().slice(0, adjacency.dim() - 2);
    std::vector<int64_t> weight_sizes(batch_sizes.begin(), batch_sizes.end());
    std::vector<int64_t> edge_sizes(batch_sizes.begin(), batch_sizes.end());
    edge_sizes.push_back(0);
    edge_sizes.push_back(2);
    return std::make_tuple(
        at::zeros(weight_sizes, adjacency.options()),
        at::empty(edge_sizes, adjacency.options().dtype(at::kLong))
    );
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
    std::vector<at::Tensor> weight_results;
    std::vector<at::Tensor> edge_results;
    weight_results.reserve(batch_numel);
    edge_results.reserve(batch_numel);

    for (int64_t b = 0; b < batch_numel; ++b) {
      at::Tensor single_adj = flat_adj[b].contiguous();

      // Handle single-node graph
      if (N == 1) {
        weight_results.push_back(at::zeros({}, single_adj.options()));
        edge_results.push_back(at::empty({0, 2}, single_adj.options().dtype(at::kLong)));
        continue;
      }

      at::Tensor single_weight, single_edges;

      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::kHalf, at::kBFloat16,
          single_adj.scalar_type(),
          "minimum_spanning_tree_cpu",
          [&] {
            const scalar_t* adj_ptr = single_adj.data_ptr<scalar_t>();
            std::tie(single_weight, single_edges) = minimum_spanning_tree_impl<scalar_t>(
                adj_ptr, N, single_adj.options()
            );
          }
      );

      weight_results.push_back(single_weight);
      edge_results.push_back(single_edges);
    }

    // Stack and reshape to match batch dimensions
    at::Tensor stacked_weight = at::stack(weight_results, 0);
    at::Tensor stacked_edges = at::stack(edge_results, 0);

    std::vector<int64_t> weight_sizes(batch_sizes.begin(), batch_sizes.end());
    std::vector<int64_t> edge_sizes(batch_sizes.begin(), batch_sizes.end());
    int64_t num_edges = N > 1 ? N - 1 : 0;
    edge_sizes.push_back(num_edges);
    edge_sizes.push_back(2);

    return std::make_tuple(
        stacked_weight.reshape(weight_sizes),
        stacked_edges.reshape(edge_sizes)
    );
  }

  // Single graph case
  // Handle sparse input
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  // Handle single-node graph
  if (N == 1) {
    return std::make_tuple(
        at::zeros({}, dense_adj.options()),
        at::empty({0, 2}, dense_adj.options().dtype(at::kLong))
    );
  }

  at::Tensor total_weight, edges;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "minimum_spanning_tree_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        std::tie(total_weight, edges) = minimum_spanning_tree_impl<scalar_t>(
            adj_ptr, N, dense_adj.options()
        );
      }
  );

  return std::make_tuple(total_weight, edges);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("minimum_spanning_tree", &torchscience::cpu::graph_theory::minimum_spanning_tree);
}
