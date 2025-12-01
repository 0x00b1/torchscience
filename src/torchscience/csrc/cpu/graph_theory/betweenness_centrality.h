#pragma once

#include <limits>
#include <queue>
#include <stack>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
at::Tensor betweenness_centrality_impl(
    const scalar_t* adj,
    int64_t N,
    bool normalized,
    const at::TensorOptions& options
) {
  // Brandes' algorithm for betweenness centrality
  // O(VE) for unweighted graphs, O(VE + V^2 log V) for weighted

  std::vector<scalar_t> betweenness(N, scalar_t(0));

  // Run BFS/Dijkstra from each source
  for (int64_t s = 0; s < N; ++s) {
    // Stack for reverse order traversal
    std::stack<int64_t> S;

    // Predecessors on shortest paths
    std::vector<std::vector<int64_t>> P(N);

    // Number of shortest paths from s to each vertex
    std::vector<scalar_t> sigma(N, scalar_t(0));
    sigma[s] = scalar_t(1);

    // Distance from s
    std::vector<scalar_t> dist(N, scalar_t(-1));
    dist[s] = scalar_t(0);

    // BFS queue for unweighted graphs
    std::queue<int64_t> Q;
    Q.push(s);

    while (!Q.empty()) {
      int64_t v = Q.front();
      Q.pop();
      S.push(v);

      // Explore neighbors
      for (int64_t w = 0; w < N; ++w) {
        scalar_t weight = adj[v * N + w];
        // Check both directions for undirected graph
        scalar_t weight_rev = adj[w * N + v];
        if (weight_rev > 0 && (weight <= 0 || weight_rev < weight)) {
          weight = weight_rev;
        }

        if (weight > 0) {  // Edge exists
          // First visit to w?
          if (dist[w] < 0) {
            dist[w] = dist[v] + scalar_t(1);
            Q.push(w);
          }

          // Is v on a shortest path to w?
          if (dist[w] == dist[v] + scalar_t(1)) {
            sigma[w] += sigma[v];
            P[w].push_back(v);
          }
        }
      }
    }

    // Accumulation phase - back propagation of dependencies
    std::vector<scalar_t> delta(N, scalar_t(0));

    while (!S.empty()) {
      int64_t w = S.top();
      S.pop();

      for (int64_t v : P[w]) {
        delta[v] += (sigma[v] / sigma[w]) * (scalar_t(1) + delta[w]);
      }

      if (w != s) {
        betweenness[w] += delta[w];
      }
    }
  }

  // For undirected graphs, each shortest path was counted twice
  // Divide by 2
  for (int64_t i = 0; i < N; ++i) {
    betweenness[i] /= scalar_t(2);
  }

  // Normalize if requested
  // For undirected graphs, max betweenness = (N-1)(N-2)/2
  if (normalized && N > 2) {
    scalar_t norm = static_cast<scalar_t>((N - 1) * (N - 2)) / scalar_t(2);
    for (int64_t i = 0; i < N; ++i) {
      betweenness[i] /= norm;
    }
  }

  // Convert to tensor
  at::Tensor result = at::empty({N}, options);
  auto result_ptr = result.data_ptr<scalar_t>();
  for (int64_t i = 0; i < N; ++i) {
    result_ptr[i] = betweenness[i];
  }

  return result;
}

}  // anonymous namespace

inline at::Tensor betweenness_centrality(
    const at::Tensor& adjacency,
    bool normalized
) {
  // Handle batched input
  if (adjacency.dim() > 2) {
    int64_t batch_size = 1;
    for (int64_t i = 0; i < adjacency.dim() - 2; ++i) {
      batch_size *= adjacency.size(i);
    }
    int64_t N = adjacency.size(-1);

    at::Tensor flat_adj = adjacency.reshape({batch_size, N, N});
    std::vector<at::Tensor> results;

    for (int64_t b = 0; b < batch_size; ++b) {
      results.push_back(betweenness_centrality(flat_adj[b], normalized));
    }

    at::Tensor result = at::stack(results, 0);
    std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
    return result.reshape(out_shape);
  }

  TORCH_CHECK(adjacency.dim() == 2, "betweenness_centrality: adjacency must be 2D");
  TORCH_CHECK(adjacency.size(0) == adjacency.size(1), "betweenness_centrality: adjacency must be square");

  int64_t N = adjacency.size(0);
  if (N == 0) {
    return at::empty({0}, adjacency.options());
  }

  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  at::Tensor result;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "betweenness_centrality_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        result = betweenness_centrality_impl<scalar_t>(
            adj_ptr, N, normalized, dense_adj.options()
        );
      }
  );

  return result;
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("betweenness_centrality", &torchscience::cpu::graph_theory::betweenness_centrality);
}
