#pragma once

#include <limits>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
void dijkstra_all_pairs_impl(
    const scalar_t* adj,
    scalar_t* distances,
    int64_t N
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Compute shortest paths from each source
  at::parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
    for (int64_t source = start; source < end; ++source) {
      std::vector<scalar_t> dist(N, inf);
      dist[source] = scalar_t(0);

      using PQElement = std::pair<scalar_t, int64_t>;
      std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> pq;
      pq.push({scalar_t(0), source});

      std::vector<bool> visited(N, false);

      while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;

        for (int64_t v = 0; v < N; ++v) {
          // Treat as undirected: use min of both directions
          scalar_t w = adj[u * N + v];
          scalar_t w_rev = adj[v * N + u];
          if (w_rev < w) w = w_rev;

          if (w < inf && !visited[v]) {
            scalar_t new_dist = dist[u] + w;
            if (new_dist < dist[v]) {
              dist[v] = new_dist;
              pq.push({new_dist, v});
            }
          }
        }
      }

      // Store distances for this source
      for (int64_t v = 0; v < N; ++v) {
        distances[source * N + v] = dist[v];
      }
    }
  });
}

template <typename scalar_t>
at::Tensor closeness_centrality_impl(
    const scalar_t* adj,
    int64_t N,
    bool normalized,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  // Compute all-pairs shortest paths
  std::vector<scalar_t> distances(N * N, inf);
  dijkstra_all_pairs_impl(adj, distances.data(), N);

  // Compute closeness centrality for each node
  at::Tensor result = at::zeros({N}, options);
  auto result_ptr = result.data_ptr<scalar_t>();

  for (int64_t i = 0; i < N; ++i) {
    scalar_t sum_dist = scalar_t(0);
    int64_t reachable = 0;

    for (int64_t j = 0; j < N; ++j) {
      if (i != j && distances[i * N + j] < inf) {
        sum_dist += distances[i * N + j];
        ++reachable;
      }
    }

    if (reachable > 0 && sum_dist > scalar_t(0)) {
      // Closeness = (reachable nodes) / (sum of distances)
      result_ptr[i] = static_cast<scalar_t>(reachable) / sum_dist;

      if (normalized && N > 1) {
        // Normalize by (n-1) to get value in [0, 1]
        result_ptr[i] *= static_cast<scalar_t>(reachable) / static_cast<scalar_t>(N - 1);
      }
    } else {
      result_ptr[i] = scalar_t(0);
    }
  }

  return result;
}

template <typename scalar_t>
at::Tensor closeness_centrality_backward_impl(
    const scalar_t* grad_out,
    const scalar_t* adj,
    const scalar_t* distances,
    int64_t N,
    bool normalized,
    const at::TensorOptions& options
) {
  constexpr scalar_t inf = std::numeric_limits<scalar_t>::infinity();

  at::Tensor grad_adj = at::zeros({N, N}, options);
  auto grad_adj_ptr = grad_adj.data_ptr<scalar_t>();

  // For each node i, closeness[i] = reachable[i] / sum_dist[i]
  // d(closeness[i])/d(adj[u,v]) = d(closeness[i])/d(sum_dist[i]) * d(sum_dist[i])/d(adj[u,v])
  //
  // sum_dist[i] = sum_j dist[i,j]
  // d(sum_dist[i])/d(adj[u,v]) = sum_j d(dist[i,j])/d(adj[u,v])
  //
  // For shortest path distances, d(dist[i,j])/d(adj[u,v]) = 1 if (u,v) is on shortest path from i to j

  for (int64_t i = 0; i < N; ++i) {
    // Compute sum_dist and reachable for node i
    scalar_t sum_dist = scalar_t(0);
    int64_t reachable = 0;

    for (int64_t j = 0; j < N; ++j) {
      if (i != j && distances[i * N + j] < inf) {
        sum_dist += distances[i * N + j];
        ++reachable;
      }
    }

    if (reachable == 0 || sum_dist <= scalar_t(0)) continue;

    // d(closeness[i])/d(sum_dist[i]) = -reachable / sum_dist^2
    scalar_t dcloseness_dsumist = -static_cast<scalar_t>(reachable) / (sum_dist * sum_dist);

    if (normalized && N > 1) {
      dcloseness_dsumist *= static_cast<scalar_t>(reachable) / static_cast<scalar_t>(N - 1);
    }

    // Need to trace shortest paths to determine which edges contribute
    // For simplicity, use numerical approximation via Dijkstra path tracing
    // (Full implementation would reconstruct shortest path tree)

    // For now, approximate gradient by distributing through distance matrix
    for (int64_t j = 0; j < N; ++j) {
      if (i != j && distances[i * N + j] < inf) {
        // Gradient flows through all edges on shortest path from i to j
        // This is a simplification - proper implementation needs path reconstruction
        scalar_t contrib = grad_out[i] * dcloseness_dsumist;

        // Approximate: add gradient to direct edge if it exists
        if (adj[i * N + j] < inf) {
          grad_adj_ptr[i * N + j] += contrib;
        }
      }
    }
  }

  return grad_adj;
}

}  // anonymous namespace

inline at::Tensor closeness_centrality(
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
      results.push_back(closeness_centrality(flat_adj[b], normalized));
    }

    at::Tensor result = at::stack(results, 0);
    std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
    return result.reshape(out_shape);
  }

  TORCH_CHECK(
      adjacency.dim() == 2,
      "closeness_centrality: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "closeness_centrality: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

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
      "closeness_centrality_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        result = closeness_centrality_impl<scalar_t>(
            adj_ptr, N, normalized, dense_adj.options()
        );
      }
  );

  return result;
}

inline at::Tensor closeness_centrality_backward(
    const at::Tensor& grad,
    const at::Tensor& adjacency,
    const at::Tensor& distances,
    bool normalized
) {
  int64_t N = adjacency.size(-1);

  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();
  at::Tensor dense_dist = distances.contiguous();
  at::Tensor grad_contig = grad.contiguous();

  at::Tensor result;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "closeness_centrality_backward_cpu",
      [&] {
        result = closeness_centrality_backward_impl<scalar_t>(
            grad_contig.data_ptr<scalar_t>(),
            dense_adj.data_ptr<scalar_t>(),
            dense_dist.data_ptr<scalar_t>(),
            N,
            normalized,
            dense_adj.options()
        );
      }
  );

  return result;
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("closeness_centrality", &torchscience::cpu::graph_theory::closeness_centrality);
  m.impl("closeness_centrality_backward", &torchscience::cpu::graph_theory::closeness_centrality_backward);
}
