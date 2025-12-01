#pragma once

#include <limits>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

// Hopcroft-Karp algorithm for maximum bipartite matching
// Returns (matching_size, left_match, right_match)
template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> maximum_bipartite_matching_impl(
    const scalar_t* biadj,
    int64_t M,  // left partition size
    int64_t N,  // right partition size
    const at::TensorOptions& options
) {
  constexpr int64_t NIL = -1;
  constexpr int64_t INF = std::numeric_limits<int64_t>::max();

  // Build adjacency list for left nodes
  std::vector<std::vector<int64_t>> adj(M);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      if (biadj[i * N + j] != scalar_t(0)) {
        adj[i].push_back(j);
      }
    }
  }

  // Matching arrays
  // left_match[i] = j means left node i is matched to right node j
  // right_match[j] = i means right node j is matched to left node i
  std::vector<int64_t> left_match(M, NIL);
  std::vector<int64_t> right_match(N, NIL);

  // Distance array for BFS (used in Hopcroft-Karp)
  std::vector<int64_t> dist(M + 1);

  // BFS to find augmenting path layers
  auto bfs = [&]() -> bool {
    std::queue<int64_t> q;

    for (int64_t u = 0; u < M; ++u) {
      if (left_match[u] == NIL) {
        dist[u] = 0;
        q.push(u);
      } else {
        dist[u] = INF;
      }
    }
    dist[M] = INF;  // NIL node represented by index M

    while (!q.empty()) {
      int64_t u = q.front();
      q.pop();

      if (dist[u] < dist[M]) {
        for (int64_t v : adj[u]) {
          int64_t pair_v = right_match[v];
          int64_t pair_idx = (pair_v == NIL) ? M : pair_v;

          if (dist[pair_idx] == INF) {
            dist[pair_idx] = dist[u] + 1;
            if (pair_idx != M) {
              q.push(pair_idx);
            }
          }
        }
      }
    }

    return dist[M] != INF;
  };

  // DFS to find augmenting path
  std::function<bool(int64_t)> dfs = [&](int64_t u) -> bool {
    if (u == M) {  // NIL node
      return true;
    }

    for (int64_t v : adj[u]) {
      int64_t pair_v = right_match[v];
      int64_t pair_idx = (pair_v == NIL) ? M : pair_v;

      if (dist[pair_idx] == dist[u] + 1 && dfs(pair_idx)) {
        left_match[u] = v;
        right_match[v] = u;
        return true;
      }
    }

    dist[u] = INF;
    return false;
  };

  // Main Hopcroft-Karp loop
  int64_t matching_size = 0;
  while (bfs()) {
    for (int64_t u = 0; u < M; ++u) {
      if (left_match[u] == NIL && dfs(u)) {
        matching_size++;
      }
    }
  }

  // Create output tensors
  at::Tensor size_tensor = at::empty({}, options.dtype(at::kLong));
  size_tensor.fill_(matching_size);

  at::Tensor left_tensor = at::empty({M}, options.dtype(at::kLong));
  at::Tensor right_tensor = at::empty({N}, options.dtype(at::kLong));

  auto left_ptr = left_tensor.data_ptr<int64_t>();
  auto right_ptr = right_tensor.data_ptr<int64_t>();

  for (int64_t i = 0; i < M; ++i) {
    left_ptr[i] = left_match[i];
  }
  for (int64_t j = 0; j < N; ++j) {
    right_ptr[j] = right_match[j];
  }

  return std::make_tuple(size_tensor, left_tensor, right_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> maximum_bipartite_matching(
    const at::Tensor& biadjacency
) {
  TORCH_CHECK(
      biadjacency.dim() == 2,
      "maximum_bipartite_matching: biadjacency must be 2D, got ",
      biadjacency.dim(), "D"
  );

  int64_t M = biadjacency.size(0);
  int64_t N = biadjacency.size(1);

  // Handle sparse input
  at::Tensor dense_biadj = biadjacency.is_sparse() ? biadjacency.to_dense() : biadjacency;
  dense_biadj = dense_biadj.contiguous();

  // Handle empty graph
  if (M == 0 || N == 0) {
    return std::make_tuple(
        at::zeros({}, dense_biadj.options().dtype(at::kLong)),
        at::full({M}, -1, dense_biadj.options().dtype(at::kLong)),
        at::full({N}, -1, dense_biadj.options().dtype(at::kLong))
    );
  }

  at::Tensor matching_size, left_match, right_match;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_biadj.scalar_type(),
      "maximum_bipartite_matching_cpu",
      [&] {
        const scalar_t* biadj_ptr = dense_biadj.data_ptr<scalar_t>();
        std::tie(matching_size, left_match, right_match) =
            maximum_bipartite_matching_impl<scalar_t>(
                biadj_ptr, M, N, dense_biadj.options()
            );
      }
  );

  return std::make_tuple(matching_size, left_match, right_match);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("maximum_bipartite_matching",
         &torchscience::cpu::graph_theory::maximum_bipartite_matching);
}
