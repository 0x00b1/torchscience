#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

inline at::Tensor katz_centrality(
    const at::Tensor& adjacency,
    double alpha,
    double beta,
    bool normalized
) {
  // Katz centrality: c = (I - alpha * A^T)^{-1} * beta * 1
  // Solved via direct matrix inversion

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
      results.push_back(katz_centrality(flat_adj[b], alpha, beta, normalized));
    }

    at::Tensor result = at::stack(results, 0);
    std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
    return result.reshape(out_shape);
  }

  TORCH_CHECK(adjacency.dim() == 2, "katz_centrality: adjacency must be 2D");
  TORCH_CHECK(adjacency.size(0) == adjacency.size(1), "katz_centrality: adjacency must be square");

  int64_t N = adjacency.size(0);
  if (N == 0) {
    return at::empty({0}, adjacency.options());
  }

  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;

  // Direct solve: c = (I - alpha * A^T)^{-1} * beta * ones
  at::Tensor I = at::eye(N, dense_adj.options());
  at::Tensor M = I - alpha * dense_adj.t();
  at::Tensor b_vec = beta * at::ones({N, 1}, dense_adj.options());

  // Solve M * c = b_vec
  at::Tensor c = at::linalg_solve(M, b_vec).squeeze(-1);

  if (normalized) {
    c = c / c.norm();
  }

  return c;
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("katz_centrality", &torchscience::cpu::graph_theory::katz_centrality);
}
