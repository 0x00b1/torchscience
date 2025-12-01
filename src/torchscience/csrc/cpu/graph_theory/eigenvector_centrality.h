#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

inline at::Tensor eigenvector_centrality(const at::Tensor& adjacency) {
  // Eigenvector centrality: the eigenvector corresponding to the largest
  // eigenvalue of the adjacency matrix

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
      results.push_back(eigenvector_centrality(flat_adj[b]));
    }

    at::Tensor result = at::stack(results, 0);
    std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
    return result.reshape(out_shape);
  }

  TORCH_CHECK(adjacency.dim() == 2, "eigenvector_centrality: adjacency must be 2D");
  TORCH_CHECK(adjacency.size(0) == adjacency.size(1), "eigenvector_centrality: adjacency must be square");

  int64_t N = adjacency.size(0);
  if (N == 0) {
    return at::empty({0}, adjacency.options());
  }

  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;

  // Use eigendecomposition for symmetric matrices
  // eigh returns eigenvalues in ascending order, so last eigenvector is for largest eigenvalue
  auto [eigenvalues, eigenvectors] = at::linalg_eigh(dense_adj);

  // Take eigenvector corresponding to largest eigenvalue (last column)
  at::Tensor max_eigenvec = eigenvectors.select(-1, N - 1);

  // Ensure positive values (eigenvector can be negated)
  // Take absolute value to ensure non-negative centrality
  max_eigenvec = max_eigenvec.abs();

  // Normalize to unit norm
  max_eigenvec = max_eigenvec / max_eigenvec.norm();

  return max_eigenvec;
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("eigenvector_centrality", &torchscience::cpu::graph_theory::eigenvector_centrality);
}
