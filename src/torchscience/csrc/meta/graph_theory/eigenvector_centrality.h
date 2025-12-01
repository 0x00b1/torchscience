#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline at::Tensor eigenvector_centrality(const at::Tensor& adjacency) {
  TORCH_CHECK(adjacency.dim() >= 2, "eigenvector_centrality: adjacency must be at least 2D");
  TORCH_CHECK(adjacency.size(-1) == adjacency.size(-2), "eigenvector_centrality: adjacency must be square");

  std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
  return at::empty(out_shape, adjacency.options());
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("eigenvector_centrality", &torchscience::meta::graph_theory::eigenvector_centrality);
}
