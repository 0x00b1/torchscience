#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline at::Tensor katz_centrality(
    const at::Tensor& adjacency,
    double alpha,
    double beta,
    bool normalized
) {
  TORCH_CHECK(adjacency.dim() >= 2, "katz_centrality: adjacency must be at least 2D");
  TORCH_CHECK(adjacency.size(-1) == adjacency.size(-2), "katz_centrality: adjacency must be square");

  std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);
  return at::empty(out_shape, adjacency.options());
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("katz_centrality", &torchscience::meta::graph_theory::katz_centrality);
}
