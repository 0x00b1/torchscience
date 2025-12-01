#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline at::Tensor closeness_centrality(
    const at::Tensor& adjacency,
    bool normalized
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "closeness_centrality: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-1) == adjacency.size(-2),
      "closeness_centrality: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );

  // Output shape is adjacency shape without last dimension
  std::vector<int64_t> out_shape(adjacency.sizes().begin(), adjacency.sizes().end() - 1);

  return at::empty(out_shape, adjacency.options());
}

inline at::Tensor closeness_centrality_backward(
    const at::Tensor& grad,
    const at::Tensor& adjacency,
    const at::Tensor& distances,
    bool normalized
) {
  return at::empty_like(adjacency);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("closeness_centrality", &torchscience::meta::graph_theory::closeness_centrality);
  m.impl("closeness_centrality_backward", &torchscience::meta::graph_theory::closeness_centrality_backward);
}
