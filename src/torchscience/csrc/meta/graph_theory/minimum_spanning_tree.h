#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor> minimum_spanning_tree(
    const at::Tensor& adjacency
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "minimum_spanning_tree: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "minimum_spanning_tree: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);

  // total_weight is a scalar
  at::Tensor total_weight = at::empty({}, adjacency.options());

  // edges has shape (N-1, 2) for N >= 1, or (0, 2) for N == 0
  int64_t num_edges = N > 1 ? N - 1 : 0;
  at::Tensor edges = at::empty({num_edges, 2}, adjacency.options().dtype(at::kLong));

  return std::make_tuple(total_weight, edges);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("minimum_spanning_tree", &torchscience::meta::graph_theory::minimum_spanning_tree);
}
