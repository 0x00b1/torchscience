#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline at::Tensor topological_sort(const at::Tensor& adjacency) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "topological_sort: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "topological_sort: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );

  int64_t N = adjacency.size(-1);

  // Build output shape: batch dims + N
  std::vector<int64_t> result_sizes;
  for (int64_t i = 0; i < adjacency.dim() - 2; ++i) {
    result_sizes.push_back(adjacency.size(i));
  }
  result_sizes.push_back(N);

  return at::empty(result_sizes, adjacency.options().dtype(at::kLong));
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("topological_sort", &torchscience::meta::graph_theory::topological_sort);
}
