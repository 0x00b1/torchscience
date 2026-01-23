#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor> breadth_first_search(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "breadth_first_search: adjacency must be at least 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-2) == adjacency.size(-1),
      "breadth_first_search: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );

  int64_t N = adjacency.size(-1);

  TORCH_CHECK(
      source >= 0 && source < N,
      "breadth_first_search: source must be in range [0, ", N, "), got ", source
  );

  // Build output shape: batch dims + N
  std::vector<int64_t> result_sizes;
  for (int64_t i = 0; i < adjacency.dim() - 2; ++i) {
    result_sizes.push_back(adjacency.size(i));
  }
  result_sizes.push_back(N);

  auto dist = at::empty(result_sizes, adjacency.options().dtype(at::kLong));
  auto pred = at::empty(result_sizes, adjacency.options().dtype(at::kLong));

  return std::make_tuple(dist, pred);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("breadth_first_search", &torchscience::meta::graph_theory::breadth_first_search);
}
