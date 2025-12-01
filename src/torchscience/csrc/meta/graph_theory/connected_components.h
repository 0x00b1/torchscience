#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<int64_t, at::Tensor> connected_components(
    const at::Tensor& adjacency,
    bool directed,
    c10::string_view connection
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "connected_components: adjacency must be at least 2D, got ",
      adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-1) == adjacency.size(-2),
      "connected_components: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );

  // Labels have shape (N,) for 2D input
  // Build the sizes vector for labels: all dims except the last
  std::vector<int64_t> labels_sizes;
  for (int64_t i = 0; i < adjacency.dim() - 1; ++i) {
    labels_sizes.push_back(adjacency.size(i));
  }

  at::Tensor labels = at::empty(
      labels_sizes,
      adjacency.options().dtype(at::kLong)
  );

  // n_components is unknown at meta time, return 0 as placeholder
  return std::make_tuple(0, labels);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("connected_components", &torchscience::meta::graph_theory::connected_components);
}
