#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, bool> bellman_ford(
    const at::Tensor& adjacency,
    int64_t source,
    bool directed
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "bellman_ford: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "bellman_ford: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);

  at::Tensor distances = at::empty({N}, adjacency.options());
  at::Tensor predecessors = at::empty({N}, adjacency.options().dtype(at::kLong));

  // Return false for negative cycle (unknown at meta time)
  return std::make_tuple(distances, predecessors, false);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("bellman_ford", &torchscience::meta::graph_theory::bellman_ford);
}
