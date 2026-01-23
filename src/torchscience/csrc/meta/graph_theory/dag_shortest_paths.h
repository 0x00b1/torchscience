#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor> dag_shortest_paths(
    const at::Tensor& adjacency,
    int64_t source
) {
  TORCH_CHECK(
      adjacency.dim() == 2,
      "dag_shortest_paths: adjacency must be 2D, got ", adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(0) == adjacency.size(1),
      "dag_shortest_paths: adjacency must be square, got ",
      adjacency.size(0), " x ", adjacency.size(1)
  );

  int64_t N = adjacency.size(0);

  at::Tensor distances = at::empty({N}, adjacency.options());
  at::Tensor predecessors = at::empty({N}, adjacency.options().dtype(at::kLong));

  return std::make_tuple(distances, predecessors);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("dag_shortest_paths", &torchscience::meta::graph_theory::dag_shortest_paths);
}
