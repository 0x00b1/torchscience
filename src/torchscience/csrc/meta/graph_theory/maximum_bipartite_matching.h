#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> maximum_bipartite_matching(
    const at::Tensor& biadjacency
) {
  TORCH_CHECK(
      biadjacency.dim() == 2,
      "maximum_bipartite_matching: biadjacency must be 2D, got ",
      biadjacency.dim(), "D"
  );

  int64_t M = biadjacency.size(0);
  int64_t N = biadjacency.size(1);

  // matching_size is a scalar
  at::Tensor matching_size = at::empty({}, biadjacency.options().dtype(at::kLong));

  // left_match has shape (M,), right_match has shape (N,)
  at::Tensor left_match = at::empty({M}, biadjacency.options().dtype(at::kLong));
  at::Tensor right_match = at::empty({N}, biadjacency.options().dtype(at::kLong));

  return std::make_tuple(matching_size, left_match, right_match);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("maximum_bipartite_matching",
         &torchscience::meta::graph_theory::maximum_bipartite_matching);
}
