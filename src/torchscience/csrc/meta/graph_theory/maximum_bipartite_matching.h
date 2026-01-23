#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> maximum_bipartite_matching(
    const at::Tensor& biadjacency
) {
  TORCH_CHECK(
      biadjacency.dim() >= 2,
      "maximum_bipartite_matching: biadjacency must be at least 2D, got ",
      biadjacency.dim(), "D"
  );

  int64_t M = biadjacency.size(-2);
  int64_t N = biadjacency.size(-1);

  // Build output shapes with batch dimensions
  std::vector<int64_t> batch_dims;
  for (int64_t i = 0; i < biadjacency.dim() - 2; ++i) {
    batch_dims.push_back(biadjacency.size(i));
  }

  std::vector<int64_t> size_shape = batch_dims;  // (*,)
  std::vector<int64_t> left_shape = batch_dims;
  left_shape.push_back(M);  // (*, M)
  std::vector<int64_t> right_shape = batch_dims;
  right_shape.push_back(N);  // (*, N)

  // matching_size has shape (*,)
  at::Tensor matching_size = at::empty(size_shape, biadjacency.options().dtype(at::kLong));

  // left_match has shape (*, M), right_match has shape (*, N)
  at::Tensor left_match = at::empty(left_shape, biadjacency.options().dtype(at::kLong));
  at::Tensor right_match = at::empty(right_shape, biadjacency.options().dtype(at::kLong));

  return std::make_tuple(matching_size, left_match, right_match);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("maximum_bipartite_matching",
         &torchscience::meta::graph_theory::maximum_bipartite_matching);
}
