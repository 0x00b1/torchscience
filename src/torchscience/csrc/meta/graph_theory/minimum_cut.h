#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> minimum_cut(
    const at::Tensor& capacity,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "minimum_cut: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "minimum_cut: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "minimum_cut: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "minimum_cut: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Return scalar cut_value with same dtype as input,
  // boolean reachable tensor of shape (N,), and
  // int64 cut_edges tensor of shape (?, 2) - unknown number of cut edges
  // For meta tensors, we use a symbolic size for the first dimension
  at::Tensor cut_value = at::empty({}, capacity.options());
  at::Tensor reachable = at::empty({N}, capacity.options().dtype(at::kBool));
  // For cut_edges, worst case is all possible edges from source side
  // Use N*N as an upper bound for shape inference purposes
  // In practice the actual number will be smaller
  at::Tensor cut_edges = at::empty({N * N, 2}, capacity.options().dtype(at::kLong));

  return std::make_tuple(cut_value, reachable, cut_edges);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("minimum_cut", &torchscience::meta::graph_theory::minimum_cut);
}
