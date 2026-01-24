#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor> push_relabel(
    const at::Tensor& capacity,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "push_relabel: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "push_relabel: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "push_relabel: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "push_relabel: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Return scalar max_flow and NxN flow matrix with same dtype as input
  at::Tensor max_flow = at::empty({}, capacity.options());
  at::Tensor flow = at::empty({N, N}, capacity.options());

  return std::make_tuple(max_flow, flow);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("push_relabel", &torchscience::meta::graph_theory::push_relabel);
}
