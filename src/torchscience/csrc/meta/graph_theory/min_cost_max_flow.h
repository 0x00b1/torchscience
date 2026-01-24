#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> min_cost_max_flow(
    const at::Tensor& capacity,
    const at::Tensor& cost,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "min_cost_max_flow: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "min_cost_max_flow: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );
  TORCH_CHECK(
      cost.dim() == 2,
      "min_cost_max_flow: cost must be 2D, got ", cost.dim(), "D"
  );
  TORCH_CHECK(
      cost.size(0) == cost.size(1),
      "min_cost_max_flow: cost must be square, got ",
      cost.size(0), " x ", cost.size(1)
  );
  TORCH_CHECK(
      capacity.size(0) == cost.size(0) && capacity.size(1) == cost.size(1),
      "min_cost_max_flow: capacity and cost must have same shape, got ",
      capacity.size(0), " x ", capacity.size(1), " vs ",
      cost.size(0), " x ", cost.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "min_cost_max_flow: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "min_cost_max_flow: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Return scalar max_flow, scalar total_cost, and NxN flow matrix with same dtype as input
  at::Tensor max_flow = at::empty({}, capacity.options());
  at::Tensor total_cost = at::empty({}, capacity.options());
  at::Tensor flow = at::empty({N, N}, capacity.options());

  return std::make_tuple(max_flow, total_cost, flow);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("min_cost_max_flow", &torchscience::meta::graph_theory::min_cost_max_flow);
}
