#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for quaternion_slerp shape inference.
 */
inline at::Tensor quaternion_slerp(const at::Tensor& q1,
                                   const at::Tensor& q2,
                                   const at::Tensor& t) {
  TORCH_CHECK(q1.size(-1) == 4,
              "quaternion_slerp: q1 must have last dimension 4, got ",
              q1.size(-1));
  TORCH_CHECK(q2.size(-1) == 4,
              "quaternion_slerp: q2 must have last dimension 4, got ",
              q2.size(-1));

  // Get batch shapes
  auto batch1 = q1.sizes().slice(0, q1.dim() - 1);
  auto batch2 = q2.sizes().slice(0, q2.dim() - 1);
  auto t_shape = t.sizes();

  // Broadcast batch dimensions
  auto broadcast_shape = at::infer_size(batch1, batch2);
  broadcast_shape = at::infer_size(broadcast_shape, t_shape);

  std::vector<int64_t> output_shape(broadcast_shape.begin(),
                                    broadcast_shape.end());
  output_shape.push_back(4);

  return at::empty(output_shape, q1.options());
}

/**
 * Meta implementation for quaternion_slerp_backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> quaternion_slerp_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q1,
    const at::Tensor& q2,
    const at::Tensor& t) {
  return std::make_tuple(at::empty_like(q1), at::empty_like(q2),
                         at::empty_like(t));
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_slerp",
         &torchscience::meta::geometry::transform::quaternion_slerp);
  m.impl("quaternion_slerp_backward",
         &torchscience::meta::geometry::transform::quaternion_slerp_backward);
}
