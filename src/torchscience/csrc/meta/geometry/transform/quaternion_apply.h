#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for quaternion_apply shape inference.
 */
inline at::Tensor quaternion_apply(const at::Tensor& q, const at::Tensor& point) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_apply: q must have last dimension 4, got ", q.size(-1));
  TORCH_CHECK(point.size(-1) == 3, "quaternion_apply: point must have last dimension 3, got ", point.size(-1));

  // Infer broadcast shape for batch dimensions
  auto batch_q = q.sizes().slice(0, q.dim() - 1);
  auto batch_point = point.sizes().slice(0, point.dim() - 1);
  auto broadcast_shape = at::infer_size(batch_q, batch_point);

  std::vector<int64_t> output_shape(broadcast_shape.begin(), broadcast_shape.end());
  output_shape.push_back(3);

  return at::empty(output_shape, point.options());
}

/**
 * Meta implementation for quaternion_apply backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor> quaternion_apply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q,
    const at::Tensor& point
) {
  return std::make_tuple(at::empty_like(q), at::empty_like(point));
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_apply", &torchscience::meta::geometry::transform::quaternion_apply);
  m.impl("quaternion_apply_backward", &torchscience::meta::geometry::transform::quaternion_apply_backward);
}
