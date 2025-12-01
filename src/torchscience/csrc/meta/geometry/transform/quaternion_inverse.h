#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for quaternion inverse shape inference.
 */
inline at::Tensor quaternion_inverse(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_inverse: q must have last dimension 4, got ", q.size(-1));

  return at::empty_like(q);
}

/**
 * Meta implementation for quaternion inverse backward shape inference.
 */
inline at::Tensor quaternion_inverse_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q
) {
  TORCH_CHECK(grad_output.size(-1) == 4, "quaternion_inverse_backward: grad_output must have last dimension 4");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_inverse_backward: q must have last dimension 4");

  return at::empty_like(q);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_inverse", &torchscience::meta::geometry::transform::quaternion_inverse);
  m.impl("quaternion_inverse_backward", &torchscience::meta::geometry::transform::quaternion_inverse_backward);
}
