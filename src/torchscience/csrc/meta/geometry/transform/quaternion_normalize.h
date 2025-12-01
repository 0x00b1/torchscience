#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for quaternion normalize shape inference.
 */
inline at::Tensor quaternion_normalize(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_normalize: q must have last dimension 4, got ", q.size(-1));

  return at::empty_like(q);
}

/**
 * Meta implementation for quaternion normalize backward shape inference.
 */
inline at::Tensor quaternion_normalize_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q
) {
  TORCH_CHECK(grad_output.size(-1) == 4, "quaternion_normalize_backward: grad_output must have last dimension 4");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_normalize_backward: q must have last dimension 4");

  return at::empty_like(q);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_normalize", &torchscience::meta::geometry::transform::quaternion_normalize);
  m.impl("quaternion_normalize_backward", &torchscience::meta::geometry::transform::quaternion_normalize_backward);
}
