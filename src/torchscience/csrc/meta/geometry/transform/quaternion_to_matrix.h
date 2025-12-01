#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for quaternion to matrix shape inference.
 * Input shape: (..., 4)
 * Output shape: (..., 3, 3)
 */
inline at::Tensor quaternion_to_matrix(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_to_matrix: q must have last dimension 4, got ", q.size(-1));

  // Output shape: (..., 3, 3) - replace last dim 4 with (3, 3)
  auto output_sizes = q.sizes().vec();
  output_sizes.pop_back();  // Remove last dim (4)
  output_sizes.push_back(3);
  output_sizes.push_back(3);

  return at::empty(output_sizes, q.options());
}

/**
 * Meta implementation for quaternion to matrix backward shape inference.
 */
inline at::Tensor quaternion_to_matrix_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q
) {
  TORCH_CHECK(grad_output.size(-1) == 3 && grad_output.size(-2) == 3,
              "quaternion_to_matrix_backward: grad_output must have last two dimensions (3, 3)");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_to_matrix_backward: q must have last dimension 4");

  return at::empty_like(q);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("quaternion_to_matrix", &torchscience::meta::geometry::transform::quaternion_to_matrix);
  m.impl("quaternion_to_matrix_backward", &torchscience::meta::geometry::transform::quaternion_to_matrix_backward);
}
