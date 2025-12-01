#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for matrix to quaternion shape inference.
 * Input shape: (..., 3, 3)
 * Output shape: (..., 4)
 */
inline at::Tensor matrix_to_quaternion(const at::Tensor& matrix) {
  TORCH_CHECK(matrix.size(-1) == 3 && matrix.size(-2) == 3,
              "matrix_to_quaternion: matrix must have last two dimensions (3, 3), got (",
              matrix.size(-2), ", ", matrix.size(-1), ")");

  // Output shape: (..., 4) - replace last two dims (3, 3) with (4)
  auto output_sizes = matrix.sizes().vec();
  output_sizes.pop_back();  // Remove last dim (3)
  output_sizes.pop_back();  // Remove second to last dim (3)
  output_sizes.push_back(4);

  return at::empty(output_sizes, matrix.options());
}

/**
 * Meta implementation for matrix to quaternion backward shape inference.
 */
inline at::Tensor matrix_to_quaternion_backward(
    const at::Tensor& grad_output,
    const at::Tensor& matrix
) {
  TORCH_CHECK(grad_output.size(-1) == 4,
              "matrix_to_quaternion_backward: grad_output must have last dimension 4");
  TORCH_CHECK(matrix.size(-1) == 3 && matrix.size(-2) == 3,
              "matrix_to_quaternion_backward: matrix must have last two dimensions (3, 3)");

  return at::empty_like(matrix);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("matrix_to_quaternion", &torchscience::meta::geometry::transform::matrix_to_quaternion);
  m.impl("matrix_to_quaternion_backward", &torchscience::meta::geometry::transform::matrix_to_quaternion_backward);
}
