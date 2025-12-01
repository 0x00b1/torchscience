#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/matrix_to_quaternion.h"
#include "../../../kernel/geometry/transform/matrix_to_quaternion_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for rotation matrix to quaternion conversion.
 *
 * Converts a 3x3 rotation matrix to a unit quaternion using Shepperd's method.
 *
 * @param matrix Input rotation matrix tensor with shape (..., 3, 3).
 * @return Quaternion tensor with shape (..., 4) in wxyz convention.
 */
inline at::Tensor matrix_to_quaternion(const at::Tensor& matrix) {
  TORCH_CHECK(matrix.size(-1) == 3 && matrix.size(-2) == 3,
              "matrix_to_quaternion: matrix must have last two dimensions (3, 3), got (",
              matrix.size(-2), ", ", matrix.size(-1), ")");

  auto matrix_contig = matrix.contiguous();

  // Output shape: (..., 4) - replace last two dims (3, 3) with (4)
  auto output_sizes = matrix_contig.sizes().vec();
  output_sizes.pop_back();  // Remove last dim (3)
  output_sizes.pop_back();  // Remove second to last dim (3)
  output_sizes.push_back(4);

  auto output = at::empty(output_sizes, matrix_contig.options());

  const int64_t num_matrices = matrix_contig.numel() / 9;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    matrix.scalar_type(),
    "matrix_to_quaternion_cpu",
    [&] {
      const scalar_t* matrix_ptr = matrix_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_matrices, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::matrix_to_quaternion_scalar(
            matrix_ptr + i * 9,
            output_ptr + i * 4
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU backward implementation for matrix to quaternion.
 *
 * Computes the gradient with respect to the input rotation matrix.
 *
 * @param grad_output Gradient of loss with respect to output, shape (..., 4).
 * @param matrix Original input rotation matrix tensor with shape (..., 3, 3).
 * @return Gradient with respect to input matrix, same shape as matrix.
 */
inline at::Tensor matrix_to_quaternion_backward(
    const at::Tensor& grad_output,
    const at::Tensor& matrix
) {
  TORCH_CHECK(grad_output.size(-1) == 4,
              "matrix_to_quaternion_backward: grad_output must have last dimension 4");
  TORCH_CHECK(matrix.size(-1) == 3 && matrix.size(-2) == 3,
              "matrix_to_quaternion_backward: matrix must have last two dimensions (3, 3)");
  TORCH_CHECK(matrix.scalar_type() == grad_output.scalar_type(),
              "matrix_to_quaternion_backward: matrix and grad_output must have the same dtype");

  auto grad_output_contig = grad_output.contiguous();
  auto matrix_contig = matrix.contiguous();
  auto grad_matrix = at::empty_like(matrix);

  const int64_t num_matrices = matrix.numel() / 9;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    matrix.scalar_type(),
    "matrix_to_quaternion_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* matrix_ptr = matrix_contig.data_ptr<scalar_t>();
      scalar_t* grad_matrix_ptr = grad_matrix.data_ptr<scalar_t>();

      at::parallel_for(0, num_matrices, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::matrix_to_quaternion_backward_scalar(
            grad_output_ptr + i * 4,
            matrix_ptr + i * 9,
            grad_matrix_ptr + i * 9
          );
        }
      });
    }
  );

  return grad_matrix;
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("matrix_to_quaternion", &torchscience::cpu::geometry::transform::matrix_to_quaternion);
  m.impl("matrix_to_quaternion_backward", &torchscience::cpu::geometry::transform::matrix_to_quaternion_backward);
}
