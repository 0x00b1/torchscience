#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_to_matrix.h"
#include "../../../kernel/geometry/transform/quaternion_to_matrix_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for quaternion to rotation matrix conversion.
 *
 * Converts a unit quaternion to a 3x3 rotation matrix.
 *
 * @param q Input quaternion tensor with shape (..., 4) in wxyz convention.
 * @return Rotation matrix tensor with shape (..., 3, 3).
 */
inline at::Tensor quaternion_to_matrix(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_to_matrix: q must have last dimension 4, got ", q.size(-1));

  auto q_contig = q.contiguous();

  // Output shape: (..., 3, 3) - replace last dim 4 with (3, 3)
  auto output_sizes = q_contig.sizes().vec();
  output_sizes.pop_back();  // Remove last dim (4)
  output_sizes.push_back(3);
  output_sizes.push_back(3);

  auto output = at::empty(output_sizes, q_contig.options());

  const int64_t num_quats = q_contig.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_to_matrix_cpu",
    [&] {
      const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_to_matrix_scalar(
            q_ptr + i * 4,
            output_ptr + i * 9
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU backward implementation for quaternion to matrix.
 *
 * Computes the gradient with respect to the input quaternion.
 *
 * @param grad_output Gradient of loss with respect to output, shape (..., 3, 3).
 * @param q Original input quaternion tensor with shape (..., 4).
 * @return Gradient with respect to input q, same shape as q.
 */
inline at::Tensor quaternion_to_matrix_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q
) {
  TORCH_CHECK(grad_output.size(-1) == 3 && grad_output.size(-2) == 3,
              "quaternion_to_matrix_backward: grad_output must have last two dimensions (3, 3)");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_to_matrix_backward: q must have last dimension 4");
  TORCH_CHECK(q.scalar_type() == grad_output.scalar_type(),
              "quaternion_to_matrix_backward: q and grad_output must have the same dtype");

  auto grad_output_contig = grad_output.contiguous();
  auto q_contig = q.contiguous();
  auto grad_q = at::empty_like(q);

  const int64_t num_quats = q.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_to_matrix_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
      scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_to_matrix_backward_scalar(
            grad_output_ptr + i * 9,
            q_ptr + i * 4,
            grad_q_ptr + i * 4
          );
        }
      });
    }
  );

  return grad_q;
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("quaternion_to_matrix", &torchscience::cpu::geometry::transform::quaternion_to_matrix);
  m.impl("quaternion_to_matrix_backward", &torchscience::cpu::geometry::transform::quaternion_to_matrix_backward);
}
