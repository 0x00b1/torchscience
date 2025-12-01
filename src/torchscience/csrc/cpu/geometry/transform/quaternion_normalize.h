#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_normalize.h"
#include "../../../kernel/geometry/transform/quaternion_normalize_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for quaternion normalize.
 *
 * Normalizes a quaternion to unit length: output = q / ||q||.
 *
 * @param q Input quaternion tensor with shape (..., 4) in wxyz convention.
 * @return Normalized quaternion tensor with the same shape as input.
 */
inline at::Tensor quaternion_normalize(const at::Tensor& q) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_normalize: q must have last dimension 4, got ", q.size(-1));

  auto q_contig = q.contiguous();
  auto output = at::empty_like(q_contig);

  const int64_t num_quats = q_contig.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_normalize_cpu",
    [&] {
      const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_normalize_scalar(
            q_ptr + i * 4,
            output_ptr + i * 4
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU backward implementation for quaternion normalize.
 *
 * Computes the gradient with respect to the input quaternion.
 *
 * @param grad_output Gradient of loss with respect to output, shape (..., 4).
 * @param q Original input quaternion tensor with shape (..., 4).
 * @return Gradient with respect to input q, same shape as q.
 */
inline at::Tensor quaternion_normalize_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q
) {
  TORCH_CHECK(grad_output.size(-1) == 4, "quaternion_normalize_backward: grad_output must have last dimension 4");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_normalize_backward: q must have last dimension 4");
  TORCH_CHECK(q.scalar_type() == grad_output.scalar_type(),
              "quaternion_normalize_backward: q and grad_output must have the same dtype");

  auto grad_output_contig = grad_output.contiguous();
  auto q_contig = q.contiguous();
  auto grad_q = at::empty_like(q);

  const int64_t num_quats = q.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_normalize_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* q_ptr = q_contig.data_ptr<scalar_t>();
      scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_normalize_backward_scalar(
            grad_output_ptr + i * 4,
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
  m.impl("quaternion_normalize", &torchscience::cpu::geometry::transform::quaternion_normalize);
  m.impl("quaternion_normalize_backward", &torchscience::cpu::geometry::transform::quaternion_normalize_backward);
}
