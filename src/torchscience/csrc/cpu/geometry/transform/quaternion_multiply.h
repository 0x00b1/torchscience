#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_multiply.h"
#include "../../../kernel/geometry/transform/quaternion_multiply_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for quaternion multiplication (Hamilton product).
 * Supports broadcasting of batch dimensions.
 */
inline at::Tensor quaternion_multiply(const at::Tensor& q1, const at::Tensor& q2) {
  TORCH_CHECK(q1.size(-1) == 4, "quaternion_multiply: q1 must have last dimension 4, got ", q1.size(-1));
  TORCH_CHECK(q2.size(-1) == 4, "quaternion_multiply: q2 must have last dimension 4, got ", q2.size(-1));
  TORCH_CHECK(q1.scalar_type() == q2.scalar_type(),
              "quaternion_multiply: q1 and q2 must have the same dtype");

  // Broadcast batch dimensions
  auto batch1 = q1.sizes().slice(0, q1.dim() - 1);
  auto batch2 = q2.sizes().slice(0, q2.dim() - 1);
  auto broadcast_shape = at::infer_size(batch1, batch2);

  // Expand inputs to broadcast shape
  std::vector<int64_t> full_shape1(broadcast_shape.begin(), broadcast_shape.end());
  full_shape1.push_back(4);
  std::vector<int64_t> full_shape2(broadcast_shape.begin(), broadcast_shape.end());
  full_shape2.push_back(4);

  auto q1_expanded = q1.expand(full_shape1).contiguous();
  auto q2_expanded = q2.expand(full_shape2).contiguous();

  auto output = at::empty_like(q1_expanded);

  const int64_t num_quats = q1_expanded.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q1.scalar_type(),
    "quaternion_multiply_cpu",
    [&] {
      const scalar_t* q1_ptr = q1_expanded.data_ptr<scalar_t>();
      const scalar_t* q2_ptr = q2_expanded.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_multiply_scalar(
            q1_ptr + i * 4,
            q2_ptr + i * 4,
            output_ptr + i * 4
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU backward implementation for quaternion multiplication.
 * Handles broadcasting: gradients are reduced along broadcast dimensions.
 */
inline std::tuple<at::Tensor, at::Tensor> quaternion_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q1,
    const at::Tensor& q2
) {
  TORCH_CHECK(grad_output.size(-1) == 4, "quaternion_multiply_backward: grad_output must have last dimension 4");
  TORCH_CHECK(q1.size(-1) == 4, "quaternion_multiply_backward: q1 must have last dimension 4");
  TORCH_CHECK(q2.size(-1) == 4, "quaternion_multiply_backward: q2 must have last dimension 4");
  TORCH_CHECK(q1.scalar_type() == q2.scalar_type() && q1.scalar_type() == grad_output.scalar_type(),
              "quaternion_multiply_backward: all inputs must have the same dtype");

  // Get original batch shapes
  auto batch1 = q1.sizes().slice(0, q1.dim() - 1);
  auto batch2 = q2.sizes().slice(0, q2.dim() - 1);
  auto broadcast_shape = at::infer_size(batch1, batch2);

  // Expand inputs to broadcast shape (same as forward pass)
  std::vector<int64_t> full_shape1(broadcast_shape.begin(), broadcast_shape.end());
  full_shape1.push_back(4);
  std::vector<int64_t> full_shape2(broadcast_shape.begin(), broadcast_shape.end());
  full_shape2.push_back(4);

  auto q1_expanded = q1.expand(full_shape1).contiguous();
  auto q2_expanded = q2.expand(full_shape2).contiguous();
  auto grad_output_contig = grad_output.contiguous();

  // Compute gradients in broadcast shape
  std::vector<int64_t> grad_shape(broadcast_shape.begin(), broadcast_shape.end());
  grad_shape.push_back(4);
  auto grad_q1_full = at::empty(grad_shape, q1.options());
  auto grad_q2_full = at::empty(grad_shape, q2.options());

  const int64_t num_quats = q1_expanded.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q1.scalar_type(),
    "quaternion_multiply_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* q1_ptr = q1_expanded.data_ptr<scalar_t>();
      const scalar_t* q2_ptr = q2_expanded.data_ptr<scalar_t>();
      scalar_t* grad_q1_ptr = grad_q1_full.data_ptr<scalar_t>();
      scalar_t* grad_q2_ptr = grad_q2_full.data_ptr<scalar_t>();

      at::parallel_for(0, num_quats, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_multiply_backward_scalar(
            grad_output_ptr + i * 4,
            q1_ptr + i * 4,
            q2_ptr + i * 4,
            grad_q1_ptr + i * 4,
            grad_q2_ptr + i * 4
          );
        }
      });
    }
  );

  // Reduce gradients along broadcast dimensions
  auto grad_q1 = grad_q1_full;
  auto grad_q2 = grad_q2_full;

  // Sum along dimensions that were broadcast in q1
  std::vector<int64_t> reduce_dims_q1;
  int64_t offset = broadcast_shape.size() - batch1.size();
  for (int64_t i = 0; i < static_cast<int64_t>(broadcast_shape.size()); ++i) {
    if (i < offset || batch1[i - offset] == 1) {
      if (i < offset || (i >= offset && broadcast_shape[i] != 1)) {
        reduce_dims_q1.push_back(i);
      }
    }
  }
  if (!reduce_dims_q1.empty()) {
    grad_q1 = grad_q1_full.sum(reduce_dims_q1, /*keepdim=*/false);
  }
  // Reshape to match original q1 shape
  grad_q1 = grad_q1.reshape(q1.sizes());

  // Sum along dimensions that were broadcast in q2
  std::vector<int64_t> reduce_dims_q2;
  offset = broadcast_shape.size() - batch2.size();
  for (int64_t i = 0; i < static_cast<int64_t>(broadcast_shape.size()); ++i) {
    if (i < offset || batch2[i - offset] == 1) {
      if (i < offset || (i >= offset && broadcast_shape[i] != 1)) {
        reduce_dims_q2.push_back(i);
      }
    }
  }
  if (!reduce_dims_q2.empty()) {
    grad_q2 = grad_q2_full.sum(reduce_dims_q2, /*keepdim=*/false);
  }
  // Reshape to match original q2 shape
  grad_q2 = grad_q2.reshape(q2.sizes());

  return std::make_tuple(grad_q1, grad_q2);
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("quaternion_multiply", &torchscience::cpu::geometry::transform::quaternion_multiply);
  m.impl("quaternion_multiply_backward", &torchscience::cpu::geometry::transform::quaternion_multiply_backward);
}
