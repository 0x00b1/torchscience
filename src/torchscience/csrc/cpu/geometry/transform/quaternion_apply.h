#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/quaternion_apply.h"
#include "../../../kernel/geometry/transform/quaternion_apply_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for quaternion_apply.
 * Rotates 3D points by quaternions using the optimized formula.
 * Supports broadcasting of batch dimensions.
 *
 * @param q Input quaternion tensor with shape (..., 4) in wxyz convention.
 * @param point Input 3D point tensor with shape (..., 3).
 * @return Rotated point tensor with broadcast shape (..., 3).
 */
inline at::Tensor quaternion_apply(const at::Tensor& q, const at::Tensor& point) {
  TORCH_CHECK(q.size(-1) == 4, "quaternion_apply: q must have last dimension 4, got ", q.size(-1));
  TORCH_CHECK(point.size(-1) == 3, "quaternion_apply: point must have last dimension 3, got ", point.size(-1));
  TORCH_CHECK(q.scalar_type() == point.scalar_type(),
              "quaternion_apply: q and point must have the same dtype");

  // Broadcast batch dimensions
  auto batch_q = q.sizes().slice(0, q.dim() - 1);
  auto batch_point = point.sizes().slice(0, point.dim() - 1);
  auto broadcast_shape = at::infer_size(batch_q, batch_point);

  // Expand inputs to broadcast shape
  std::vector<int64_t> full_shape_q(broadcast_shape.begin(), broadcast_shape.end());
  full_shape_q.push_back(4);
  std::vector<int64_t> full_shape_point(broadcast_shape.begin(), broadcast_shape.end());
  full_shape_point.push_back(3);

  auto q_expanded = q.expand(full_shape_q).contiguous();
  auto point_expanded = point.expand(full_shape_point).contiguous();

  // Output has shape (..., 3)
  std::vector<int64_t> output_shape(broadcast_shape.begin(), broadcast_shape.end());
  output_shape.push_back(3);
  auto output = at::empty(output_shape, point.options());

  const int64_t num_elements = q_expanded.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_apply_cpu",
    [&] {
      const scalar_t* q_ptr = q_expanded.data_ptr<scalar_t>();
      const scalar_t* point_ptr = point_expanded.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_apply_scalar(
            q_ptr + i * 4,
            point_ptr + i * 3,
            output_ptr + i * 3
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU backward implementation for quaternion_apply.
 * Handles broadcasting: gradients are reduced along broadcast dimensions.
 *
 * @param grad_output Gradient of loss with respect to output, shape (..., 3).
 * @param q Original input quaternion tensor with shape (..., 4).
 * @param point Original input point tensor with shape (..., 3).
 * @return Tuple of gradients (grad_q, grad_point).
 */
inline std::tuple<at::Tensor, at::Tensor> quaternion_apply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& q,
    const at::Tensor& point
) {
  TORCH_CHECK(grad_output.size(-1) == 3, "quaternion_apply_backward: grad_output must have last dimension 3");
  TORCH_CHECK(q.size(-1) == 4, "quaternion_apply_backward: q must have last dimension 4");
  TORCH_CHECK(point.size(-1) == 3, "quaternion_apply_backward: point must have last dimension 3");
  TORCH_CHECK(q.scalar_type() == point.scalar_type() && q.scalar_type() == grad_output.scalar_type(),
              "quaternion_apply_backward: all inputs must have the same dtype");

  // Get original batch shapes
  auto batch_q = q.sizes().slice(0, q.dim() - 1);
  auto batch_point = point.sizes().slice(0, point.dim() - 1);
  auto broadcast_shape = at::infer_size(batch_q, batch_point);

  // Expand inputs to broadcast shape (same as forward pass)
  std::vector<int64_t> full_shape_q(broadcast_shape.begin(), broadcast_shape.end());
  full_shape_q.push_back(4);
  std::vector<int64_t> full_shape_point(broadcast_shape.begin(), broadcast_shape.end());
  full_shape_point.push_back(3);

  auto q_expanded = q.expand(full_shape_q).contiguous();
  auto point_expanded = point.expand(full_shape_point).contiguous();
  auto grad_output_contig = grad_output.contiguous();

  // Compute gradients in broadcast shape
  std::vector<int64_t> grad_q_shape(broadcast_shape.begin(), broadcast_shape.end());
  grad_q_shape.push_back(4);
  std::vector<int64_t> grad_point_shape(broadcast_shape.begin(), broadcast_shape.end());
  grad_point_shape.push_back(3);

  auto grad_q_full = at::empty(grad_q_shape, q.options());
  auto grad_point_full = at::empty(grad_point_shape, point.options());

  const int64_t num_elements = q_expanded.numel() / 4;

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    q.scalar_type(),
    "quaternion_apply_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* q_ptr = q_expanded.data_ptr<scalar_t>();
      const scalar_t* point_ptr = point_expanded.data_ptr<scalar_t>();
      scalar_t* grad_q_ptr = grad_q_full.data_ptr<scalar_t>();
      scalar_t* grad_point_ptr = grad_point_full.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::quaternion_apply_backward_scalar(
            grad_output_ptr + i * 3,
            q_ptr + i * 4,
            point_ptr + i * 3,
            grad_q_ptr + i * 4,
            grad_point_ptr + i * 3
          );
        }
      });
    }
  );

  // Reduce gradients along broadcast dimensions
  auto grad_q = grad_q_full;
  auto grad_point = grad_point_full;

  // Sum along dimensions that were broadcast in q
  std::vector<int64_t> reduce_dims_q;
  int64_t offset = broadcast_shape.size() - batch_q.size();
  for (int64_t i = 0; i < static_cast<int64_t>(broadcast_shape.size()); ++i) {
    if (i < offset || batch_q[i - offset] == 1) {
      if (i < offset || (i >= offset && broadcast_shape[i] != 1)) {
        reduce_dims_q.push_back(i);
      }
    }
  }
  if (!reduce_dims_q.empty()) {
    grad_q = grad_q_full.sum(reduce_dims_q, /*keepdim=*/false);
  }
  // Reshape to match original q shape
  grad_q = grad_q.reshape(q.sizes());

  // Sum along dimensions that were broadcast in point
  std::vector<int64_t> reduce_dims_point;
  offset = broadcast_shape.size() - batch_point.size();
  for (int64_t i = 0; i < static_cast<int64_t>(broadcast_shape.size()); ++i) {
    if (i < offset || batch_point[i - offset] == 1) {
      if (i < offset || (i >= offset && broadcast_shape[i] != 1)) {
        reduce_dims_point.push_back(i);
      }
    }
  }
  if (!reduce_dims_point.empty()) {
    grad_point = grad_point_full.sum(reduce_dims_point, /*keepdim=*/false);
  }
  // Reshape to match original point shape
  grad_point = grad_point.reshape(point.sizes());

  return std::make_tuple(grad_q, grad_point);
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("quaternion_apply", &torchscience::cpu::geometry::transform::quaternion_apply);
  m.impl("quaternion_apply_backward", &torchscience::cpu::geometry::transform::quaternion_apply_backward);
}
