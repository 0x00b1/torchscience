#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/transform/reflect.h"
#include "../../../kernel/geometry/transform/reflect_backward.h"

namespace torchscience::cpu::geometry::transform {

/**
 * CPU implementation for vector reflection.
 */
inline at::Tensor reflect(const at::Tensor& direction, const at::Tensor& normal) {
  TORCH_CHECK(direction.size(-1) == 3, "reflect: direction must have last dimension 3, got ", direction.size(-1));
  TORCH_CHECK(normal.size(-1) == 3, "reflect: normal must have last dimension 3, got ", normal.size(-1));
  TORCH_CHECK(direction.scalar_type() == normal.scalar_type(),
              "reflect: direction and normal must have the same dtype");
  TORCH_CHECK(direction.sizes().slice(0, direction.dim() - 1) ==
              normal.sizes().slice(0, normal.dim() - 1),
              "reflect: direction and normal must have matching batch dimensions");

  auto output = at::empty_like(direction);

  const int64_t num_vectors = direction.numel() / 3;

  auto direction_contig = direction.contiguous();
  auto normal_contig = normal.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    direction.scalar_type(),
    "reflect_cpu",
    [&] {
      const scalar_t* direction_ptr = direction_contig.data_ptr<scalar_t>();
      const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::reflect_scalar(
            direction_ptr + i * 3,
            normal_ptr + i * 3,
            output_ptr + i * 3
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU implementation for vector reflection backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor> reflect_backward(
    const at::Tensor& grad_output,
    const at::Tensor& direction,
    const at::Tensor& normal
) {
  TORCH_CHECK(grad_output.size(-1) == 3, "reflect_backward: grad_output must have last dimension 3");
  TORCH_CHECK(direction.size(-1) == 3, "reflect_backward: direction must have last dimension 3");
  TORCH_CHECK(normal.size(-1) == 3, "reflect_backward: normal must have last dimension 3");

  auto grad_direction = at::empty_like(direction);
  auto grad_normal = at::empty_like(normal);

  const int64_t num_vectors = direction.numel() / 3;

  auto grad_output_contig = grad_output.contiguous();
  auto direction_contig = direction.contiguous();
  auto normal_contig = normal.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    direction.scalar_type(),
    "reflect_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* direction_ptr = direction_contig.data_ptr<scalar_t>();
      const scalar_t* normal_ptr = normal_contig.data_ptr<scalar_t>();
      scalar_t* grad_direction_ptr = grad_direction.data_ptr<scalar_t>();
      scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();

      at::parallel_for(0, num_vectors, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::geometry::transform::reflect_backward_scalar(
            grad_output_ptr + i * 3,
            direction_ptr + i * 3,
            normal_ptr + i * 3,
            grad_direction_ptr + i * 3,
            grad_normal_ptr + i * 3
          );
        }
      });
    }
  );

  return std::make_tuple(grad_direction, grad_normal);
}

}  // namespace torchscience::cpu::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("reflect", &torchscience::cpu::geometry::transform::reflect);
  m.impl("reflect_backward", &torchscience::cpu::geometry::transform::reflect_backward);
}
