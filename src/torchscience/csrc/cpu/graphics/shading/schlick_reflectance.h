#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/graphics/shading/schlick_reflectance.h"
#include "../../../kernel/graphics/shading/schlick_reflectance_backward.h"

namespace torchscience::cpu::graphics::shading {

/**
 * CPU implementation for Schlick reflectance approximation.
 */
inline at::Tensor schlick_reflectance(
    const at::Tensor& cosine,
    const at::Tensor& r0
) {
  TORCH_CHECK(cosine.scalar_type() == r0.scalar_type(),
              "schlick_reflectance: cosine and r0 must have the same dtype");

  // Broadcast cosine and r0 to compatible shape
  auto broadcasted = at::broadcast_tensors({cosine, r0});
  auto cosine_expanded = broadcasted[0].contiguous();
  auto r0_expanded = broadcasted[1].contiguous();

  auto output = at::empty_like(cosine_expanded);
  int64_t num_elements = output.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    cosine.scalar_type(),
    "schlick_reflectance_cpu",
    [&] {
      const scalar_t* cosine_ptr = cosine_expanded.data_ptr<scalar_t>();
      const scalar_t* r0_ptr = r0_expanded.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          output_ptr[i] = kernel::graphics::shading::schlick_reflectance_scalar(
            cosine_ptr[i],
            r0_ptr[i]
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU implementation for Schlick reflectance backward pass.
 */
inline at::Tensor schlick_reflectance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& cosine,
    const at::Tensor& r0
) {
  // Broadcast all tensors to compatible shape
  auto broadcasted = at::broadcast_tensors({grad_output, cosine, r0});
  auto grad_output_expanded = broadcasted[0].contiguous();
  auto cosine_expanded = broadcasted[1].contiguous();
  auto r0_expanded = broadcasted[2].contiguous();

  auto grad_cosine = at::empty_like(cosine_expanded);
  int64_t num_elements = grad_cosine.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    cosine.scalar_type(),
    "schlick_reflectance_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_expanded.data_ptr<scalar_t>();
      const scalar_t* cosine_ptr = cosine_expanded.data_ptr<scalar_t>();
      const scalar_t* r0_ptr = r0_expanded.data_ptr<scalar_t>();
      scalar_t* grad_cosine_ptr = grad_cosine.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          grad_cosine_ptr[i] = kernel::graphics::shading::schlick_reflectance_backward_scalar(
            grad_output_ptr[i],
            cosine_ptr[i],
            r0_ptr[i]
          );
        }
      });
    }
  );

  return grad_cosine;
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("schlick_reflectance", &torchscience::cpu::graphics::shading::schlick_reflectance);
  m.impl("schlick_reflectance_backward", &torchscience::cpu::graphics::shading::schlick_reflectance_backward);
}
