#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/graphics/color/hsv_to_srgb.h"
#include "../../../kernel/graphics/color/hsv_to_srgb_backward.h"

namespace torchscience::cpu::graphics::color {

/**
 * CPU implementation for HSV to sRGB conversion.
 */
inline at::Tensor hsv_to_srgb(const at::Tensor& input) {
  TORCH_CHECK(input.size(-1) == 3, "hsv_to_srgb: input must have last dimension 3, got ", input.size(-1));

  auto output = at::empty_like(input);

  const int64_t num_pixels = input.numel() / 3;

  auto input_contig = input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    input.scalar_type(),
    "hsv_to_srgb_cpu",
    [&] {
      const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_pixels, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::graphics::color::hsv_to_srgb_scalar(
            input_ptr + i * 3,
            output_ptr + i * 3
          );
        }
      });
    }
  );

  return output;
}

/**
 * CPU implementation for HSV to sRGB backward pass.
 */
inline at::Tensor hsv_to_srgb_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  TORCH_CHECK(grad_output.size(-1) == 3, "hsv_to_srgb_backward: grad_output must have last dimension 3");
  TORCH_CHECK(input.size(-1) == 3, "hsv_to_srgb_backward: input must have last dimension 3");

  auto grad_input = at::empty_like(input);

  const int64_t num_pixels = input.numel() / 3;

  auto grad_output_contig = grad_output.contiguous();
  auto input_contig = input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    input.scalar_type(),
    "hsv_to_srgb_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
      scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

      at::parallel_for(0, num_pixels, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          kernel::graphics::color::hsv_to_srgb_backward_scalar(
            grad_output_ptr + i * 3,
            input_ptr + i * 3,
            grad_input_ptr + i * 3
          );
        }
      });
    }
  );

  return grad_input;
}

}  // namespace torchscience::cpu::graphics::color

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("hsv_to_srgb", &torchscience::cpu::graphics::color::hsv_to_srgb);
  m.impl("hsv_to_srgb_backward", &torchscience::cpu::graphics::color::hsv_to_srgb_backward);
}
