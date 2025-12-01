#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/graphics/color/srgb_linear_to_srgb.h"
#include "../../../kernel/graphics/color/srgb_linear_to_srgb_backward.h"

namespace torchscience::cpu::graphics::color {

/**
 * CPU implementation for linear sRGB to sRGB conversion.
 */
inline at::Tensor srgb_linear_to_srgb(const at::Tensor& input) {
  auto output = at::empty_like(input);

  const int64_t num_elements = input.numel();

  auto input_contig = input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    input.scalar_type(),
    "srgb_linear_to_srgb_cpu",
    [&] {
      const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
      scalar_t* output_ptr = output.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {
        kernel::graphics::color::srgb_linear_to_srgb_scalar(
          input_ptr + begin,
          output_ptr + begin,
          end - begin
        );
      });
    }
  );

  return output;
}

/**
 * CPU implementation for linear sRGB to sRGB backward pass.
 */
inline at::Tensor srgb_linear_to_srgb_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  auto grad_input = at::empty_like(input);

  const int64_t num_elements = input.numel();

  auto grad_output_contig = grad_output.contiguous();
  auto input_contig = input.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    input.scalar_type(),
    "srgb_linear_to_srgb_backward_cpu",
    [&] {
      const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
      const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
      scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

      at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {
        kernel::graphics::color::srgb_linear_to_srgb_backward_scalar(
          grad_output_ptr + begin,
          input_ptr + begin,
          grad_input_ptr + begin,
          end - begin
        );
      });
    }
  );

  return grad_input;
}

}  // namespace torchscience::cpu::graphics::color

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("srgb_linear_to_srgb", &torchscience::cpu::graphics::color::srgb_linear_to_srgb);
  m.impl("srgb_linear_to_srgb_backward", &torchscience::cpu::graphics::color::srgb_linear_to_srgb_backward);
}
