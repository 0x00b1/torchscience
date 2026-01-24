#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::color {

/**
 * Meta implementation for sRGB to YUV shape inference.
 */
inline at::Tensor srgb_to_yuv(const at::Tensor& input) {
  TORCH_CHECK(input.size(-1) == 3, "srgb_to_yuv: input must have last dimension 3, got ", input.size(-1));
  return at::empty_like(input);
}

/**
 * Meta implementation for sRGB to YUV backward shape inference.
 */
inline at::Tensor srgb_to_yuv_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  return at::empty_like(input);
}

}  // namespace torchscience::meta::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("srgb_to_yuv", &torchscience::meta::graphics::color::srgb_to_yuv);
  m.impl("srgb_to_yuv_backward", &torchscience::meta::graphics::color::srgb_to_yuv_backward);
}
