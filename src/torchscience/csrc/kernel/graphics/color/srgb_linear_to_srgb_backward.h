#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Compute gradients for linear sRGB to sRGB conversion.
 *
 * Implements the backward pass for IEC 61966-2-1 standard inverse conversion:
 *   if linear <= 0.0031308: d_srgb/d_linear = 12.92
 *   else: d_srgb/d_linear = (1.055 / 2.4) * linear^(1/2.4 - 1)
 *
 * @param grad_output Pointer to incoming gradients
 * @param linear Pointer to original linear RGB input values
 * @param grad_input Pointer to output gradients
 * @param n Number of elements
 */
template <typename T>
void srgb_linear_to_srgb_backward_scalar(
    const T* grad_output,
    const T* linear,
    T* grad_input,
    int64_t n
) {
  const T threshold = T(0.0031308);
  const T linear_slope = T(12.92);
  const T scale = T(1.055);
  const T gamma = T(2.4);
  const T inverse_gamma_minus_one = T(1.0 / 2.4 - 1.0);

  for (int64_t i = 0; i < n; ++i) {
    const T value = linear[i];
    T local_grad;
    if (value <= threshold) {
      // d/d_linear (linear * 12.92) = 12.92
      local_grad = linear_slope;
    } else {
      // d/d_linear (1.055 * linear^(1/2.4) - 0.055)
      // = 1.055 * (1/2.4) * linear^(1/2.4 - 1)
      // = (1.055 / 2.4) * linear^(1/2.4 - 1)
      local_grad = (scale / gamma) * std::pow(value, inverse_gamma_minus_one);
    }
    grad_input[i] = grad_output[i] * local_grad;
  }
}

}  // namespace torchscience::kernel::graphics::color
