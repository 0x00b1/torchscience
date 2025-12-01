#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Compute gradients for sRGB to linear sRGB conversion.
 *
 * Implements the backward pass for IEC 61966-2-1 standard conversion:
 *   if srgb <= 0.04045: d_linear/d_srgb = 1 / 12.92
 *   else: d_linear/d_srgb = (2.4 / 1.055) * ((srgb + 0.055) / 1.055)^1.4
 *
 * @param grad_output Pointer to incoming gradients
 * @param srgb Pointer to original sRGB input values
 * @param grad_input Pointer to output gradients
 * @param n Number of elements
 */
template <typename T>
void srgb_to_srgb_linear_backward_scalar(
    const T* grad_output,
    const T* srgb,
    T* grad_input,
    int64_t n
) {
  const T threshold = T(0.04045);
  const T linear_slope = T(12.92);
  const T offset = T(0.055);
  const T scale = T(1.055);
  const T gamma = T(2.4);
  const T gamma_minus_one = T(1.4);

  for (int64_t i = 0; i < n; ++i) {
    const T value = srgb[i];
    T local_grad;
    if (value <= threshold) {
      // d/d_srgb (srgb / 12.92) = 1 / 12.92
      local_grad = T(1) / linear_slope;
    } else {
      // d/d_srgb ((srgb + 0.055) / 1.055)^2.4
      // = 2.4 * ((srgb + 0.055) / 1.055)^1.4 * (1 / 1.055)
      // = (2.4 / 1.055) * ((srgb + 0.055) / 1.055)^1.4
      local_grad = (gamma / scale) * std::pow((value + offset) / scale, gamma_minus_one);
    }
    grad_input[i] = grad_output[i] * local_grad;
  }
}

}  // namespace torchscience::kernel::graphics::color
