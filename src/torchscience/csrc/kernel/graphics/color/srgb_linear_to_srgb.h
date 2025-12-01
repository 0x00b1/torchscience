#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single linear sRGB value to sRGB.
 *
 * Implements the IEC 61966-2-1 standard inverse (linear to sRGB) conversion:
 *   if linear <= 0.0031308: srgb = linear * 12.92
 *   else: srgb = 1.055 * linear^(1/2.4) - 0.055
 *
 * @param linear Pointer to input linear RGB values
 * @param srgb Pointer to output sRGB values
 * @param n Number of elements to convert
 */
template <typename T>
void srgb_linear_to_srgb_scalar(const T* linear, T* srgb, int64_t n) {
  const T threshold = T(0.0031308);
  const T linear_slope = T(12.92);
  const T offset = T(0.055);
  const T scale = T(1.055);
  const T inverse_gamma = T(1.0 / 2.4);

  for (int64_t i = 0; i < n; ++i) {
    const T value = linear[i];
    if (value <= threshold) {
      srgb[i] = value * linear_slope;
    } else {
      srgb[i] = scale * std::pow(value, inverse_gamma) - offset;
    }
  }
}

}  // namespace torchscience::kernel::graphics::color
