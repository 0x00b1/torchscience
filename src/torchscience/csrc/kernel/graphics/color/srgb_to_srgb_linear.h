#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB value to linear sRGB.
 *
 * Implements the IEC 61966-2-1 standard sRGB to linear conversion:
 *   if srgb <= 0.04045: linear = srgb / 12.92
 *   else: linear = ((srgb + 0.055) / 1.055)^2.4
 *
 * @param srgb Pointer to input sRGB values
 * @param linear Pointer to output linear RGB values
 * @param n Number of elements to convert
 */
template <typename T>
void srgb_to_srgb_linear_scalar(const T* srgb, T* linear, int64_t n) {
  const T threshold = T(0.04045);
  const T linear_slope = T(12.92);
  const T offset = T(0.055);
  const T scale = T(1.055);
  const T gamma = T(2.4);

  for (int64_t i = 0; i < n; ++i) {
    const T value = srgb[i];
    if (value <= threshold) {
      linear[i] = value / linear_slope;
    } else {
      linear[i] = std::pow((value + offset) / scale, gamma);
    }
  }
}

}  // namespace torchscience::kernel::graphics::color
