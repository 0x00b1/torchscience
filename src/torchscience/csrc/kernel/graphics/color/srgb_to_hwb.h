#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hwb_eps() { return T(1e-7); }

template <typename T>
inline T srgb_to_hwb_two_pi() { return T(6.283185307179586476925286766559); }

template <typename T>
inline T srgb_to_hwb_pi_over_3() { return T(1.0471975511965976310501693706873); }

/**
 * Convert a single sRGB pixel to HWB.
 *
 * @param rgb Input array [R, G, B]
 * @param hwb Output array [H, W, B] where H is in [0, 2*pi]
 */
template <typename T>
void srgb_to_hwb_scalar(const T* rgb, T* hwb) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;

  // Whiteness = min(R, G, B)
  const T w = min_val;

  // Blackness = 1 - max(R, G, B)
  const T bk = T(1) - max_val;

  // Hue (same as HSV)
  T h;
  if (delta < srgb_to_hwb_eps<T>()) {
    h = T(0);  // Achromatic
  } else if (r >= g && r >= b) {
    // Red is max
    h = srgb_to_hwb_pi_over_3<T>() * std::fmod((g - b) / delta + T(6), T(6));
  } else if (g >= r && g >= b) {
    // Green is max
    h = srgb_to_hwb_pi_over_3<T>() * ((b - r) / delta + T(2));
  } else {
    // Blue is max
    h = srgb_to_hwb_pi_over_3<T>() * ((r - g) / delta + T(4));
  }

  hwb[0] = h;
  hwb[1] = w;
  hwb[2] = bk;
}

}  // namespace torchscience::kernel::graphics::color
