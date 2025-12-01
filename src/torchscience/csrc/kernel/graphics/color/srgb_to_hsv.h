#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hsv_eps() { return T(1e-7); }

template <typename T>
inline T two_pi() { return T(6.283185307179586476925286766559); }

template <typename T>
inline T pi_over_3() { return T(1.0471975511965976310501693706873); }

/**
 * Convert a single sRGB pixel to HSV.
 *
 * @param rgb Input array [R, G, B]
 * @param hsv Output array [H, S, V] where H is in [0, 2π]
 */
template <typename T>
void srgb_to_hsv_scalar(const T* rgb, T* hsv) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;

  // Value
  const T v = max_val;

  // Saturation
  T s;
  if (max_val < srgb_to_hsv_eps<T>()) {
    s = T(0);
  } else {
    s = delta / max_val;
  }

  // Hue
  T h;
  if (delta < srgb_to_hsv_eps<T>()) {
    h = T(0);  // Achromatic
  } else if (r >= g && r >= b) {
    // Red is max
    h = pi_over_3<T>() * std::fmod((g - b) / delta + T(6), T(6));
  } else if (g >= r && g >= b) {
    // Green is max
    h = pi_over_3<T>() * ((b - r) / delta + T(2));
  } else {
    // Blue is max
    h = pi_over_3<T>() * ((r - g) / delta + T(4));
  }

  hsv[0] = h;
  hsv[1] = s;
  hsv[2] = v;
}

}  // namespace torchscience::kernel::graphics::color
