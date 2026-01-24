#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hls_eps() { return T(1e-7); }

/**
 * Convert a single sRGB pixel to HLS.
 *
 * @param rgb Input array [R, G, B]
 * @param hls Output array [H, L, S] where H is in [0, 2π]
 */
template <typename T>
void srgb_to_hls_scalar(const T* rgb, T* hls) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;

  // Lightness
  const T l = (max_val + min_val) / T(2);

  // Saturation
  T s;
  if (delta < srgb_to_hls_eps<T>()) {
    s = T(0);
  } else if (l > T(0.5)) {
    s = delta / (T(2) - max_val - min_val);
  } else {
    s = delta / (max_val + min_val);
  }

  // Hue (in radians, [0, 2π])
  const T pi_3 = T(1.0471975511965976310501693706873);  // π/3
  T h;
  if (delta < srgb_to_hls_eps<T>()) {
    h = T(0);  // Achromatic
  } else if (r >= g && r >= b) {
    // Red is max
    h = pi_3 * std::fmod((g - b) / delta + T(6), T(6));
  } else if (g >= r && g >= b) {
    // Green is max
    h = pi_3 * ((b - r) / delta + T(2));
  } else {
    // Blue is max
    h = pi_3 * ((r - g) / delta + T(4));
  }

  hls[0] = h;
  hls[1] = l;
  hls[2] = s;
}

}  // namespace torchscience::kernel::graphics::color
