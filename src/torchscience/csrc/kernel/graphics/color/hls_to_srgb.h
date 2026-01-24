#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Helper function for HLS to sRGB conversion.
 * Converts a hue value to an RGB component.
 *
 * @param p First intermediate value
 * @param q Second intermediate value
 * @param t Hue offset (normalized to [0, 1])
 * @return RGB component value
 */
template <typename T>
T hue_to_rgb_component(T p, T q, T t) {
  // Wrap t to [0, 1)
  if (t < T(0)) t += T(1);
  if (t > T(1)) t -= T(1);

  if (t < T(1) / T(6)) {
    return p + (q - p) * T(6) * t;
  }
  if (t < T(0.5)) {
    return q;
  }
  if (t < T(2) / T(3)) {
    return p + (q - p) * (T(2) / T(3) - t) * T(6);
  }
  return p;
}

/**
 * Convert a single HLS pixel to sRGB.
 *
 * @param hls Input array [H, L, S] where H is in [0, 2π]
 * @param rgb Output array [R, G, B]
 */
template <typename T>
void hls_to_srgb_scalar(const T* hls, T* rgb) {
  const T h = hls[0];
  const T l = hls[1];
  const T s = hls[2];

  T r, g, b;

  if (s < T(1e-7)) {
    // Achromatic
    r = l;
    g = l;
    b = l;
  } else {
    // Compute q and p
    T q;
    if (l < T(0.5)) {
      q = l * (T(1) + s);
    } else {
      q = l + s - l * s;
    }
    const T p = T(2) * l - q;

    // Convert hue from [0, 2π] to [0, 1]
    const T one_over_two_pi = T(0.15915494309189533576888376337251);
    const T h_norm = h * one_over_two_pi;

    // Compute RGB components
    r = hue_to_rgb_component(p, q, h_norm + T(1) / T(3));
    g = hue_to_rgb_component(p, q, h_norm);
    b = hue_to_rgb_component(p, q, h_norm - T(1) / T(3));
  }

  rgb[0] = r;
  rgb[1] = g;
  rgb[2] = b;
}

}  // namespace torchscience::kernel::graphics::color
