#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single HWB pixel to sRGB.
 *
 * @param hwb Input array [H, W, B] where H is in [0, 2*pi]
 * @param rgb Output array [R, G, B]
 */
template <typename T>
void hwb_to_srgb_scalar(const T* hwb, T* rgb) {
  const T h = hwb[0];
  const T w = hwb[1];
  const T bk = hwb[2];

  // Handle achromatic case: W + B >= 1
  const T w_plus_b = w + bk;
  if (w_plus_b >= T(1)) {
    const T gray = w / w_plus_b;
    rgb[0] = gray;
    rgb[1] = gray;
    rgb[2] = gray;
    return;
  }

  // Convert HWB to HSV
  // V = 1 - B
  // S = 1 - W / (1 - B) = 1 - W / V
  const T v = T(1) - bk;
  const T s = T(1) - w / v;

  // Now convert HSV to RGB (same as hsv_to_srgb)
  // C = V * S (chroma)
  const T c = v * s;

  // H' = H * (3/pi) to scale from [0, 2*pi] to [0, 6]
  const T three_over_pi = T(0.9549296585513721);
  const T h_prime = h * three_over_pi;

  // X = C * (1 - |H' mod 2 - 1|)
  const T h_mod_2 = std::fmod(h_prime, T(2));
  const T x = c * (T(1) - std::abs(h_mod_2 - T(1)));

  // m = V - C
  const T m = v - c;

  // Determine sector and compute RGB
  T r, g, b;

  const int sector = static_cast<int>(std::floor(h_prime)) % 6;
  const int sector_wrapped = (sector + 6) % 6;

  switch (sector_wrapped) {
    case 0:  // H' in [0, 1)
      r = c; g = x; b = T(0);
      break;
    case 1:  // H' in [1, 2)
      r = x; g = c; b = T(0);
      break;
    case 2:  // H' in [2, 3)
      r = T(0); g = c; b = x;
      break;
    case 3:  // H' in [3, 4)
      r = T(0); g = x; b = c;
      break;
    case 4:  // H' in [4, 5)
      r = x; g = T(0); b = c;
      break;
    case 5:  // H' in [5, 6)
    default:
      r = c; g = T(0); b = x;
      break;
  }

  rgb[0] = r + m;
  rgb[1] = g + m;
  rgb[2] = b + m;
}

}  // namespace torchscience::kernel::graphics::color
