#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hwb_backward_eps() { return T(1e-7); }

/**
 * Compute gradients for sRGB to HWB conversion.
 *
 * @param grad_hwb Input gradients [grad_H, grad_W, grad_B]
 * @param rgb Original input [R, G, B]
 * @param grad_rgb Output gradients [grad_R, grad_G, grad_B]
 */
template <typename T>
void srgb_to_hwb_backward_scalar(const T* grad_hwb, const T* rgb, T* grad_rgb) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T grad_h = grad_hwb[0];
  const T grad_w = grad_hwb[1];
  const T grad_bk = grad_hwb[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;

  const T eps = srgb_to_hwb_backward_eps<T>();
  const T pi_3 = T(1.0471975511965976310501693706873);

  // Initialize gradients to zero
  T grad_r = T(0);
  T grad_g = T(0);
  T grad_b = T(0);

  // Determine which channel is max and which is min
  const bool r_is_max = (r >= g && r >= b);
  const bool g_is_max = (g >= r && g >= b && !r_is_max);
  const bool b_is_max = (!r_is_max && !g_is_max);

  const bool r_is_min = (r <= g && r <= b);
  const bool g_is_min = (g <= r && g <= b && !r_is_min);
  const bool b_is_min = (!r_is_min && !g_is_min);

  // dW/drgb: W = min(r, g, b)
  if (r_is_min) grad_r += grad_w;
  else if (g_is_min) grad_g += grad_w;
  else grad_b += grad_w;

  // dB/drgb: B = 1 - max(r, g, b)
  // dB/dmax = -1
  if (r_is_max) grad_r += -grad_bk;
  else if (g_is_max) grad_g += -grad_bk;
  else grad_b += -grad_bk;

  // dH/drgb: Hue depends on which channel is max (same as HSV)
  if (delta > eps) {
    const T inv_delta = T(1) / delta;
    const T inv_delta_sq = inv_delta * inv_delta;

    if (r_is_max) {
      // H = (pi/3) * ((g - b) / delta mod 6)
      const T g_minus_b = g - b;
      const T dH_dg = pi_3 * inv_delta;
      const T dH_db = -pi_3 * inv_delta;
      const T dH_ddelta = -pi_3 * g_minus_b * inv_delta_sq;

      grad_g += grad_h * dH_dg;
      grad_b += grad_h * dH_db;
      // delta = max - min, d(delta)/dmax = 1, d(delta)/dmin = -1
      grad_r += grad_h * dH_ddelta;  // r is max
      if (g_is_min) grad_g += grad_h * (-dH_ddelta);
      else grad_b += grad_h * (-dH_ddelta);

    } else if (g_is_max) {
      // H = (pi/3) * ((b - r) / delta + 2)
      const T b_minus_r = b - r;
      const T dH_db = pi_3 * inv_delta;
      const T dH_dr = -pi_3 * inv_delta;
      const T dH_ddelta = -pi_3 * b_minus_r * inv_delta_sq;

      grad_b += grad_h * dH_db;
      grad_r += grad_h * dH_dr;
      grad_g += grad_h * dH_ddelta;  // g is max
      if (r_is_min) grad_r += grad_h * (-dH_ddelta);
      else grad_b += grad_h * (-dH_ddelta);

    } else {
      // H = (pi/3) * ((r - g) / delta + 4)
      const T r_minus_g = r - g;
      const T dH_dr = pi_3 * inv_delta;
      const T dH_dg = -pi_3 * inv_delta;
      const T dH_ddelta = -pi_3 * r_minus_g * inv_delta_sq;

      grad_r += grad_h * dH_dr;
      grad_g += grad_h * dH_dg;
      grad_b += grad_h * dH_ddelta;  // b is max
      if (r_is_min) grad_r += grad_h * (-dH_ddelta);
      else grad_g += grad_h * (-dH_ddelta);
    }
  }

  grad_rgb[0] = grad_r;
  grad_rgb[1] = grad_g;
  grad_rgb[2] = grad_b;
}

}  // namespace torchscience::kernel::graphics::color
