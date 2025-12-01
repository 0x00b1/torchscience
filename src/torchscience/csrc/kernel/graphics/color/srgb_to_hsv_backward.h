#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hsv_backward_eps() { return T(1e-7); }

/**
 * Compute gradients for sRGB to HSV conversion.
 *
 * @param grad_hsv Input gradients [grad_H, grad_S, grad_V]
 * @param rgb Original input [R, G, B]
 * @param grad_rgb Output gradients [grad_R, grad_G, grad_B]
 */
template <typename T>
void srgb_to_hsv_backward_scalar(const T* grad_hsv, const T* rgb, T* grad_rgb) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T grad_h = grad_hsv[0];
  const T grad_s = grad_hsv[1];
  const T grad_v = grad_hsv[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;

  const T eps = srgb_to_hsv_backward_eps<T>();
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

  // ∂V/∂rgb: V = max(r, g, b)
  if (r_is_max) grad_r += grad_v;
  else if (g_is_max) grad_g += grad_v;
  else grad_b += grad_v;

  // ∂S/∂rgb: S = delta / max = (max - min) / max
  // ∂S/∂max = min / max² = (max - delta) / max²
  // ∂S/∂min = -1 / max
  if (max_val > eps) {
    const T inv_max = T(1) / max_val;
    const T inv_max_sq = inv_max * inv_max;
    const T dS_dmax = min_val * inv_max_sq;
    const T dS_dmin = -inv_max;

    if (r_is_max) grad_r += grad_s * dS_dmax;
    else if (g_is_max) grad_g += grad_s * dS_dmax;
    else grad_b += grad_s * dS_dmax;

    if (r_is_min) grad_r += grad_s * dS_dmin;
    else if (g_is_min) grad_g += grad_s * dS_dmin;
    else grad_b += grad_s * dS_dmin;
  }

  // ∂H/∂rgb: Hue depends on which channel is max
  if (delta > eps) {
    const T inv_delta = T(1) / delta;
    const T inv_delta_sq = inv_delta * inv_delta;

    if (r_is_max) {
      // H = (π/3) * ((g - b) / delta mod 6)
      // ∂H/∂g = (π/3) / delta
      // ∂H/∂b = -(π/3) / delta
      // ∂H/∂max = -(π/3) * (g - b) / delta²
      // ∂H/∂min = (π/3) * (g - b) / delta²
      const T g_minus_b = g - b;
      const T dH_dg = pi_3 * inv_delta;
      const T dH_db = -pi_3 * inv_delta;
      const T dH_ddelta = -pi_3 * g_minus_b * inv_delta_sq;

      grad_g += grad_h * dH_dg;
      grad_b += grad_h * dH_db;
      // delta = max - min, ∂delta/∂max = 1, ∂delta/∂min = -1
      grad_r += grad_h * dH_ddelta;  // r is max
      if (g_is_min) grad_g += grad_h * (-dH_ddelta);
      else grad_b += grad_h * (-dH_ddelta);

    } else if (g_is_max) {
      // H = (π/3) * ((b - r) / delta + 2)
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
      // H = (π/3) * ((r - g) / delta + 4)
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
