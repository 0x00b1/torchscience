#pragma once

#include <algorithm>
#include <cmath>

namespace torchscience::kernel::graphics::color {

template <typename T>
inline T srgb_to_hls_backward_eps() { return T(1e-7); }

/**
 * Compute gradients for sRGB to HLS conversion.
 *
 * @param grad_hls Input gradients [grad_H, grad_L, grad_S]
 * @param rgb Original input [R, G, B]
 * @param grad_rgb Output gradients [grad_R, grad_G, grad_B]
 */
template <typename T>
void srgb_to_hls_backward_scalar(const T* grad_hls, const T* rgb, T* grad_rgb) {
  const T r = rgb[0];
  const T g = rgb[1];
  const T b = rgb[2];

  const T grad_h = grad_hls[0];
  const T grad_l = grad_hls[1];
  const T grad_s = grad_hls[2];

  const T max_val = std::max({r, g, b});
  const T min_val = std::min({r, g, b});
  const T delta = max_val - min_val;
  const T sum = max_val + min_val;
  const T l = sum / T(2);

  const T eps = srgb_to_hls_backward_eps<T>();
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

  // ∂L/∂rgb: L = (max + min) / 2
  // ∂L/∂max = 0.5, ∂L/∂min = 0.5
  const T dL_dmax = T(0.5);
  const T dL_dmin = T(0.5);

  if (r_is_max) grad_r += grad_l * dL_dmax;
  else if (g_is_max) grad_g += grad_l * dL_dmax;
  else grad_b += grad_l * dL_dmax;

  if (r_is_min) grad_r += grad_l * dL_dmin;
  else if (g_is_min) grad_g += grad_l * dL_dmin;
  else grad_b += grad_l * dL_dmin;

  // ∂S/∂rgb: S depends on L
  // if L > 0.5: S = delta / (2 - sum)
  // else:       S = delta / sum
  if (delta > eps) {
    T dS_dmax, dS_dmin;

    if (l > T(0.5)) {
      // S = delta / (2 - sum) = (max - min) / (2 - max - min)
      const T denom = T(2) - sum;
      const T inv_denom = T(1) / denom;
      const T inv_denom_sq = inv_denom * inv_denom;

      // ∂S/∂max = (1 * denom + delta) / denom² = (denom + delta) / denom²
      //         = (2 - sum + delta) / denom² = (2 - max - min + max - min) / denom²
      //         = (2 - 2*min) / denom²
      dS_dmax = (T(2) - T(2) * min_val) * inv_denom_sq;

      // ∂S/∂min = (-1 * denom + delta) / denom² = (-denom + delta) / denom²
      //         = (-(2 - sum) + delta) / denom² = (-2 + max + min + max - min) / denom²
      //         = (2*max - 2) / denom²
      dS_dmin = (T(2) * max_val - T(2)) * inv_denom_sq;
    } else {
      // S = delta / sum = (max - min) / (max + min)
      const T inv_sum = T(1) / sum;
      const T inv_sum_sq = inv_sum * inv_sum;

      // ∂S/∂max = (1 * sum - delta) / sum² = (sum - delta) / sum²
      //         = (max + min - max + min) / sum² = (2*min) / sum²
      dS_dmax = T(2) * min_val * inv_sum_sq;

      // ∂S/∂min = (-1 * sum - delta) / sum² = (-sum - delta) / sum²
      //         = (-(max + min) - (max - min)) / sum² = (-2*max) / sum²
      dS_dmin = -T(2) * max_val * inv_sum_sq;
    }

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
      const T g_minus_b = g - b;
      const T dH_dg = pi_3 * inv_delta;
      const T dH_db = -pi_3 * inv_delta;
      const T dH_ddelta = -pi_3 * g_minus_b * inv_delta_sq;

      grad_g += grad_h * dH_dg;
      grad_b += grad_h * dH_db;
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
