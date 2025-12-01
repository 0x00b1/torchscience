#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Compute gradients for HSV to sRGB conversion.
 *
 * @param grad_rgb Input gradients [grad_R, grad_G, grad_B]
 * @param hsv Original input [H, S, V]
 * @param grad_hsv Output gradients [grad_H, grad_S, grad_V]
 */
template <typename T>
void hsv_to_srgb_backward_scalar(const T* grad_rgb, const T* hsv, T* grad_hsv) {
  const T h = hsv[0];
  const T s = hsv[1];
  const T v = hsv[2];

  const T grad_r = grad_rgb[0];
  const T grad_g = grad_rgb[1];
  const T grad_b = grad_rgb[2];

  // Forward pass values
  const T c = v * s;
  const T three_over_pi = T(0.9549296585513721);
  const T h_prime = h * three_over_pi;
  const T h_mod_2 = std::fmod(h_prime, T(2));
  const T abs_arg = h_mod_2 - T(1);
  const T x = c * (T(1) - std::abs(abs_arg));

  // Gradients for m = V - C
  // ∂m/∂V = 1, ∂m/∂S = -V, ∂m/∂H = 0
  // All RGB components include +m, so grad flows through
  const T grad_m = grad_r + grad_g + grad_b;

  // Determine sector
  const int sector = static_cast<int>(std::floor(h_prime)) % 6;
  const int sector_wrapped = (sector + 6) % 6;

  // Gradients for C and X based on sector
  T grad_c = T(0);
  T grad_x = T(0);

  switch (sector_wrapped) {
    case 0:  // r = c + m, g = x + m, b = m
      grad_c = grad_r;
      grad_x = grad_g;
      break;
    case 1:  // r = x + m, g = c + m, b = m
      grad_x = grad_r;
      grad_c = grad_g;
      break;
    case 2:  // r = m, g = c + m, b = x + m
      grad_c = grad_g;
      grad_x = grad_b;
      break;
    case 3:  // r = m, g = x + m, b = c + m
      grad_x = grad_g;
      grad_c = grad_b;
      break;
    case 4:  // r = x + m, g = m, b = c + m
      grad_x = grad_r;
      grad_c = grad_b;
      break;
    case 5:  // r = c + m, g = m, b = x + m
    default:
      grad_c = grad_r;
      grad_x = grad_b;
      break;
  }

  // Backprop through X = C * (1 - |H' mod 2 - 1|)
  // Let f = 1 - |H' mod 2 - 1|
  // ∂X/∂C = f
  // ∂X/∂H' = C * ∂f/∂H' = C * (-sign(H' mod 2 - 1))
  // ∂X/∂H = ∂X/∂H' * (3/π)
  const T f = T(1) - std::abs(abs_arg);
  const T sign_abs_arg = (abs_arg >= T(0)) ? T(1) : T(-1);
  const T dX_dH_prime = -c * sign_abs_arg;

  // Backprop through C = V * S
  // ∂C/∂V = S, ∂C/∂S = V
  const T grad_c_from_x = grad_x * f;  // from X = C * f
  const T total_grad_c = grad_c + grad_c_from_x;

  // Compute final gradients
  // ∂/∂H: only from X
  const T grad_h = grad_x * dX_dH_prime * three_over_pi;

  // ∂/∂S: from C and from m
  // m = V - C = V - V*S = V*(1-S)
  // ∂m/∂S = -V
  const T grad_s = total_grad_c * v + grad_m * (-v);

  // ∂/∂V: from C and from m
  // ∂m/∂V = 1 - S
  const T grad_v = total_grad_c * s + grad_m * (T(1) - s);

  grad_hsv[0] = grad_h;
  grad_hsv[1] = grad_s;
  grad_hsv[2] = grad_v;
}

}  // namespace torchscience::kernel::graphics::color
