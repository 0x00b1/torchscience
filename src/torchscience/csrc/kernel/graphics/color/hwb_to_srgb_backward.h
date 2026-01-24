#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Compute gradients for HWB to sRGB conversion.
 *
 * @param grad_rgb Input gradients [grad_R, grad_G, grad_B]
 * @param hwb Original input [H, W, B]
 * @param grad_hwb Output gradients [grad_H, grad_W, grad_B]
 */
template <typename T>
void hwb_to_srgb_backward_scalar(const T* grad_rgb, const T* hwb, T* grad_hwb) {
  const T h = hwb[0];
  const T w = hwb[1];
  const T bk = hwb[2];

  const T grad_r = grad_rgb[0];
  const T grad_g = grad_rgb[1];
  const T grad_b = grad_rgb[2];

  const T w_plus_b = w + bk;

  // Handle achromatic case: W + B >= 1
  if (w_plus_b >= T(1)) {
    // gray = W / (W + B)
    // d(gray)/dW = B / (W + B)^2
    // d(gray)/dB = -W / (W + B)^2
    const T grad_gray = grad_r + grad_g + grad_b;
    const T inv_w_plus_b_sq = T(1) / (w_plus_b * w_plus_b);
    grad_hwb[0] = T(0);  // No hue dependency
    grad_hwb[1] = grad_gray * bk * inv_w_plus_b_sq;
    grad_hwb[2] = grad_gray * (-w) * inv_w_plus_b_sq;
    return;
  }

  // Forward pass values (chromatic case)
  // V = 1 - B
  // S = 1 - W / V = (V - W) / V
  const T v = T(1) - bk;
  const T s = T(1) - w / v;
  const T c = v * s;

  const T three_over_pi = T(0.9549296585513721);
  const T h_prime = h * three_over_pi;
  const T h_mod_2 = std::fmod(h_prime, T(2));
  const T abs_arg = h_mod_2 - T(1);
  const T x = c * (T(1) - std::abs(abs_arg));

  // Gradients for m = V - C
  // All RGB components include +m
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
  const T f = T(1) - std::abs(abs_arg);
  const T sign_abs_arg = (abs_arg >= T(0)) ? T(1) : T(-1);
  const T dX_dH_prime = -c * sign_abs_arg;

  const T grad_c_from_x = grad_x * f;
  const T total_grad_c = grad_c + grad_c_from_x;

  // d/dH: only from X
  const T grad_h = grad_x * dX_dH_prime * three_over_pi;

  // Now backprop through C, V, S to W, B
  // C = V * S = V * (1 - W/V) = V - W
  // m = V - C = V - (V - W) = W
  //
  // Actually, let's be more careful:
  // V = 1 - B
  // S = 1 - W/V = (V - W) / V
  // C = V * S = V - W
  // m = V - C = V - (V - W) = W
  //
  // So RGB = (r1, g1, b1) + m where (r1, g1, b1) depends on sector
  // and m = W
  //
  // Let's verify: grad_m flows to W directly!
  //
  // dC/dW = -1
  // dC/dV = 1
  // dC/dB = dC/dV * dV/dB = 1 * (-1) = -1
  //
  // For S:
  // S = 1 - W/V
  // dS/dW = -1/V
  // dS/dV = W/V^2
  // dS/dB = dS/dV * dV/dB = W/V^2 * (-1) = -W/V^2
  //
  // For the gradients, we have:
  // total_grad_c flows back through C = V * S
  // grad_m flows back through m = W
  //
  // Actually m = V - C = V - V*S = V(1-S)
  // Let's re-derive:
  // m = V - C where C = V*S
  // dm/dV = 1 - S
  // dm/dS = -V
  //
  // Since S = 1 - W/V:
  // dm/dW = dm/dS * dS/dW = (-V) * (-1/V) = 1
  // dm/dB = dm/dV * dV/dB + dm/dS * dS/dB
  //       = (1-S)*(-1) + (-V)*(W/V^2)*(-1)
  //       = -(1-S) + W/V
  //       = -1 + S + W/V
  //       = -1 + (1 - W/V) + W/V
  //       = 0
  // Wait, that's not right either...
  //
  // Let me be very explicit:
  // V = 1 - bk
  // S = 1 - w/V
  // C = V * S = V - w
  // m = V - C = V - (V - w) = w
  //
  // So m = w directly! This means:
  // dm/dw = 1, dm/dbk = 0, dm/dh = 0

  // For C = V - w = (1 - bk) - w:
  // dC/dw = -1
  // dC/dbk = -1

  // grad_w from m: grad_m * dm/dw = grad_m * 1 = grad_m
  // grad_w from C: total_grad_c * dC/dw = total_grad_c * (-1)
  const T grad_w = grad_m + total_grad_c * T(-1);

  // grad_bk from C: total_grad_c * dC/dbk = total_grad_c * (-1)
  const T grad_bk = total_grad_c * T(-1);

  grad_hwb[0] = grad_h;
  grad_hwb[1] = grad_w;
  grad_hwb[2] = grad_bk;
}

}  // namespace torchscience::kernel::graphics::color
