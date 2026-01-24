#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Compute gradients for HLS to sRGB conversion.
 *
 * @param grad_rgb Input gradients [grad_R, grad_G, grad_B]
 * @param hls Original input [H, L, S]
 * @param grad_hls Output gradients [grad_H, grad_L, grad_S]
 */
template <typename T>
void hls_to_srgb_backward_scalar(const T* grad_rgb, const T* hls, T* grad_hls) {
  const T h = hls[0];
  const T l = hls[1];
  const T s = hls[2];

  const T grad_r = grad_rgb[0];
  const T grad_g = grad_rgb[1];
  const T grad_b = grad_rgb[2];

  const T eps = T(1e-7);

  T grad_h = T(0);
  T grad_l = T(0);
  T grad_s = T(0);

  if (s < eps) {
    // Achromatic: R = G = B = L
    // ∂R/∂L = ∂G/∂L = ∂B/∂L = 1
    grad_l = grad_r + grad_g + grad_b;
    // grad_h and grad_s remain 0
  } else {
    // Compute forward pass values
    T q;
    T dq_dl, dq_ds;
    if (l < T(0.5)) {
      q = l * (T(1) + s);
      dq_dl = T(1) + s;
      dq_ds = l;
    } else {
      q = l + s - l * s;
      dq_dl = T(1) - s;
      dq_ds = T(1) - l;
    }
    const T p = T(2) * l - q;
    const T dp_dl = T(2) - dq_dl;
    const T dp_ds = -dq_ds;

    // Hue normalization
    const T one_over_two_pi = T(0.15915494309189533576888376337251);
    const T two_pi = T(6.283185307179586476925286766559);
    const T h_norm = h * one_over_two_pi;

    // For each RGB channel, compute gradients through hue_to_rgb_component
    // R uses t = h_norm + 1/3
    // G uses t = h_norm
    // B uses t = h_norm - 1/3

    // Helper to compute gradients for hue_to_rgb_component
    // Returns (∂output/∂p, ∂output/∂q, ∂output/∂t)
    T t_r = h_norm + T(1) / T(3);
    T t_g = h_norm;
    T t_b = h_norm - T(1) / T(3);

    // Wrap to [0, 1)
    if (t_r < T(0)) t_r += T(1);
    if (t_r > T(1)) t_r -= T(1);
    if (t_g < T(0)) t_g += T(1);
    if (t_g > T(1)) t_g -= T(1);
    if (t_b < T(0)) t_b += T(1);
    if (t_b > T(1)) t_b -= T(1);

    // For R channel (t = t_r)
    T dr_dp, dr_dq, dr_dt;
    if (t_r < T(1) / T(6)) {
      // output = p + (q - p) * 6 * t
      dr_dp = T(1) - T(6) * t_r;
      dr_dq = T(6) * t_r;
      dr_dt = (q - p) * T(6);
    } else if (t_r < T(0.5)) {
      // output = q
      dr_dp = T(0);
      dr_dq = T(1);
      dr_dt = T(0);
    } else if (t_r < T(2) / T(3)) {
      // output = p + (q - p) * (2/3 - t) * 6
      dr_dp = T(1) - (T(2) / T(3) - t_r) * T(6);
      dr_dq = (T(2) / T(3) - t_r) * T(6);
      dr_dt = -(q - p) * T(6);
    } else {
      // output = p
      dr_dp = T(1);
      dr_dq = T(0);
      dr_dt = T(0);
    }

    // For G channel (t = t_g)
    T dg_dp, dg_dq, dg_dt;
    if (t_g < T(1) / T(6)) {
      dg_dp = T(1) - T(6) * t_g;
      dg_dq = T(6) * t_g;
      dg_dt = (q - p) * T(6);
    } else if (t_g < T(0.5)) {
      dg_dp = T(0);
      dg_dq = T(1);
      dg_dt = T(0);
    } else if (t_g < T(2) / T(3)) {
      dg_dp = T(1) - (T(2) / T(3) - t_g) * T(6);
      dg_dq = (T(2) / T(3) - t_g) * T(6);
      dg_dt = -(q - p) * T(6);
    } else {
      dg_dp = T(1);
      dg_dq = T(0);
      dg_dt = T(0);
    }

    // For B channel (t = t_b)
    T db_dp, db_dq, db_dt;
    if (t_b < T(1) / T(6)) {
      db_dp = T(1) - T(6) * t_b;
      db_dq = T(6) * t_b;
      db_dt = (q - p) * T(6);
    } else if (t_b < T(0.5)) {
      db_dp = T(0);
      db_dq = T(1);
      db_dt = T(0);
    } else if (t_b < T(2) / T(3)) {
      db_dp = T(1) - (T(2) / T(3) - t_b) * T(6);
      db_dq = (T(2) / T(3) - t_b) * T(6);
      db_dt = -(q - p) * T(6);
    } else {
      db_dp = T(1);
      db_dq = T(0);
      db_dt = T(0);
    }

    // Accumulate gradients
    // ∂output/∂p and ∂output/∂q for each channel
    const T total_grad_p = grad_r * dr_dp + grad_g * dg_dp + grad_b * db_dp;
    const T total_grad_q = grad_r * dr_dq + grad_g * dg_dq + grad_b * db_dq;

    // ∂output/∂t flows to ∂output/∂h through t = h * one_over_two_pi
    // dt/dh = one_over_two_pi
    const T total_grad_t = grad_r * dr_dt + grad_g * dg_dt + grad_b * db_dt;
    grad_h = total_grad_t * one_over_two_pi;

    // Chain through p and q to l and s
    // p = 2*l - q, so dp/dl = 2 - dq/dl, dp/ds = -dq/ds
    grad_l = total_grad_p * dp_dl + total_grad_q * dq_dl;
    grad_s = total_grad_p * dp_ds + total_grad_q * dq_ds;
  }

  grad_hls[0] = grad_h;
  grad_hls[1] = grad_l;
  grad_hls[2] = grad_s;
}

}  // namespace torchscience::kernel::graphics::color
