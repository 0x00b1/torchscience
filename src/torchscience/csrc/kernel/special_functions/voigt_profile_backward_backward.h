#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "faddeeva_w.h"
#include "voigt_profile.h"
#include "voigt_profile_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order gradients of the Voigt profile
//
// This computes the gradients of the backward pass with respect to the inputs.
// For a ternary operator with inputs (x, sigma, gamma) and grad_output,
// the backward returns (grad_x, grad_sigma, grad_gamma).
//
// The backward_backward needs to compute:
//   gg_grad_output = gg_x * d(grad_x)/d(grad_output) + gg_sigma * d(grad_sigma)/d(grad_output) + gg_gamma * d(grad_gamma)/d(grad_output)
//   new_grad_x = gg_x * grad_output * d2V/dx2 + gg_sigma * grad_output * d2V/dsigma_dx + gg_gamma * grad_output * d2V/dgamma_dx
//   new_grad_sigma = ... (similar pattern)
//   new_grad_gamma = ... (similar pattern)
template <typename T>
std::tuple<T, T, T, T> voigt_profile_backward_backward(
    T gg_x, T gg_sigma, T gg_gamma,
    T gradient, T x, T sigma, T gamma) {

  const T sqrt_2 = static_cast<T>(1.4142135623730950488016887242096980786);
  const T sqrt_pi = static_cast<T>(1.7724538509055160272981674833411451828);
  const T sqrt_2pi = static_cast<T>(2.5066282746310005024157652848110452530);
  const T one_over_sqrt_2pi = static_cast<T>(0.39894228040143267793994605993438186848);

  // Handle invalid parameters
  if (sigma <= T(0) || gamma < T(0)) {
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    return {nan_val, nan_val, nan_val, nan_val};
  }

  T s = sigma * sqrt_2;
  c10::complex<T> z(x / s, gamma / s);
  c10::complex<T> w_z = faddeeva_w(z);

  // dw/dz = -2z*w(z) + 2i/sqrt(pi)
  c10::complex<T> two_i_sqrt_pi(T(0), T(2) / sqrt_pi);
  c10::complex<T> dw_dz = -T(2) * z * w_z + two_i_sqrt_pi;

  // d2w/dz2 = (4z^2 - 2)*w(z) - 4iz/sqrt(pi)
  c10::complex<T> four_iz_sqrt_pi(T(0), T(4) / sqrt_pi);
  c10::complex<T> d2w_dz2 = (T(4) * z * z - c10::complex<T>(T(2), T(0))) * w_z
                          - four_iz_sqrt_pi * z;

  // V and first derivatives (same as in backward)
  T V = w_z.real() * one_over_sqrt_2pi / sigma;
  T dV_dx = dw_dz.real() / (s * sigma * sqrt_2pi);
  c10::complex<T> dw_dz_times_z = dw_dz * z;
  T dV_dsigma = -dw_dz_times_z.real() / (sigma * sigma * sqrt_2pi) - V / sigma;
  T dV_dgamma = -dw_dz.imag() / (s * sigma * sqrt_2pi);

  // Contribution to grad_output gradient
  // d(grad_x)/d(grad_output) = dV/dx, etc.
  T gg_out = gg_x * dV_dx + gg_sigma * dV_dsigma + gg_gamma * dV_dgamma;

  // Second derivatives
  // dz/dx = 1/s, dz/dsigma = -z/sigma, dz/dgamma = i/s
  c10::complex<T> i(T(0), T(1));

  // d2V/dx2 = Re[d2w/dz2] * (dz/dx)^2 / (sigma * sqrt(2*pi))
  //         = Re[d2w/dz2] / (s^2 * sigma * sqrt(2*pi))
  T d2V_dx2 = d2w_dz2.real() / (s * s * sigma * sqrt_2pi);

  // d2V/dx_dsigma: more complex, use chain rule
  // V = Re[w(z)] / (sigma * sqrt(2*pi))
  // dV/dx = Re[dw/dz] / (s * sigma * sqrt(2*pi))
  // d2V/dx_dsigma = d/dsigma[ Re[dw/dz] / (s * sigma * sqrt(2*pi)) ]
  // This involves multiple terms from the chain rule
  c10::complex<T> d2w_dz2_times_dz_dsigma = d2w_dz2 * (-z / sigma);
  T d2V_dx_dsigma = d2w_dz2_times_dz_dsigma.real() / (s * sigma * sqrt_2pi)
                  - T(2) * dV_dx / sigma;

  // d2V/dx_dgamma:
  c10::complex<T> d2w_dz2_times_dz_dgamma = d2w_dz2 * (i / s);
  T d2V_dx_dgamma = d2w_dz2_times_dz_dgamma.real() / (s * sigma * sqrt_2pi);

  // d2V/dsigma2: complex expression
  // dV/dsigma = -Re[dw_dz * z] / (sigma^2 * sqrt(2*pi)) - V / sigma
  // For the second derivative, we need to differentiate this
  // Simplified approximation using the main terms
  c10::complex<T> d2w_times_z2 = d2w_dz2 * z * (-z / sigma);  // d/dsigma of dw_dz (via chain rule)
  c10::complex<T> dw_times_dz_dsigma = dw_dz * (-T(1) / sigma);  // d/dsigma of z
  T term1 = -(d2w_times_z2.real() + dw_times_dz_dsigma.real() * z.real() - dw_times_dz_dsigma.imag() * z.imag())
            / (sigma * sigma * sqrt_2pi);
  T term2 = T(2) * dw_dz_times_z.real() / (sigma * sigma * sigma * sqrt_2pi);
  T term3 = -dV_dsigma / sigma + V / (sigma * sigma);
  T d2V_dsigma2 = term1 + term2 + term3;

  // d2V/dsigma_dgamma:
  c10::complex<T> d2w_z_dgamma = d2w_dz2 * (i / s) * z + dw_dz * (i / s);
  T d2V_dsigma_dgamma = -d2w_z_dgamma.real() / (sigma * sigma * sqrt_2pi)
                      - dV_dgamma / sigma;

  // d2V/dgamma2:
  c10::complex<T> d2w_dgamma2 = d2w_dz2 * (i / s) * (i / s);
  T d2V_dgamma2 = d2w_dgamma2.real() / (sigma * sqrt_2pi);

  // Compute new gradients
  T new_grad_x = gradient * (gg_x * d2V_dx2 + gg_sigma * d2V_dx_dsigma + gg_gamma * d2V_dx_dgamma);
  T new_grad_sigma = gradient * (gg_x * d2V_dx_dsigma + gg_sigma * d2V_dsigma2 + gg_gamma * d2V_dsigma_dgamma);
  T new_grad_gamma = gradient * (gg_x * d2V_dx_dgamma + gg_sigma * d2V_dsigma_dgamma + gg_gamma * d2V_dgamma2);

  return {gg_out, new_grad_x, new_grad_sigma, new_grad_gamma};
}

}  // namespace torchscience::kernel::special_functions
