#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "faddeeva_w.h"
#include "voigt_profile.h"

namespace torchscience::kernel::special_functions {

// Gradients of the Voigt profile V(x, sigma, gamma)
//
// V = Re[w(z)] / (sigma * sqrt(2*pi))
// where z = (x + i*gamma) / (sigma * sqrt(2))
//
// Let s = sigma * sqrt(2), then z = (x + i*gamma) / s
//
// dV/dx = (1/s) * Re[dw/dz] / (sigma * sqrt(2*pi))
//       = Re[dw/dz] / (sigma^2 * sqrt(4*pi))
//
// Since dw/dz = -2z*w(z) + 2i/sqrt(pi), we have:
// dw/dz = -2z*w + 2i/sqrt(pi)
//
// For the x derivative:
// dV/dx = (1/(sigma*sqrt(2))) * Re[-2z*w + 2i/sqrt(pi)] / (sigma*sqrt(2*pi))
//       = Re[-2z*w + 2i/sqrt(pi)] / (sigma^2 * sqrt(4*pi))
//
// For sigma:
// dV/dsigma = d/dsigma [ Re[w(z)] / (sigma*sqrt(2*pi)) ]
//           = [ Re[dw/dz * dz/dsigma] / (sigma*sqrt(2*pi)) ]
//             - [ Re[w(z)] / (sigma^2 * sqrt(2*pi)) ]
// where dz/dsigma = -(x + i*gamma) / (sigma^2 * sqrt(2)) = -z/sigma
//
// For gamma:
// dV/dgamma = Re[dw/dz * dz/dgamma] / (sigma*sqrt(2*pi))
// where dz/dgamma = i / (sigma*sqrt(2))
template <typename T>
std::tuple<T, T, T> voigt_profile_backward(T gradient, T x, T sigma, T gamma) {
  const T sqrt_2 = static_cast<T>(1.4142135623730950488016887242096980786);
  const T sqrt_pi = static_cast<T>(1.7724538509055160272981674833411451828);
  const T sqrt_2pi = static_cast<T>(2.5066282746310005024157652848110452530);
  const T one_over_sqrt_2pi = static_cast<T>(0.39894228040143267793994605993438186848);

  // Handle invalid parameters
  if (sigma <= T(0) || gamma < T(0)) {
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    return {nan_val, nan_val, nan_val};
  }

  T s = sigma * sqrt_2;  // sigma * sqrt(2)
  c10::complex<T> z(x / s, gamma / s);

  c10::complex<T> w_z = faddeeva_w(z);

  // dw/dz = -2z*w(z) + 2i/sqrt(pi)
  c10::complex<T> two_i_sqrt_pi(T(0), T(2) / sqrt_pi);
  c10::complex<T> dw_dz = -T(2) * z * w_z + two_i_sqrt_pi;

  // V = Re[w(z)] / (sigma * sqrt(2*pi))
  T V = w_z.real() * one_over_sqrt_2pi / sigma;

  // dV/dx: dz/dx = 1/s = 1/(sigma*sqrt(2))
  // dV/dx = Re[dw_dz] * (1/s) / (sigma * sqrt(2*pi))
  //       = Re[dw_dz] / (sigma^2 * sqrt(4*pi))
  T dV_dx = dw_dz.real() / (s * sigma * sqrt_2pi);

  // dV/dsigma: dz/dsigma = -z/sigma
  // dV/dsigma = Re[dw_dz * (-z/sigma)] / (sigma*sqrt(2*pi)) - V/sigma
  //           = -Re[dw_dz * z] / (sigma^2 * sqrt(2*pi)) - V/sigma
  c10::complex<T> dw_dz_times_z = dw_dz * z;
  T dV_dsigma = -dw_dz_times_z.real() / (sigma * sigma * sqrt_2pi) - V / sigma;

  // dV/dgamma: dz/dgamma = i/s = i/(sigma*sqrt(2))
  // dV/dgamma = Re[dw_dz * i/s] / (sigma*sqrt(2*pi))
  //           = -Im[dw_dz] / (s * sigma * sqrt(2*pi))
  T dV_dgamma = -dw_dz.imag() / (s * sigma * sqrt_2pi);

  return {gradient * dV_dx, gradient * dV_dsigma, gradient * dV_dgamma};
}

}  // namespace torchscience::kernel::special_functions
