#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "faddeeva_w.h"

namespace torchscience::kernel::special_functions {

// Voigt profile (Voigt function)
// V(x, sigma, gamma) = Re[w(z)] / (sigma * sqrt(2*pi))
// where z = (x + i*gamma) / (sigma * sqrt(2))
//
// This is the convolution of a Gaussian with standard deviation sigma
// and a Lorentzian with half-width at half-maximum gamma.
//
// Properties:
//   V(x, sigma, gamma) is a valid probability distribution (integrates to 1)
//   V(x, sigma, 0) = Gaussian with std dev sigma
//   V(x, 0, gamma) = Lorentzian with HWHM gamma (when sigma -> 0)
//   V(-x, sigma, gamma) = V(x, sigma, gamma) (even in x)
//   V > 0 for all x when sigma > 0, gamma >= 0
//
// Parameters:
//   x: position (can be any real number)
//   sigma: Gaussian standard deviation (must be > 0)
//   gamma: Lorentzian half-width at half-maximum (must be >= 0)
template <typename T>
T voigt_profile(T x, T sigma, T gamma) {
  const T sqrt_2 = static_cast<T>(1.4142135623730950488016887242096980786);
  const T one_over_sqrt_2pi = static_cast<T>(0.39894228040143267793994605993438186848);

  // Handle invalid parameters
  if (sigma <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (gamma < T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Handle special cases
  if (std::isnan(x) || std::isnan(sigma) || std::isnan(gamma)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Compute z = (x + i*gamma) / (sigma * sqrt(2))
  T sigma_sqrt2 = sigma * sqrt_2;
  c10::complex<T> z(x / sigma_sqrt2, gamma / sigma_sqrt2);

  // V(x, sigma, gamma) = Re[w(z)] / (sigma * sqrt(2*pi))
  c10::complex<T> w_z = faddeeva_w(z);

  return w_z.real() * one_over_sqrt_2pi / sigma;
}

}  // namespace torchscience::kernel::special_functions
