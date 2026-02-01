#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "faddeeva_w.h"

namespace torchscience::kernel::special_functions {

// Dawson's integral (Dawson function)
// D(z) = exp(-z^2) * integral from 0 to z of exp(t^2) dt
//
// For real x: D(x) = sqrt(pi)/2 * Im[w(x)]
// where w(z) is the Faddeeva function.
//
// For complex z: D(z) = sqrt(pi)/2 * i * (exp(-z^2) - w(z))
// Equivalently: D(z) = -i * sqrt(pi)/2 * (w(z) - exp(-z^2))
//
// Properties:
//   D(0) = 0
//   D(-z) = -D(z) (odd function)
//   D(x) has maximum at x ~ 0.924 where D(x) ~ 0.541
//   D(x) -> 1/(2x) as x -> infinity (asymptotic)
template <typename T>
T dawson(T x) {
  const T sqrt_pi_over_2 = static_cast<T>(0.88622692545275801364908374167057259139);

  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == T(0)) {
    return T(0);
  }

  // For real x, D(x) = sqrt(pi)/2 * Im[w(x)]
  // w(x) for real x gives exp(-x^2) + 2i/sqrt(pi) * D(x)
  // So Im[w(x)] = 2/sqrt(pi) * D(x)
  // Therefore D(x) = sqrt(pi)/2 * Im[w(x)]
  c10::complex<T> w_x = faddeeva_w(x);
  return sqrt_pi_over_2 * w_x.imag();
}

// Complex version of Dawson's integral
template <typename T>
c10::complex<T> dawson(c10::complex<T> z) {
  const T sqrt_pi_over_2 = static_cast<T>(0.88622692545275801364908374167057259139);

  // Handle z = 0
  if (z.real() == T(0) && z.imag() == T(0)) {
    return c10::complex<T>(T(0), T(0));
  }

  // D(z) = sqrt(pi)/2 * i * (exp(-z^2) - w(z))
  //      = -i * sqrt(pi)/2 * (w(z) - exp(-z^2))
  c10::complex<T> w_z = faddeeva_w(z);
  c10::complex<T> exp_neg_z2 = std::exp(-z * z);
  c10::complex<T> i(T(0), T(1));

  return sqrt_pi_over_2 * i * (exp_neg_z2 - w_z);
}

}  // namespace torchscience::kernel::special_functions
