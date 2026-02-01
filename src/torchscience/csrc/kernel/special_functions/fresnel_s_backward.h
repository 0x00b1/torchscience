#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

// Backward for fresnel_s: d/dz S(z) = sin(pi*z^2/2)
//
// Derivation:
//   S(z) = integral from 0 to z of sin(pi*t^2/2) dt
//   By the fundamental theorem of calculus:
//   d/dz S(z) = sin(pi*z^2/2)
template <typename T>
T fresnel_s_backward(T gradient, T z) {
  const T pi_over_2 = static_cast<T>(1.5707963267948966);
  T arg = pi_over_2 * z * z;
  T deriv = std::sin(arg);
  return gradient * deriv;
}

// Complex version (c10::complex)
template <typename T>
c10::complex<T> fresnel_s_backward(c10::complex<T> gradient, c10::complex<T> z) {
  const T pi_over_2 = static_cast<T>(1.5707963267948966);
  c10::complex<T> arg = pi_over_2 * z * z;
  c10::complex<T> deriv = c10_complex_math::sin(arg);
  return gradient * deriv;
}

}  // namespace torchscience::kernel::special_functions
