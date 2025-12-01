#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T tanh_pi(T x) {
  return std::tanh(static_cast<T>(M_PI) * x);
}

template <typename T>
c10::complex<T> tanh_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  T two_pi_x = static_cast<T>(2 * M_PI) * x;
  T two_pi_y = static_cast<T>(2 * M_PI) * y;

  T denom = std::cosh(two_pi_x) + std::cos(two_pi_y);

  return c10::complex<T>(std::sinh(two_pi_x) / denom, std::sin(two_pi_y) / denom);
}

} // namespace torchscience::kernel::special_functions
