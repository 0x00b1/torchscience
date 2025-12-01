#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T sinh_pi(T x) {
  return std::sinh(static_cast<T>(M_PI) * x);
}

template <typename T>
c10::complex<T> sinh_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  T pi_x = static_cast<T>(M_PI) * x;
  T pi_y = static_cast<T>(M_PI) * y;

  T sinh_x = std::sinh(pi_x);
  T cosh_x = std::cosh(pi_x);

  T sin_y = std::sin(pi_y);
  T cos_y = std::cos(pi_y);

  return c10::complex<T>(sinh_x * cos_y, cosh_x * sin_y);
}

} // namespace torchscience::kernel::special_functions
