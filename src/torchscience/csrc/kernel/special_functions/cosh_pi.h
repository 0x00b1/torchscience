#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T cosh_pi(T x) {
  return std::cosh(static_cast<T>(M_PI) * x);
}

template <typename T>
c10::complex<T> cosh_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  return c10::complex<T>(std::cosh(static_cast<T>(M_PI) * x) * std::cos(static_cast<T>(M_PI) * y), std::sinh(static_cast<T>(M_PI) * x) * std::sin(static_cast<T>(M_PI) * y));
}

} // namespace torchscience::kernel::special_functions
