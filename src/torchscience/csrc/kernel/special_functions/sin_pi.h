#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
c10::complex<T> sin_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  T sin_x = std::sin(static_cast<T>(M_PI) * x_mod);
  T cos_x = std::cos(static_cast<T>(M_PI) * x_mod);

  T sinh_y = std::sinh(static_cast<T>(M_PI) * y);
  T cosh_y = std::cosh(static_cast<T>(M_PI) * y);

  return c10::complex<T>(sin_x * cosh_y, cos_x * sinh_y);
}

} // namespace torchscience::kernel::special_functions
