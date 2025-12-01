#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T cos_pi(T x) {
  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  return std::cos(static_cast<T>(M_PI) * x_mod);
}

template <typename T>
c10::complex<T> cos_pi(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  return c10::complex<T>(
    std::cos(static_cast<T>(M_PI) * x_mod) * std::cosh(static_cast<T>(M_PI) * y),
    -std::sin(static_cast<T>(M_PI) * x_mod) * std::sinh(static_cast<T>(M_PI) * y)
  );
}

} // namespace torchscience::kernel::special_functions
