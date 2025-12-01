#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T trigamma(T z) {
  T psi1 = T(0);

  T y = z;

  while (y < T(6)) {
    psi1 += T(1) / (y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);

  return psi1 + (T(1) / y + T(0.5) * y2 + y2 / y * (T(1.0 / 6) - y2 * (T(1.0 / 30) - y2 * T(1.0 / 42))));
}

template <typename T>
c10::complex<T> trigamma(c10::complex<T> z) {
  c10::complex<T> psi1(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    auto sin_piz = sin_pi(y);

    return static_cast<T>(M_PI * M_PI) / (sin_piz * sin_piz) - trigamma(c10::complex<T>(T(1), T(0)) - y);
  }

  while (std::abs(y) < T(6)) {
    psi1 = psi1 + (c10::complex<T>(T(1), T(0)) / (y * y));

    y = y + (c10::complex<T>(T(1), T(0)));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);

  return psi1 + (c10::complex<T>(T(1), T(0)) / y + c10::complex<T>(T(0.5), T(0)) * y2 + y2 / y * (c10::complex<T>(T(1.0 / 6), T(0)) - y2 * (c10::complex<T>(T(1.0 / 30), T(0)) - y2 * c10::complex<T>(T(1.0 / 42), T(0)))));
}

} // namespace torchscience::kernel::special_functions
