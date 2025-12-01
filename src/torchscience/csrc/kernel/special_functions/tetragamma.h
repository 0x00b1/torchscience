#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tetragamma(T z) {
  T psi2 = T(0);

  T y = z;

  while (y < T(6)) {
    psi2 -= T(2) / (y * y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);
  T y3 = y2 / y;

  // Asymptotic expansion: psi''(y) = -1/y^2 - 1/y^3 - 1/(2y^4) + 1/(6y^6) - 1/(6y^8) + ...
  psi2 += -y2 - y3 - y2 * y2 * (T(0.5) - y2 * (T(1.0/6) - y2 * T(1.0/6)));

  return psi2;
}

template <typename T>
c10::complex<T> tetragamma(c10::complex<T> z) {
  c10::complex<T> psi2(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    return static_cast<T>(M_PI * M_PI * M_PI) * static_cast<T>(2) * cos_pi(y) / (sin_pi(y) * sin_pi(y) * sin_pi(y)) - tetragamma(c10::complex<T>(T(1), T(0)) - y);
  }

  while (std::abs(y) < T(6)) {
    psi2 = psi2 - c10::complex<T>(T(2), T(0)) / (y * y * y);

    y = y + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);
  c10::complex<T> y3 = y2 / y;

  return psi2 + (-y2 - y3 - y2 * y2 * (c10::complex<T>(T(0.5), T(0)) - y2 * (c10::complex<T>(T(1.0/6), T(0)) - y2 * c10::complex<T>(T(1.0/6), T(0)))));
}

} // namespace torchscience::kernel::special_functions
