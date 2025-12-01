#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T digamma(T z) {
  T psi = T(0);

  T x = z;

  while (x < T(6)) {
    psi -= T(1) / x;

    x += T(1);
  }

  T x2 = T(1) / (x * x);

  return psi + (std::log(x) - T(0.5) / x - x2 * (T(1.0 / 12) - x2 * (T(1.0 / 120) - x2 * T(1.0 / 252))));
}

template <typename T>
c10::complex<T> digamma(c10::complex<T> z) {
  c10::complex<T> psi(T(0), T(0));

  c10::complex<T> x = z;

  if (x.real() < T(0.5)) {
    return digamma(c10::complex<T>(T(1), T(0)) - x) + static_cast<T>(M_PI) * sin_pi(x - c10::complex<T>(T(0.5), T(0))) / sin_pi(x);
  }

  while (std::abs(x) < T(6)) {
    psi = psi - c10::complex<T>(T(1), T(0)) / x;

    x = x + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> x2 = c10::complex<T>(T(1), T(0)) / (x * x);

  return psi + (std::log(x) - c10::complex<T>(T(0.5), T(0)) / x - x2 * (c10::complex<T>(T(1.0 / 12), T(0)) - x2 * (c10::complex<T>(T(1.0 / 120), T(0)) - x2 * c10::complex<T>(T(1.0 / 252), T(0)))));
}

} // namespace torchscience::kernel::special_functions
