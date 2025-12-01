#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T pentagamma(T z) {
  T psi3 = T(0);

  T y = z;

  while (y < T(6)) {
    psi3 += T(6) / (y * y * y * y);

    y += T(1);
  }

  T y2 = T(1) / (y * y);
  T y3 = y2 / y;
  T y4 = y2 * y2;

  return psi3 + (T(2) * y3 + T(3) * y4 + T(2) * y4 / y - y4 * y3 + T(2) * y4 * y4 / y);
}

template <typename T>
c10::complex<T> pentagamma(c10::complex<T> z) {
  c10::complex<T> psi3(T(0), T(0));

  c10::complex<T> y = z;

  if (y.real() < T(0.5)) {
    // Reflection formula: psi'''(z) - psi'''(1-z) = -2*pi^4*(1 + 2*cos^2(pi*z))/sin^4(pi*z)
    auto sin_piz = sin_pi(y);
    auto cos_piz = cos_pi(y);
    auto sin_piz_sq = sin_piz * sin_piz;
    auto sin_piz_fourth = sin_piz_sq * sin_piz_sq;
    auto cos_piz_sq = cos_piz * cos_piz;
    auto pi_fourth = static_cast<T>(M_PI * M_PI * M_PI * M_PI);

    return -pi_fourth * static_cast<T>(2) * (c10::complex<T>(T(1), T(0)) + static_cast<T>(2) * cos_piz_sq) / sin_piz_fourth + pentagamma(c10::complex<T>(T(1), T(0)) - y);
  }

  while (std::abs(y) < T(6)) {
    psi3 = psi3 + c10::complex<T>(T(6), T(0)) / (y * y * y * y);

    y = y + c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> y2 = c10::complex<T>(T(1), T(0)) / (y * y);
  c10::complex<T> y3 = y2 / y;
  c10::complex<T> y4 = y2 * y2;

  return psi3 + (c10::complex<T>(T(2), T(0)) * y3 + c10::complex<T>(T(3), T(0)) * y4 + c10::complex<T>(T(2), T(0)) * y4 / y - y4 * y3 + c10::complex<T>(T(2), T(0)) * y4 * y4 / y);
}

} // namespace torchscience::kernel::special_functions
