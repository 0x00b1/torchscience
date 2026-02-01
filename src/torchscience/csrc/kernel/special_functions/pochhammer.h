#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T pochhammer(T z, T m) {
  // Pochhammer symbol (z)_m = Gamma(z+m) / Gamma(z)
  // Computed as exp(log_gamma(z+m) - log_gamma(z)) for numerical stability

  // Special cases
  if (m == T(0)) {
    return T(1);  // (z)_0 = 1 for all z
  }

  // For general case, use log-gamma
  T log_result = log_gamma(z + m) - log_gamma(z);
  return std::exp(log_result);
}

template <typename T>
c10::complex<T> pochhammer(c10::complex<T> z, c10::complex<T> m) {
  if (m.real() == T(0) && m.imag() == T(0)) {
    return c10::complex<T>(T(1), T(0));
  }

  c10::complex<T> log_result = log_gamma(z + m) - log_gamma(z);
  return std::exp(log_result);
}

} // namespace torchscience::kernel::special_functions
