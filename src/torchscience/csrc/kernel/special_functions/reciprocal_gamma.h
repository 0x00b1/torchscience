#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T reciprocal_gamma(T z) {
  // 1/Gamma(z) is entire - no poles, returns 0 at non-positive integers
  T g = gamma(z);
  if (std::isinf(g)) {
    return T(0);
  }
  return T(1) / g;
}

template <typename T>
c10::complex<T> reciprocal_gamma(c10::complex<T> z) {
  c10::complex<T> g = gamma(z);
  // Check for infinity (pole of gamma)
  if (std::isinf(g.real()) || std::isinf(g.imag())) {
    return c10::complex<T>(T(0), T(0));
  }
  return c10::complex<T>(T(1), T(0)) / g;
}

} // namespace torchscience::kernel::special_functions
