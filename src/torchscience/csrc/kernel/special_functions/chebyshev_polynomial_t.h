#pragma once

#include <cmath>

namespace torchscience::kernel::special_functions {

// Chebyshev polynomial of the first kind T_n(x)
//
// T_n(x) = cos(n * acos(x)) for |x| <= 1
// T_n(x) = cosh(n * acosh(x)) for x > 1
// T_n(x) = cos(n * pi) * cosh(n * acosh(-x)) for x < -1
template <typename T>
T chebyshev_polynomial_t(T x, T n) {
  const T pi = T(3.14159265358979323846);

  if (std::abs(x) <= T(1)) {
    return std::cos(n * std::acos(x));
  }

  if (x > T(1)) {
    return std::cosh(n * std::acosh(x));
  }

  // x < -1: Use cos(n*pi) instead of (-1)^n to handle non-integer n
  return std::cos(n * pi) * std::cosh(n * std::acosh(-x));
}

} // namespace torchscience::kernel::special_functions
