#pragma once

#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

template <typename T>
T gamma_sign(T x) {
  // Sign of Gamma(x) for real x
  // Gamma(x) > 0 for x > 0
  // Gamma(x) alternates sign for x < 0 between consecutive poles
  // Sign is (-1)^n for -n < x < -(n-1) where n >= 1

  if (x > T(0)) {
    return T(1);
  }

  if (x == std::floor(x)) {
    // At poles, sign is undefined - return NaN
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For x < 0, count number of poles between x and 0
  // Poles at 0, -1, -2, ...
  // If -n < x < -(n-1), then n poles are crossed, sign is (-1)^n
  int n = static_cast<int>(-std::floor(x));
  return (n % 2 == 0) ? T(1) : T(-1);
}

// Note: gamma_sign is only defined for real arguments

} // namespace torchscience::kernel::special_functions
