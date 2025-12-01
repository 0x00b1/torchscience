#pragma once

#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T chebyshev_polynomial_u(T x, T n) {
  if (std::abs(x) < T(1)) {
    T theta = std::acos(x);

    return std::sin((n + T(1)) * theta) / std::sin(theta);
  }

  if (x >= T(1)) {
    if (x == T(1)) {
      return n + T(1);
    }

    T eta = std::acosh(x);

    return std::sinh((n + T(1)) * eta) / std::sinh(eta);
  }

  if (x == T(-1)) {
    T sign;
    if (static_cast<int>(n) % 2 == 0) {
      sign = T(1);
    } else {
      sign = T(-1);
    }

    return sign * (n + T(1));
  }

  T eta = std::acosh(-x);

  T sign;

  if (static_cast<int>(n) % 2 == 0) {
    sign = T(1);
  } else {
    sign = T(-1);
  }

  return sign * std::sinh((n + T(1)) * eta) / std::sinh(eta);
}

} // namespace torchscience::kernel::special_functions
