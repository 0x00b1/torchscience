#pragma once

#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T chebyshev_polynomial_w(T x, T n) {
  if (std::abs(x) < T(1)) {
    T theta = std::acos(x);

    T sin_half = std::sin(theta / T(2));

    if (std::abs(sin_half) < T(1e-10)) {
      return T(2) * n + T(1);
    }

    return std::sin((n + T(0.5)) * theta) / sin_half;
  }

  if (x >= T(1)) {
    if (x == T(1)) {
      return T(2) * n + T(1);
    }

    T eta = std::acosh(x);

    return std::sinh((n + T(0.5)) * eta) / std::sinh(eta / T(2));
  }

  if (x == T(-1)) {
    T sign;

    if (static_cast<int>(n) % 2 == 0) {
      sign = T(1);
    } else {
      sign = T(-1);
    }

    return sign;
  }

  T eta = std::acosh(-x);

  T sign;

  if (static_cast<int>(n) % 2 == 0) {
    sign = T(1);
  } else {
    sign = T(-1);
  }

  return sign * std::sinh((n + T(0.5)) * eta) / std::cosh(eta / T(2));
}

} // namespace torchscience::kernel::special_functions
