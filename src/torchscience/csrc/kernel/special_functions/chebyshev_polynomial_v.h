#pragma once

#include <cmath>

namespace torchscience::kernel::special_functions {

template <typename T>
T chebyshev_polynomial_v(T x, T n) {
  if (std::abs(x) < T(1)) {
    T theta = std::acos(x);

    T half_theta = theta / T(2);

    T cos_half = std::cos(half_theta);

    if (std::abs(cos_half) < T(1e-10)) {
      T sign;

      if (static_cast<int>(n) % 2 == 0) {
        sign = T(1);
      } else {
        sign = T(-1);
      }

      return sign * (T(2) * n + T(1));
    }

    return std::cos((n + T(0.5)) * theta) / cos_half;
  }

  if (x >= T(1)) {
    if (x == T(1)) {
      return T(1);
    }

    T eta = std::acosh(x);

    T half_eta = eta / T(2);

    return std::cosh((n + T(0.5)) * eta) / std::cosh(half_eta);
  }

  if (x == T(-1)) {
    T sign;

    if (static_cast<int>(n) % 2 == 0) {
      sign = T(1);
    } else {
      sign = T(-1);
    }

    return sign * (T(2) * n + T(1));
  }

  T eta = std::acosh(-x);

  T half_eta = eta / T(2);

  T sign;

  if (static_cast<int>(n) % 2 == 0) {
    sign = T(1);
  } else {
    sign = T(-1);
  }

  return sign * std::cosh((n + T(0.5)) * eta) / std::sinh(half_eta);
}

} // namespace torchscience::kernel::special_functions
