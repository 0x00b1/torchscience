#pragma once

#include <tuple>

#include "chebyshev_polynomial_t.h"
#include "chebyshev_polynomial_u.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> chebyshev_polynomial_u_backward(T gradient, T x, T n) {
  T x2_minus_1 = x * x - T(1);

  T gradient_x;

  if (std::abs(x2_minus_1) < T(1e-10)) {
    T sign;

    if (x > T(0)) {
      sign = T(1);
    } else {
      if (static_cast<int>(n) % 2 == 0) {
        sign = T(-1);
      } else {
        sign = T(1);
      }
    }

    gradient_x = gradient * sign * (n + T(1)) * n * (n + T(2)) / T(3);
  } else {
    gradient_x = gradient * ((n + T(1)) * chebyshev_polynomial_t(x, n + T(1)) - x * chebyshev_polynomial_u(x, n)) / x2_minus_1;
  }

  return {
    gradient_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
