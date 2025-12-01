#pragma once

#include <tuple>

#include "chebyshev_polynomial_w.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> chebyshev_polynomial_v_backward(T gradient, T x, T n) {
  T gradient_x;

  if (n < T(1)) {
    gradient_x = T(0);
  } else {
    gradient_x = gradient * (T(2) * n + T(1)) * chebyshev_polynomial_w(x, n - T(1)) / T(2);
  }

  return {
    gradient_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
