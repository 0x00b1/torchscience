#pragma once

#include <tuple>

#include "chebyshev_polynomial_v.h"
#include "chebyshev_polynomial_w.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> chebyshev_polynomial_v_backward_backward(
  T gradient_gradient_x,
  T gradient_gradient_n,
  T gradient,
  T x,
  T n
) {
  if (n < T(1)) {
    return {T(0), T(0), T(0)};
  }

  T coeff = (T(2) * n + T(1)) / T(2);

  T w_nm1 = chebyshev_polynomial_w(x, n - T(1));

  T gg_out = gradient_gradient_x * coeff * w_nm1;

  T new_grad_x;

  T one_minus_x = T(1) - x;

  if (n < T(2)) {
    if (std::abs(one_minus_x) < T(1e-10)) {
      new_grad_x = T(0);
    } else {
      new_grad_x = gradient_gradient_x * gradient * coeff * (chebyshev_polynomial_v(x, T(0)) / T(2) + chebyshev_polynomial_w(x, T(0)) / (T(2) * one_minus_x));
    }
  } else {
    T m = n - T(1);

    if (std::abs(one_minus_x) < T(1e-10)) {
      new_grad_x = gradient_gradient_x * gradient * coeff * ((T(2) * m + T(1)) * chebyshev_polynomial_v(x, m) / T(2));
    } else {
      new_grad_x = gradient_gradient_x * gradient * coeff * ((T(2) * m + T(1)) * chebyshev_polynomial_v(x, m) / T(2) + w_nm1 / (T(2) * one_minus_x));
    }
  }

  return {
    gg_out,
    new_grad_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
