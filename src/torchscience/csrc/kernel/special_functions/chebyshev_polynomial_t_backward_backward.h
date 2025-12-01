#pragma once

#include <tuple>

#include "chebyshev_polynomial_v.h"
#include "chebyshev_polynomial_w.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> chebyshev_polynomial_t_backward_backward(
  T gradient_gradient_x,
  T gradient_gradient_n,
  T gradient,
  T x,
  T n
) {
  if (n <= T(0)) {
    return {
      T(0),
      T(0),
      T(0)
    };
  }

  T gradient_gradient_output = gradient_gradient_x * n * ((chebyshev_polynomial_v(x, n - T(1)) + chebyshev_polynomial_w(x, n - T(1))) / T(2));

  T m = n - T(1);
  T coeff = T(2) * m + T(1);

  T new_grad_x;
  T one_minus_x = T(1) - x;

  if (m < T(1)) {
    if (std::abs(one_minus_x) < T(1e-10)) {
      new_grad_x = gradient_gradient_x * gradient * n * ((T(0) + T(0.5)) / T(2));
    } else {
      new_grad_x = gradient_gradient_x * gradient * n * ((T(0) + (chebyshev_polynomial_v(x, T(0)) / T(2) + chebyshev_polynomial_w(x, n - T(1)) / (T(2) * one_minus_x))) / T(2));
    }
  } else {
    if (std::abs(one_minus_x) < T(1e-10)) {
      new_grad_x = gradient_gradient_x * gradient * n * ((coeff * chebyshev_polynomial_w(x, m - T(1)) / T(2) + coeff * chebyshev_polynomial_v(x, n - T(1)) / T(2)) / T(2));
    } else {
      new_grad_x = gradient_gradient_x * gradient * n * ((coeff * chebyshev_polynomial_w(x, m - T(1)) / T(2) + (coeff * chebyshev_polynomial_v(x, n - T(1)) / T(2) + chebyshev_polynomial_w(x, n - T(1)) / (T(2) * one_minus_x))) / T(2));
    }
  }

  return {
    gradient_gradient_output,
    new_grad_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
