#pragma once

#include <cmath>
#include <tuple>

#include "legendre_polynomial_q.h"

namespace torchscience::kernel::special_functions {

// Backward for Legendre function of the second kind Q_n(x)
//
// The derivative with respect to x:
// For integer n, the derivative formula is:
//   dQ_n/dx = (n * x * Q_n(x) - n * Q_{n-1}(x)) / (x^2 - 1)
//
// For n=0:
//   dQ_0/dx = d/dx[arctanh(x)] = 1 / (1 - x^2)
//
// For dQ_n/dn, we use finite differences since the analytical formula
// is complex and involves parameter derivatives.
template <typename T>
std::tuple<T, T> legendre_polynomial_q_backward(T gradient, T x, T n) {
  T one = T(1);
  T eps = T(1e-7);

  // Check for integer n
  T n_floor = std::floor(n + T(0.5));
  T n_diff = std::abs(n - n_floor);
  bool is_integer_n = (n_diff < T(1e-10) && n_floor >= T(0));

  // Compute dQ_n/dx
  T gradient_x;

  if (is_integer_n) {
    int n_int = static_cast<int>(n_floor);

    // Handle singularity at x = +/- 1
    T x_sq_minus_1 = x * x - one;
    if (std::abs(x_sq_minus_1) < T(1e-14)) {
      // Near singularity, the gradient blows up
      gradient_x = gradient * std::numeric_limits<T>::infinity();
    } else if (n_int == 0) {
      // dQ_0/dx = 1 / (1 - x^2) = -1 / (x^2 - 1)
      gradient_x = gradient * (-one / x_sq_minus_1);
    } else {
      // dQ_n/dx = (n * x * Q_n(x) - n * Q_{n-1}(x)) / (x^2 - 1)
      T Q_n = legendre_polynomial_q(x, n);
      T Q_n_minus_1 = legendre_polynomial_q(x, n - one);
      gradient_x = gradient * (n * x * Q_n - n * Q_n_minus_1) / x_sq_minus_1;
    }
  } else {
    // Non-integer n: use finite differences for derivative w.r.t. x
    T Q_plus = legendre_polynomial_q(x + eps, n);
    T Q_minus = legendre_polynomial_q(x - eps, n);
    gradient_x = gradient * (Q_plus - Q_minus) / (T(2) * eps);
  }

  // Compute dQ_n/dn via finite differences
  T Q_plus = legendre_polynomial_q(x, n + eps);
  T Q_minus = legendre_polynomial_q(x, n - eps);
  T gradient_n = gradient * (Q_plus - Q_minus) / (T(2) * eps);

  return {gradient_x, gradient_n};
}

} // namespace torchscience::kernel::special_functions
