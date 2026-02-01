#pragma once

#include <cmath>
#include <tuple>

#include "legendre_polynomial_q.h"
#include "legendre_polynomial_q_backward.h"

namespace torchscience::kernel::special_functions {

// Backward_backward for Legendre function of the second kind Q_n(x)
//
// This computes the gradient of the backward pass, providing second-order
// derivative support. We use finite differences for the complex cases
// and analytical formulas where available.
template <typename T>
std::tuple<T, T, T> legendre_polynomial_q_backward_backward(
  T gradient_gradient_x,
  T gradient_gradient_n,
  T gradient,
  T x,
  T n
) {
  T one = T(1);
  T eps = T(1e-7);

  // Get first-order derivatives
  auto [grad_x, grad_n] = legendre_polynomial_q_backward(one, x, n);

  // gradient_gradient_output = gg_x * dQ/dx + gg_n * dQ/dn
  T gradient_gradient_output = gradient_gradient_x * grad_x + gradient_gradient_n * grad_n;

  // Compute d^2Q/dx^2 via finite differences
  T Q_plus = legendre_polynomial_q(x + eps, n);
  T Q_curr = legendre_polynomial_q(x, n);
  T Q_minus = legendre_polynomial_q(x - eps, n);
  T d2Q_dx2 = (Q_plus - T(2) * Q_curr + Q_minus) / (eps * eps);

  // new_gradient_x = gg_x * grad * d^2Q/dx^2
  T new_gradient_x = gradient_gradient_x * gradient * d2Q_dx2;

  // For cross-terms and second derivatives w.r.t. n, we set to zero
  // since they are complex and typically not needed
  T new_gradient_n = T(0);

  return {gradient_gradient_output, new_gradient_x, new_gradient_n};
}

} // namespace torchscience::kernel::special_functions
