#pragma once

#include <cmath>
#include <tuple>

#include "chebyshev_polynomial_t.h"
#include "chebyshev_polynomial_u.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> chebyshev_polynomial_u_backward_backward(
  T gradient_gradient_x,
  T gradient_gradient_n,
  T gradient,
  T x,
  T n
) {
  // Forward: U_n(x) = sin((n+1)*theta)/sin(theta) where theta = acos(x)
  // First derivative: dU_n/dx = ((n+1)*T_{n+1}(x) - x*U_n(x)) / (x^2 - 1)
  //
  // Second derivative: d^2U_n/dx^2 = ...
  // This is computed by differentiating dU_n/dx with respect to x.
  //
  // Using the recurrence relation approach for numerical stability.
  //
  // gradient_gradient_x: incoming gradient w.r.t. gradient_x from backward
  // gradient_gradient_n: incoming gradient w.r.t. gradient_n (always 0 since n grad is 0)
  // gradient: the original gradient from the forward pass backward
  // x, n: original inputs
  //
  // Returns:
  //   gradient_gradient_output: gradient flowing back to the output gradient
  //   new_grad_x: gradient flowing back to x
  //   new_grad_n: gradient flowing back to n (always 0)

  T x2_minus_1 = x * x - T(1);

  if (std::abs(x2_minus_1) < T(1e-10)) {
    // At x = ±1, use limiting form
    // d^2U_n/dx^2 at x = ±1 requires special treatment
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

    // At x = ±1: dU_n/dx = sign * (n+1)*n*(n+2)/3
    // d^2U_n/dx^2 at x = ±1 involves higher order terms
    // For simplicity, use a finite difference approximation implicitly
    // by returning the first derivative contribution only
    T grad_x_val = sign * (n + T(1)) * n * (n + T(2)) / T(3);

    // Second derivative at boundary - use Taylor expansion limit
    // d^2U_n/dx^2 at x=1 = (n+1)*n^2*(n+2)*(n+3)/15
    T second_deriv;
    if (x > T(0)) {
      second_deriv = (n + T(1)) * n * n * (n + T(2)) * (n + T(3)) / T(15);
    } else {
      second_deriv = sign * (n + T(1)) * n * n * (n + T(2)) * (n + T(3)) / T(15);
    }

    T gradient_gradient_output = gradient_gradient_x * grad_x_val;
    T new_grad_x = gradient_gradient_x * gradient * second_deriv;

    return {
      gradient_gradient_output,
      new_grad_x,
      T(0)
    };
  }

  // General case: |x| < 1 or |x| > 1 (away from boundaries)
  // dU_n/dx = ((n+1)*T_{n+1}(x) - x*U_n(x)) / (x^2 - 1)

  T T_np1 = chebyshev_polynomial_t(x, n + T(1));
  T U_n = chebyshev_polynomial_u(x, n);
  T U_np1 = chebyshev_polynomial_u(x, n + T(1));

  // First derivative
  T dUn_dx = ((n + T(1)) * T_np1 - x * U_n) / x2_minus_1;

  // Second derivative via chain rule on the first derivative formula
  // d/dx[((n+1)*T_{n+1}(x) - x*U_n(x)) / (x^2 - 1)]
  // Using quotient rule: (f'/g - f*g'/g^2) where
  // f = (n+1)*T_{n+1} - x*U_n
  // g = x^2 - 1
  // f' = (n+1)*dT_{n+1}/dx - (U_n + x*dU_n/dx)
  //    = (n+1)*(n+1)*U_n - U_n - x*dU_n/dx
  //    = ((n+1)^2 - 1)*U_n - x*dU_n/dx
  //    = n*(n+2)*U_n - x*dU_n/dx
  // g' = 2x

  T f = (n + T(1)) * T_np1 - x * U_n;
  T g = x2_minus_1;
  T f_prime = n * (n + T(2)) * U_n - x * dUn_dx;
  T g_prime = T(2) * x;

  T d2Un_dx2 = (f_prime * g - f * g_prime) / (g * g);

  // gradient_gradient_output = gg_x * dUn/dx
  T gradient_gradient_output = gradient_gradient_x * dUn_dx;

  // new_grad_x = gg_x * gradient * d2Un/dx2
  T new_grad_x = gradient_gradient_x * gradient * d2Un_dx2;

  return {
    gradient_gradient_output,
    new_grad_x,
    T(0)
  };
}

} // namespace torchscience::kernel::special_functions
