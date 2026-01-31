#pragma once

#include <cmath>
#include <tuple>

#include "erfcinv.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for erfcinv
//
// Let f(x) = erfcinv(x), y = f(x)
// f'(x) = -sqrt(pi)/2 * exp(y^2)
// f''(x) = -sqrt(pi)/2 * 2y * y' * exp(y^2)
//       = -sqrt(pi)/2 * 2y * [-sqrt(pi)/2 * exp(y^2)] * exp(y^2)
//       = pi/2 * y * exp(2*y^2)
//
// Note: The second derivative has the same form as erfinv's second derivative!
// This is because erfcinv(x) = erfinv(1-x), and the second derivative of
// a composition involves the chain rule squared.
//
// The backward function computes: gradient * f'(x)
// We need gradients with respect to both gradient and x.
//
// d/d(gradient) [gradient * f'(x)] = f'(x)
// d/dx [gradient * f'(x)] = gradient * f''(x)
template <typename T>
std::tuple<T, T> erfcinv_backward_backward(
    T gradient_gradient,
    T gradient,
    T x
) {
  const T neg_sqrt_pi_over_2 = static_cast<T>(-0.8862269254527580136490837416705725914);
  const T pi_over_2 = static_cast<T>(1.5707963267948966192313216916397514421);

  T y = erfcinv(x);
  T y2 = y * y;

  // Handle edge cases where y is infinite
  if (std::isinf(y)) {
    T inf = std::numeric_limits<T>::infinity();
    // f'(x) is -inf or +inf depending on sign
    T sign = (y > static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1);
    return {gradient_gradient * sign * inf, gradient_gradient * gradient * inf};
  }

  // First derivative: f'(x) = -sqrt(pi)/2 * exp(y^2)
  T exp_y2 = std::exp(y2);
  T f_prime = neg_sqrt_pi_over_2 * exp_y2;

  // Second derivative: f''(x) = pi/2 * y * exp(2*y^2)
  T exp_2y2 = std::exp(static_cast<T>(2) * y2);
  T f_double_prime = pi_over_2 * y * exp_2y2;

  // Gradient w.r.t. the incoming gradient
  T grad_gradient = gradient_gradient * f_prime;

  // Gradient w.r.t. x
  T grad_x = gradient_gradient * gradient * f_double_prime;

  return {grad_gradient, grad_x};
}

}  // namespace torchscience::kernel::special_functions
