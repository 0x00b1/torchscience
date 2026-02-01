#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "inverse_regularized_gamma_p.h"
#include "regularized_gamma_p.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> inverse_regularized_gamma_p_backward(T gradient, T a, T y) {
  // x = inverse_regularized_gamma_p(a, y) satisfies P(a, x) = y
  //
  // By implicit differentiation of P(a, x) = y:
  //
  // For dx/dy:
  //   dP/dx * dx/dy = 1
  //   dx/dy = 1 / (dP/dx)
  //
  // where dP/dx = x^(a-1) * e^(-x) / Gamma(a)
  //
  // For dx/da:
  //   dP/da + dP/dx * dx/da = 0
  //   dx/da = -dP/da / (dP/dx)
  //
  // The derivative dP/da involves the derivative of the incomplete gamma
  // with respect to the shape parameter, which is complex. We use numerical
  // differentiation for this.

  // Handle edge cases
  if (y <= T(0) || y >= T(1) || a <= T(0)) {
    return std::make_tuple(T(0), T(0));
  }

  // Compute x = inverse_regularized_gamma_p(a, y)
  T x = inverse_regularized_gamma_p(a, y);

  if (x <= T(0) || !std::isfinite(x)) {
    return std::make_tuple(T(0), T(0));
  }

  // Compute dP/dx = x^(a-1) * e^(-x) / Gamma(a)
  T log_gamma_a = std::lgamma(a);
  T log_dPdx = (a - T(1)) * std::log(x) - x - log_gamma_a;
  T dPdx = std::exp(log_dPdx);

  // dx/dy = 1 / (dP/dx) = Gamma(a) * e^x / x^(a-1)
  T dxdy;
  if (dPdx > std::numeric_limits<T>::min()) {
    dxdy = T(1) / dPdx;
  } else {
    dxdy = T(0);  // Derivative undefined
  }

  // For dx/da, we need dP/da at the solution x
  // Use numerical differentiation
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) *
          std::max(T(1), std::abs(a));

  T p_plus = regularized_gamma_p(a + eps, x);
  T p_minus = regularized_gamma_p(a - eps, x);
  T dPda = (p_plus - p_minus) / (T(2) * eps);

  // dx/da = -dP/da / (dP/dx)
  T dxda;
  if (dPdx > std::numeric_limits<T>::min()) {
    dxda = -dPda / dPdx;
  } else {
    dxda = T(0);  // Derivative undefined
  }

  T grad_a = gradient * dxda;
  T grad_y = gradient * dxdy;

  return std::make_tuple(grad_a, grad_y);
}

} // namespace torchscience::kernel::special_functions
