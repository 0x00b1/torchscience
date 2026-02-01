#pragma once

#include <cmath>
#include <tuple>
#include <limits>

#include "inverse_regularized_incomplete_beta.h"
#include "incomplete_beta.h"
#include "log_beta.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> inverse_regularized_incomplete_beta_backward(T gradient, T a, T b, T y) {
  // x = inverse_regularized_incomplete_beta(a, b, y) satisfies I_x(a, b) = y
  //
  // By implicit differentiation of I_x(a, b) = y:
  //
  // For dx/dy:
  //   dI/dx * dx/dy = 1
  //   dx/dy = 1 / (dI/dx)
  //
  // where dI/dx = x^(a-1) * (1-x)^(b-1) / B(a, b)
  //
  // For dx/da and dx/db:
  //   dI/da + dI/dx * dx/da = 0
  //   dx/da = -dI/da / (dI/dx)
  //
  //   dI/db + dI/dx * dx/db = 0
  //   dx/db = -dI/db / (dI/dx)
  //
  // The derivatives dI/da and dI/db involve derivatives of the incomplete
  // beta with respect to the shape parameters, which we compute numerically.

  // Handle edge cases
  if (y <= T(0) || y >= T(1) || a <= T(0) || b <= T(0)) {
    return std::make_tuple(T(0), T(0), T(0));
  }

  // Compute x = inverse_regularized_incomplete_beta(a, b, y)
  T x = inverse_regularized_incomplete_beta(a, b, y);

  if (x <= T(0) || x >= T(1) || !std::isfinite(x)) {
    return std::make_tuple(T(0), T(0), T(0));
  }

  // Compute log B(a, b)
  T log_beta_ab = log_beta(a, b);

  // Compute dI/dx = x^(a-1) * (1-x)^(b-1) / B(a, b)
  T log_dIdx = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x) - log_beta_ab;
  T dIdx = std::exp(log_dIdx);

  // dx/dy = 1 / (dI/dx)
  T dxdy;
  if (dIdx > std::numeric_limits<T>::min()) {
    dxdy = T(1) / dIdx;
  } else {
    dxdy = T(0);  // Derivative undefined
  }

  // For dx/da and dx/db, we need dI/da and dI/db at the solution x
  // Use numerical differentiation
  T eps_a = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1), std::abs(a));
  T eps_b = std::sqrt(std::numeric_limits<T>::epsilon()) *
            std::max(T(1), std::abs(b));

  T I_a_plus = incomplete_beta(x, a + eps_a, b);
  T I_a_minus = incomplete_beta(x, a - eps_a, b);
  T dIda = (I_a_plus - I_a_minus) / (T(2) * eps_a);

  T I_b_plus = incomplete_beta(x, a, b + eps_b);
  T I_b_minus = incomplete_beta(x, a, b - eps_b);
  T dIdb = (I_b_plus - I_b_minus) / (T(2) * eps_b);

  // dx/da = -dI/da / (dI/dx)
  // dx/db = -dI/db / (dI/dx)
  T dxda, dxdb;
  if (dIdx > std::numeric_limits<T>::min()) {
    dxda = -dIda / dIdx;
    dxdb = -dIdb / dIdx;
  } else {
    dxda = T(0);
    dxdb = T(0);
  }

  T grad_a = gradient * dxda;
  T grad_b = gradient * dxdb;
  T grad_y = gradient * dxdy;

  return std::make_tuple(grad_a, grad_b, grad_y);
}

} // namespace torchscience::kernel::special_functions
