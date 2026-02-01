#pragma once

#include <cmath>
#include <limits>

#include "erfinv.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Newton-Raphson refinement step for erfcinv
// erfc'(y) = -2/sqrt(pi) * exp(-y^2)
// erfcinv'(x) = 1/erfc'(erfcinv(x)) = -sqrt(pi)/2 * exp(erfcinv(x)^2)
// Newton step: y_new = y - (erfc(y) - x) / erfc'(y)
//            = y - (erfc(y) - x) * (-sqrt(pi)/2) * exp(y^2)
//            = y + (erfc(y) - x) * sqrt(pi)/2 * exp(y^2)
template <typename T>
T erfcinv_newton_refine(T x, T y) {
  const T sqrt_pi_over_2 = static_cast<T>(0.8862269254527580136490837416705725914);

  // Compute erfc(y) using std::erfc
  T erfc_y = std::erfc(y);
  T exp_y2 = std::exp(y * y);

  // Newton step
  T delta = (erfc_y - x) * sqrt_pi_over_2 * exp_y2;
  return y + delta;
}

// Asymptotic expansion for erfcinv(x) when x is very small
// For small x: erfcinv(x) ~ sqrt(-log(x * sqrt(pi)))
// with higher-order corrections
template <typename T>
T erfcinv_asymptotic(T x) {
  const T sqrt_pi = static_cast<T>(1.7724538509055160272981674833411451828);

  // Initial guess: erfcinv(x) ~ sqrt(-log(x * sqrt(pi)))
  T u = -std::log(x * sqrt_pi);
  T y = std::sqrt(u);

  // Corrections for better accuracy
  // Use series expansion around large y
  T inv_y = static_cast<T>(1) / y;
  T inv_y2 = inv_y * inv_y;

  // Asymptotic series coefficients
  T correction = inv_y * (static_cast<T>(0.5) * std::log(u) / u);
  y = y - correction;

  return y;
}

}  // namespace detail

// Inverse complementary error function: erfcinv(x) returns y such that erfc(y) = x
// Domain: x in (0, 2)
// Range: all real numbers
// Special cases:
//   erfcinv(0) = +inf
//   erfcinv(1) = 0
//   erfcinv(2) = -inf
//   erfcinv(x < 0 or x > 2) = NaN
//
// Relation to erfinv: erfcinv(x) = erfinv(1 - x)
template <typename T>
T erfcinv(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == static_cast<T>(0)) {
    return std::numeric_limits<T>::infinity();
  }

  if (x == static_cast<T>(2)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (x < static_cast<T>(0) || x > static_cast<T>(2)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == static_cast<T>(1)) {
    return static_cast<T>(0);
  }

  T y;

  // For x near 0, use asymptotic expansion with Newton refinement
  if (x < static_cast<T>(1e-100)) {
    y = detail::erfcinv_asymptotic(x);
    // Newton refinement
    y = detail::erfcinv_newton_refine(x, y);
    y = detail::erfcinv_newton_refine(x, y);
    return y;
  }

  // For most values, use the relation erfcinv(x) = erfinv(1 - x)
  // This works well when 1 - x is not too close to +/-1
  y = erfinv(static_cast<T>(1) - x);

  // Newton refinement for full precision (2 iterations)
  y = detail::erfcinv_newton_refine(x, y);
  y = detail::erfcinv_newton_refine(x, y);

  return y;
}

}  // namespace torchscience::kernel::special_functions
