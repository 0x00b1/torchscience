#pragma once

#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Coefficients for the central region approximation (|x| < 0.9959)
// Based on Winitzki's rational approximation with high-precision coefficients
template <typename T>
T erfinv_central(T x) {
  // w = -log(1 - x^2)
  T w = -std::log((static_cast<T>(1) - x) * (static_cast<T>(1) + x));
  T w_adj = w - static_cast<T>(2.5);

  // Rational approximation coefficients
  constexpr double a0 = 1.50140941;
  constexpr double a1 = 0.246640727;
  constexpr double a2 = -0.00417768164;
  constexpr double a3 = -0.00125372503;
  constexpr double a4 = 0.00021858087;
  constexpr double a5 = -4.39150654e-6;
  constexpr double a6 = -3.5233877e-6;
  constexpr double a7 = 3.43273939e-7;
  constexpr double a8 = 2.81022636e-8;

  T p = static_cast<T>(a8);
  p = p * w_adj + static_cast<T>(a7);
  p = p * w_adj + static_cast<T>(a6);
  p = p * w_adj + static_cast<T>(a5);
  p = p * w_adj + static_cast<T>(a4);
  p = p * w_adj + static_cast<T>(a3);
  p = p * w_adj + static_cast<T>(a2);
  p = p * w_adj + static_cast<T>(a1);
  p = p * w_adj + static_cast<T>(a0);

  return p * x;
}

// Coefficients for the tail region approximation (|x| >= 0.9959)
template <typename T>
T erfinv_tail(T x) {
  T sign = (x >= static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(-1);
  T ax = std::abs(x);

  // w = -log(1 - x^2) = -log((1-|x|)(1+|x|))
  T w = -std::log((static_cast<T>(1) - ax) * (static_cast<T>(1) + ax));
  T w_adj = std::sqrt(w) - static_cast<T>(3.0);

  // Rational approximation coefficients for tail
  constexpr double b0 = 2.83297682;
  constexpr double b1 = 1.00167406;
  constexpr double b2 = 0.00943887047;
  constexpr double b3 = -0.0076224613;
  constexpr double b4 = 0.00573950773;
  constexpr double b5 = -0.00367342844;
  constexpr double b6 = 0.00134934322;
  constexpr double b7 = 0.000100950558;
  constexpr double b8 = -0.000200214257;

  T p = static_cast<T>(b8);
  p = p * w_adj + static_cast<T>(b7);
  p = p * w_adj + static_cast<T>(b6);
  p = p * w_adj + static_cast<T>(b5);
  p = p * w_adj + static_cast<T>(b4);
  p = p * w_adj + static_cast<T>(b3);
  p = p * w_adj + static_cast<T>(b2);
  p = p * w_adj + static_cast<T>(b1);
  p = p * w_adj + static_cast<T>(b0);

  return sign * p;
}

// Newton-Raphson refinement step for erfinv
// erf'(y) = 2/sqrt(pi) * exp(-y^2)
// erfinv'(x) = 1/erf'(erfinv(x)) = sqrt(pi)/2 * exp(erfinv(x)^2)
// Newton step: y_new = y - (erf(y) - x) / erf'(y)
//            = y + (x - erf(y)) * sqrt(pi)/2 * exp(y^2)
template <typename T>
T erfinv_newton_refine(T x, T y) {
  const T sqrt_pi_over_2 = static_cast<T>(0.8862269254527580136490837416705725914);

  // Compute erf(y) using std::erf
  T erf_y = std::erf(y);
  T exp_y2 = std::exp(y * y);

  // Newton step
  T delta = (x - erf_y) * sqrt_pi_over_2 * exp_y2;
  return y + delta;
}

}  // namespace detail

// Inverse error function: erfinv(x) returns y such that erf(y) = x
// Domain: x in (-1, 1)
// Range: all real numbers
// Special cases:
//   erfinv(0) = 0
//   erfinv(1) = +inf
//   erfinv(-1) = -inf
//   erfinv(|x| > 1) = NaN
template <typename T>
T erfinv(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == static_cast<T>(0)) {
    return static_cast<T>(0);
  }

  if (x == static_cast<T>(1)) {
    return std::numeric_limits<T>::infinity();
  }

  if (x == static_cast<T>(-1)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (x < static_cast<T>(-1) || x > static_cast<T>(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Initial approximation using rational approximation
  T y;
  T ax = std::abs(x);
  if (ax < static_cast<T>(0.9959)) {
    y = detail::erfinv_central(x);
  } else {
    y = detail::erfinv_tail(x);
  }

  // Newton refinement for full precision (2 iterations)
  y = detail::erfinv_newton_refine(x, y);
  y = detail::erfinv_newton_refine(x, y);

  return y;
}

}  // namespace torchscience::kernel::special_functions
