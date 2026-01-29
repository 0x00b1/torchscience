#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "exponential_integral_e_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for exponential_integral_e
template <typename T>
constexpr T exponential_integral_e_eps();

template <>
constexpr float exponential_integral_e_eps<float>() { return 1e-6f; }

template <>
constexpr double exponential_integral_e_eps<double>() { return 1e-12; }

template <>
inline c10::Half exponential_integral_e_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 exponential_integral_e_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Compute E_n(x) using upward recurrence from E_1(x)
// E_n(x) = (e^{-x} - x * E_{n-1}(x)) / (n-1) for n >= 2
// This requires E_1(x) as the base case
template <typename T>
T exponential_integral_e_recurrence(int n_int, T x) {
  // Base case: E_1(x)
  T e_prev = exponential_integral_e_1(x);

  if (n_int == 1) {
    return e_prev;
  }

  T exp_neg_x = std::exp(-x);

  // Upward recurrence: E_n(x) = (e^{-x} - x * E_{n-1}(x)) / (n-1)
  for (int k = 2; k <= n_int; ++k) {
    T e_curr = (exp_neg_x - x * e_prev) / T(k - 1);
    e_prev = e_curr;
  }

  return e_prev;
}

// Continued fraction for E_n(x) for x > 1
// More stable for larger x values
// E_n(x) = e^{-x} * (1 / (x + n / (1 + 1 / (x + (n+1) / (1 + 2 / (x + ...))))))
template <typename T>
T exponential_integral_e_continued_fraction(int n_int, T x) {
  const T tiny = std::numeric_limits<T>::min() * T(1e10);
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);
  const int max_iterations = 100;

  // Modified Lentz's method for the continued fraction
  T b0 = x + T(n_int);
  T f = b0;
  if (std::abs(f) < tiny) f = tiny;
  T C = f;
  T D = T(0);

  for (int m = 1; m <= max_iterations; ++m) {
    T a_m = -T(m) * T(n_int - 1 + m);
    T b_m = x + T(n_int + 2 * m);

    D = b_m + a_m * D;
    if (std::abs(D) < tiny) D = tiny;
    D = T(1) / D;

    C = b_m + a_m / C;
    if (std::abs(C) < tiny) C = tiny;

    T delta = C * D;
    f *= delta;

    if (std::abs(delta - T(1)) < epsilon) {
      break;
    }
  }

  return std::exp(-x) / f;
}

// Complex version of recurrence
template <typename T>
c10::complex<T> exponential_integral_e_recurrence(int n_int, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Base case: E_1(z)
  Complex e_prev = exponential_integral_e_1(z);

  if (n_int == 1) {
    return e_prev;
  }

  Complex exp_neg_z = std::exp(-z);

  // Upward recurrence: E_n(z) = (e^{-z} - z * E_{n-1}(z)) / (n-1)
  for (int k = 2; k <= n_int; ++k) {
    Complex e_curr = (exp_neg_z - z * e_prev) / Complex(T(k - 1), T(0));
    e_prev = e_curr;
  }

  return e_prev;
}

// Complex continued fraction
template <typename T>
c10::complex<T> exponential_integral_e_continued_fraction(int n_int, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  const T tiny = std::numeric_limits<T>::min() * T(1e10);
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);
  const int max_iterations = 100;

  Complex n_c(T(n_int), T(0));
  Complex b0 = z + n_c;
  Complex f = b0;
  if (std::abs(f) < tiny) f = Complex(tiny, T(0));
  Complex C = f;
  Complex D(T(0), T(0));

  for (int m = 1; m <= max_iterations; ++m) {
    Complex a_m(-T(m) * T(n_int - 1 + m), T(0));
    Complex b_m = z + Complex(T(n_int + 2 * m), T(0));

    D = b_m + a_m * D;
    if (std::abs(D) < tiny) D = Complex(tiny, T(0));
    D = Complex(T(1), T(0)) / D;

    C = b_m + a_m / C;
    if (std::abs(C) < tiny) C = Complex(tiny, T(0));

    Complex delta = C * D;
    f = f * delta;

    if (std::abs(delta - Complex(T(1), T(0))) < epsilon) {
      break;
    }
  }

  return std::exp(-z) / f;
}

} // namespace detail

// Generalized exponential integral E_n(x)
// E_n(x) = integral from 1 to infinity of e^{-xt}/t^n dt for x > 0, n >= 0
// E_0(x) = e^{-x} / x
// E_1(x) = existing exponential_integral_e_1
// E_n(x) = (e^{-x} - x * E_{n-1}(x)) / (n-1) for n >= 2
template <typename T>
T exponential_integral_e(T n, T x) {
  const T eps = detail::exponential_integral_e_eps<T>();

  // Handle special cases
  if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Check if n is a non-negative integer
  T n_rounded = std::round(n);
  if (n < T(0) || std::abs(n - n_rounded) > eps) {
    // n must be a non-negative integer for real inputs
    return std::numeric_limits<T>::quiet_NaN();
  }

  int n_int = static_cast<int>(n_rounded);

  // For x < 0, E_n is undefined for real inputs
  if (x < T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // x = 0 special cases
  if (x == T(0)) {
    if (n_int == 0 || n_int == 1) {
      // E_0(0) = +inf, E_1(0) = +inf
      return std::numeric_limits<T>::infinity();
    } else {
      // E_n(0) = 1/(n-1) for n >= 2
      return T(1) / T(n_int - 1);
    }
  }

  // x = +inf
  if (std::isinf(x) && x > T(0)) {
    return T(0);  // E_n(+inf) = 0
  }

  // E_0(x) = e^{-x} / x
  if (n_int == 0) {
    return std::exp(-x) / x;
  }

  // E_1(x) = exponential_integral_e_1
  if (n_int == 1) {
    return exponential_integral_e_1(x);
  }

  // For n >= 2, use recurrence for small x, continued fraction for large x
  if (x <= T(1)) {
    return detail::exponential_integral_e_recurrence(n_int, x);
  } else {
    return detail::exponential_integral_e_continued_fraction(n_int, x);
  }
}

// Complex version
template <typename T>
c10::complex<T> exponential_integral_e(c10::complex<T> n, c10::complex<T> z) {
  using Complex = c10::complex<T>;
  const T eps = detail::exponential_integral_e_eps<T>();

  // Handle NaN inputs
  if (std::isnan(n.real()) || std::isnan(n.imag()) ||
      std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // n must be a real non-negative integer
  if (std::abs(n.imag()) > eps) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  T n_real = n.real();
  T n_rounded = std::round(n_real);
  if (n_real < T(0) || std::abs(n_real - n_rounded) > eps) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  int n_int = static_cast<int>(n_rounded);

  // z = 0 special cases
  if (std::abs(z) < eps) {
    if (n_int == 0 || n_int == 1) {
      return Complex(std::numeric_limits<T>::infinity(), T(0));
    } else {
      return Complex(T(1) / T(n_int - 1), T(0));
    }
  }

  // E_0(z) = e^{-z} / z
  if (n_int == 0) {
    return std::exp(-z) / z;
  }

  // E_1(z) = exponential_integral_e_1
  if (n_int == 1) {
    return exponential_integral_e_1(z);
  }

  // For n >= 2, use recurrence for small |z|, continued fraction for large |z|
  if (std::abs(z) <= T(1)) {
    return detail::exponential_integral_e_recurrence(n_int, z);
  } else {
    return detail::exponential_integral_e_continued_fraction(n_int, z);
  }
}

} // namespace torchscience::kernel::special_functions
