#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T pole_tolerance() {
  // Default for low-precision types
  return T(1e-3);
}

template <>
inline float pole_tolerance<float>() { return 1e-6f; }

template <>
inline double pole_tolerance<double>() { return 1e-12; }

// Real sin_pi with range reduction for accurate computation at large arguments
template <typename T>
T sin_pi_real(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For integers, sin(n*pi) = 0 exactly
  T rounded = std::round(x);
  if (x == rounded) {
    // Return signed zero to preserve sign information
    return std::copysign(T(0), x);
  }

  // Range reduction: x_mod = x mod 2, in range [-1, 1]
  T x_mod = std::fmod(x, T(2));

  // Now compute sin(pi * x_mod) where x_mod is in [-2, 2]
  // Use symmetry to reduce to [0, 1]
  if (x_mod < T(-1)) {
    x_mod += T(2);
  } else if (x_mod > T(1)) {
    x_mod -= T(2);
  }

  // Now x_mod is in [-1, 1]
  // sin(pi*x) for x in [-1, 1]
  return std::sin(static_cast<T>(M_PI) * x_mod);
}

// Check if a real value is at a pole (non-positive integer)
template <typename T>
bool is_pole(T z) {
  if (z > T(0)) return false;
  T rounded = std::round(z);
  return std::abs(z - rounded) < pole_tolerance<T>();
}

} // namespace detail

template <typename T>
T gamma(T z) {
  constexpr double kGammaG = 7.0;

  constexpr double coefficients[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  };

  // Handle poles at non-positive integers
  if (detail::is_pole(z)) {
    return std::numeric_limits<T>::infinity();
  }

  if (z < T(0.5)) {
    // Use reflection formula: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
    T sin_piz = detail::sin_pi_real(z);

    // If sin(pi*z) is zero, we're at a pole (should be caught above, but safety check)
    if (sin_piz == T(0)) {
      return std::numeric_limits<T>::infinity();
    }

    T gamma_1mz = gamma(T(1) - z);

    // If Gamma(1-z) overflows, Gamma(z) underflows to zero
    if (std::isinf(gamma_1mz)) {
      return T(0);
    }

    return static_cast<T>(M_PI) / (sin_piz * gamma_1mz);
  }

  T z_adj = z - T(1);

  T x = static_cast<T>(coefficients[0]);

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + T(i));
  }

  T t = z_adj + static_cast<T>(kGammaG) + T(0.5);

  // Use log-space computation to avoid intermediate overflow:
  // sqrt(2π) * t^(z_adj + 0.5) * exp(-t) * x = sqrt(2π) * x * exp((z_adj + 0.5) * log(t) - t)
  T log_result = std::log(std::sqrt(static_cast<T>(2 * M_PI)) * x) + (z_adj + T(0.5)) * std::log(t) - t;
  return std::exp(log_result);
}

template <typename T>
c10::complex<T> gamma(c10::complex<T> z) {
  constexpr double kGammaG = 7.0;

  constexpr double coefficients[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  };

  const T tol = detail::pole_tolerance<T>();

  if (std::abs(z.imag()) < tol) {
    T real_part = z.real();

    if (real_part <= T(0) && std::abs(real_part - std::round(real_part)) < tol) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
  }

  if (z.real() < T(0.5)) {
    // Use reflection formula: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
    c10::complex<T> sin_piz = sin_pi(z);
    c10::complex<T> gamma_1mz = gamma(c10::complex<T>(T(1), T(0)) - z);

    // If Gamma(1-z) overflows, Gamma(z) underflows to zero
    if (std::isinf(gamma_1mz.real()) || std::isinf(gamma_1mz.imag())) {
      return c10::complex<T>(T(0), T(0));
    }

    return static_cast<T>(M_PI) / (sin_piz * gamma_1mz);
  }

  c10::complex<T> z_adj = z - c10::complex<T>(T(1), T(0));

  c10::complex<T> x(static_cast<T>(coefficients[0]), T(0));

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + c10::complex<T>(T(i), T(0)));
  }

  c10::complex<T> t = z_adj + c10::complex<T>(static_cast<T>(kGammaG) + T(0.5), T(0));

  // Use log-space computation to avoid intermediate overflow:
  // sqrt(2π) * t^(z_adj + 0.5) * exp(-t) * x = sqrt(2π) * x * exp((z_adj + 0.5) * log(t) - t)
  c10::complex<T> log_result = std::log(std::sqrt(static_cast<T>(2 * M_PI)) * x) + (z_adj + c10::complex<T>(T(0.5), T(0))) * std::log(t) - t;
  return std::exp(log_result);
}

} // namespace torchscience::kernel::special_functions
