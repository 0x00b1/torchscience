#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T log_gamma_pole_tolerance();

template <>
constexpr float log_gamma_pole_tolerance<float>() { return 1e-6f; }

template <>
constexpr double log_gamma_pole_tolerance<double>() { return 1e-12; }

} // namespace detail

template <typename T>
T log_gamma(T z) {
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

  // Use reflection formula for z < 0.5: ln(Γ(z)) = ln(π) - ln(sin(πz)) - ln(Γ(1-z))
  if (z < T(0.5)) {
    T sin_piz = std::sin(static_cast<T>(M_PI) * z);

    // Handle poles at non-positive integers
    if (sin_piz == T(0)) {
      return std::numeric_limits<T>::infinity();
    }

    return std::log(static_cast<T>(M_PI)) - std::log(std::abs(sin_piz)) - log_gamma(T(1) - z);
  }

  T z_adj = z - T(1);

  T x = static_cast<T>(coefficients[0]);

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + T(i));
  }

  const T gradient = static_cast<T>(kGammaG);

  T t = z_adj + gradient + T(0.5);

  // ln(Γ(z)) = 0.5*ln(2π) + (z-0.5)*ln(t) - t + ln(x)
  return T(0.5) * std::log(static_cast<T>(2 * M_PI)) + (z_adj + T(0.5)) * std::log(t) - t + std::log(x);
}

template <typename T>
c10::complex<T> log_gamma(c10::complex<T> z) {
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

  const T tol = detail::log_gamma_pole_tolerance<T>();

  // Check for poles at non-positive integers
  if (std::abs(z.imag()) < tol) {
    T real_part = z.real();

    if (real_part <= T(0) && std::abs(real_part - std::round(real_part)) < tol) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
  }

  // Use reflection formula for Re(z) < 0.5: ln(Γ(z)) = ln(π) - ln(sin(πz)) - ln(Γ(1-z))
  if (z.real() < T(0.5)) {
    auto sin_piz = sin_pi(z);

    return std::log(c10::complex<T>(static_cast<T>(M_PI), T(0)))
         - std::log(sin_piz)
         - log_gamma(c10::complex<T>(T(1), T(0)) - z);
  }

  c10::complex<T> z_adj = z - c10::complex<T>(T(1), T(0));

  c10::complex<T> x(static_cast<T>(coefficients[0]), T(0));

  for (int i = 1; i < 9; ++i) {
    x += static_cast<T>(coefficients[i]) / (z_adj + c10::complex<T>(T(i), T(0)));
  }

  const T gradient = static_cast<T>(kGammaG);

  c10::complex<T> t = z_adj + c10::complex<T>(gradient + T(0.5), T(0));

  // ln(Γ(z)) = 0.5*ln(2π) + (z-0.5)*ln(t) - t + ln(x)
  return c10::complex<T>(T(0.5) * std::log(static_cast<T>(2 * M_PI)), T(0))
       + (z_adj + c10::complex<T>(T(0.5), T(0))) * std::log(t)
       - t
       + std::log(x);
}

} // namespace torchscience::kernel::special_functions
