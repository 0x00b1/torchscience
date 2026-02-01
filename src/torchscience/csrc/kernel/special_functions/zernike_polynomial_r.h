#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_polynomial_p.h"

namespace torchscience::kernel::special_functions {

// Radial Zernike polynomial R_n^m(rho)
//
// The radial Zernike polynomials are defined for non-negative integers n and m
// where |m| <= n and (n - |m|) is even.
//
// Mathematical Definition:
// R_n^m(rho) = (-1)^((n-m)/2) * rho^m * P_{(n-m)/2}^{(m, 0)}(1 - 2*rho^2)
//
// where P_k^(alpha,beta)(x) is the Jacobi polynomial.
//
// Alternatively, using the explicit sum formula:
// R_n^m(rho) = sum_{k=0}^{(n-|m|)/2} (-1)^k * C(n-k, k) * C(n-2k, (n-|m|)/2 - k) * rho^(n-2k)
//
// Special cases:
// - R_n^n(rho) = rho^n
// - R_n^m(0) = 0 for m > 0, and R_n^0(0) = (-1)^(n/2) for even n
// - R_n^m(1) = 1 for all valid (n, m)
//
// Constraints:
// - n >= 0
// - |m| <= n
// - (n - |m|) must be even
// - 0 <= rho <= 1 (polynomials are defined on unit disk)
//
// For invalid (n, m) combinations (odd n-m), return 0.
//
// Applications:
// - Optical aberration analysis (wavefront decomposition)
// - Corneal topography
// - Image analysis and pattern recognition
// - Moment invariants in computer vision

template <typename T>
T zernike_polynomial_r(T n, T m, T rho) {
  // Take absolute value of m (R_n^m = R_n^{-m})
  T abs_m = std::abs(m);
  T diff = n - abs_m;

  // Check if n < |m| (invalid)
  if (diff < T(0)) {
    return T(0);
  }

  // Check if (n - |m|) is even
  // Use fmod to check if diff is an even integer
  T diff_mod2 = std::fmod(diff, T(2));
  if (std::abs(diff_mod2) > T(0.5)) {
    return T(0);  // (n - |m|) is odd, return 0
  }

  // k = (n - |m|) / 2
  T k = diff / T(2);

  // Handle special case: R_n^n(rho) = rho^n
  if (std::abs(k) < T(1e-10)) {
    return std::pow(rho, abs_m);
  }

  // Compute the sign factor: (-1)^k
  // For integer k, this is 1 if k is even, -1 if k is odd
  T sign;
  T k_mod2 = std::fmod(k, T(2));
  if (std::abs(k_mod2) < T(0.5)) {
    sign = T(1);
  } else {
    sign = T(-1);
  }

  // Compute x = 1 - 2*rho^2 (argument for Jacobi polynomial)
  T x = T(1) - T(2) * rho * rho;

  // Compute P_k^(|m|, 0)(x) using Jacobi polynomial
  T jacobi = jacobi_polynomial_p(k, abs_m, T(0), x);

  // R_n^m(rho) = (-1)^k * rho^|m| * P_k^(|m|, 0)(1 - 2*rho^2)
  T rho_power = std::pow(rho, abs_m);

  return sign * rho_power * jacobi;
}

// Complex version
template <typename T>
c10::complex<T> zernike_polynomial_r(c10::complex<T> n, c10::complex<T> m, c10::complex<T> rho) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  // Take absolute value of m (real part for the check)
  T m_real = m.real();
  T abs_m_real = std::abs(m_real);
  c10::complex<T> abs_m(abs_m_real, T(0));

  c10::complex<T> diff = n - abs_m;

  // Check if n < |m| (invalid)
  if (diff.real() < T(0)) {
    return zero;
  }

  // Check if (n - |m|) is even
  T diff_mod2 = std::fmod(diff.real(), T(2));
  if (std::abs(diff_mod2) > T(0.5)) {
    return zero;  // (n - |m|) is odd
  }

  // k = (n - |m|) / 2
  c10::complex<T> k = diff / two;

  // Handle special case: R_n^n(rho) = rho^n
  if (std::abs(k) < T(1e-10)) {
    return std::pow(rho, abs_m);
  }

  // Compute the sign factor: (-1)^k
  c10::complex<T> sign;
  T k_mod2 = std::fmod(k.real(), T(2));
  if (std::abs(k_mod2) < T(0.5)) {
    sign = one;
  } else {
    sign = -one;
  }

  // Compute x = 1 - 2*rho^2
  c10::complex<T> x = one - two * rho * rho;

  // Compute P_k^(|m|, 0)(x)
  c10::complex<T> jacobi = jacobi_polynomial_p(k, abs_m, zero, x);

  // R_n^m(rho) = (-1)^k * rho^|m| * P_k^(|m|, 0)(1 - 2*rho^2)
  c10::complex<T> rho_power = std::pow(rho, abs_m);

  return sign * rho_power * jacobi;
}

} // namespace torchscience::kernel::special_functions
