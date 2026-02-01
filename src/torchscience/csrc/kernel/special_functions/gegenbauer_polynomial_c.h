#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "hypergeometric_2_f_1.h"
#include "gamma.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

// Gegenbauer (ultraspherical) polynomial C_n^lambda(z)
// C_n^lambda(z) = Gamma(n+2*lambda) / (Gamma(2*lambda) * Gamma(n+1)) * 2F1(-n, n+2*lambda; lambda+1/2; (1-z)/2)
//
// Special cases:
// - C_0^lambda(z) = 1
// - C_1^lambda(z) = 2*lambda*z
// - C_n^(1/2)(z) = P_n(z) (Legendre polynomial)
// - C_n^1(z) = U_n(z) (Chebyshev polynomial of the second kind)
//
// The recurrence relation is:
// n * C_n^lambda(z) = 2*(n + lambda - 1)*z*C_{n-1}^lambda(z) - (n + 2*lambda - 2)*C_{n-2}^lambda(z)
//
// Applications:
// - Orthogonal polynomials with weight function (1-z^2)^(lambda-1/2) on [-1, 1]
// - Angular momentum in quantum mechanics
// - Expansions in spherical coordinates

template <typename T>
T gegenbauer_polynomial_c(T n, T lambda, T z) {
  // C_0^lambda(z) = 1
  if (std::abs(n) < T(1e-10)) {
    return T(1);
  }

  // Handle lambda = 0 case (Chebyshev limit)
  // C_n^0(z) = (2/n) * T_n(z) for n > 0
  if (std::abs(lambda) < T(1e-10)) {
    // For lambda -> 0, the coefficient becomes singular
    // Use the limiting form: lim_{lambda->0} C_n^lambda(z) = (2/n) * cos(n * acos(z))
    // But this only works for |z| <= 1
    // For general z, use numerical limiting approach
    T eps = T(1e-6);
    return gegenbauer_polynomial_c(n, eps, z);
  }

  // For non-negative integer n, use recurrence for better numerical stability
  // when n is small
  T n_val = n;
  if (n_val > T(0) && std::abs(n_val - std::round(static_cast<double>(n_val))) < T(1e-10)) {
    int n_int = static_cast<int>(std::round(static_cast<double>(n_val)));

    if (n_int <= 20) {
      // Use recurrence relation for small integer n
      // C_0^lambda = 1
      // C_1^lambda = 2*lambda*z
      // n * C_n^lambda = 2*(n + lambda - 1)*z*C_{n-1}^lambda - (n + 2*lambda - 2)*C_{n-2}^lambda

      if (n_int == 0) {
        return T(1);
      }
      if (n_int == 1) {
        return T(2) * lambda * z;
      }

      T C_prev = T(1);                  // C_0
      T C_curr = T(2) * lambda * z;     // C_1

      for (int k = 2; k <= n_int; ++k) {
        T k_val = static_cast<T>(k);
        T C_next = (T(2) * (k_val + lambda - T(1)) * z * C_curr - (k_val + T(2) * lambda - T(2)) * C_prev) / k_val;
        C_prev = C_curr;
        C_curr = C_next;
      }

      return C_curr;
    }
  }

  // For larger n or non-integer n, use the hypergeometric representation
  // C_n^lambda(z) = Gamma(n+2*lambda) / (Gamma(2*lambda) * Gamma(n+1)) * 2F1(-n, n+2*lambda; lambda+1/2; (1-z)/2)

  // Compute the coefficient: Gamma(n+2*lambda) / (Gamma(2*lambda) * Gamma(n+1))
  T log_coeff = log_gamma(n + T(2) * lambda) - log_gamma(T(2) * lambda) - log_gamma(n + T(1));
  T coeff = std::exp(log_coeff);

  // Handle case where coefficient might be NaN due to gamma poles
  if (!std::isfinite(static_cast<double>(coeff))) {
    // Try computing directly
    coeff = gamma(n + T(2) * lambda) / (gamma(T(2) * lambda) * gamma(n + T(1)));
  }

  // Compute 2F1(-n, n+2*lambda; lambda+1/2; (1-z)/2)
  T a = -n;
  T b = n + T(2) * lambda;
  T c = lambda + T(0.5);
  T w = (T(1) - z) / T(2);
  T F = hypergeometric_2_f_1(a, b, c, w);

  return coeff * F;
}

// Complex version
template <typename T>
c10::complex<T> gegenbauer_polynomial_c(c10::complex<T> n, c10::complex<T> lambda, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> half(T(0.5), T(0));

  // C_0^lambda(z) = 1
  if (std::abs(n) < T(1e-10)) {
    return one;
  }

  // Handle lambda = 0 case
  if (std::abs(lambda) < T(1e-10)) {
    c10::complex<T> eps(T(1e-6), T(0));
    return gegenbauer_polynomial_c(n, eps, z);
  }

  // For non-negative integer n, use recurrence for better numerical stability
  if (std::abs(n.imag()) < T(1e-10) && n.real() > T(0)) {
    T n_real = n.real();
    if (std::abs(n_real - std::round(static_cast<double>(n_real))) < T(1e-10)) {
      int n_int = static_cast<int>(std::round(static_cast<double>(n_real)));

      if (n_int <= 20) {
        if (n_int == 0) {
          return one;
        }
        if (n_int == 1) {
          return two * lambda * z;
        }

        c10::complex<T> C_prev = one;               // C_0
        c10::complex<T> C_curr = two * lambda * z;  // C_1

        for (int k = 2; k <= n_int; ++k) {
          c10::complex<T> k_val(static_cast<T>(k), T(0));
          c10::complex<T> C_next = (two * (k_val + lambda - one) * z * C_curr - (k_val + two * lambda - two) * C_prev) / k_val;
          C_prev = C_curr;
          C_curr = C_next;
        }

        return C_curr;
      }
    }
  }

  // Use the hypergeometric representation
  c10::complex<T> log_coeff = log_gamma(n + two * lambda) - log_gamma(two * lambda) - log_gamma(n + one);
  c10::complex<T> coeff = std::exp(log_coeff);

  c10::complex<T> a = -n;
  c10::complex<T> b = n + two * lambda;
  c10::complex<T> c = lambda + half;
  c10::complex<T> w = (one - z) / two;
  c10::complex<T> F = hypergeometric_2_f_1(a, b, c, w);

  return coeff * F;
}

} // namespace torchscience::kernel::special_functions
