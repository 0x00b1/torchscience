#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Charlier polynomial C_n(x; a)
//
// The Charlier polynomials are discrete orthogonal polynomials associated with
// the Poisson distribution. They satisfy the orthogonality relation:
//
//   sum_{x=0}^{infinity} a^x/x! * C_m(x;a) * C_n(x;a) = e^a * n!/a^n * delta_{mn}
//
// The polynomials can be defined via the recurrence relation:
//   C_0(x; a) = 1
//   C_1(x; a) = (x - a) / a = x/a - 1
//   a * C_{n+1}(x; a) = (x - n - a) * C_n(x; a) - n * C_{n-1}(x; a)
//
// Or equivalently via the hypergeometric representation (formal):
//   C_n(x; a) = (-1)^n * 2F0(-n, -x; ; -1/a)
//
// Special values:
//   C_0(x; a) = 1
//   C_1(x; a) = x/a - 1
//   C_2(x; a) = x^2/a^2 - (2a + 1)x/a^2 + 1 = (x^2 - (2a+1)x + a^2) / a^2
//
// Parameters:
//   n: degree (n >= 0)
//   x: argument
//   a: parameter (a > 0)
//
// Applications:
//   - Poisson distribution: orthogonal polynomials for the Poisson weight
//   - Combinatorics: counting problems related to permutations
//   - Quantum optics: coherent states

template <typename T>
T charlier_polynomial_c(T n, T x, T a) {
  // C_0(x; a) = 1
  if (std::abs(n) < T(1e-10)) {
    return T(1);
  }

  // C_1(x; a) = x/a - 1
  if (std::abs(n - T(1)) < T(1e-10)) {
    return x / a - T(1);
  }

  // For non-negative integer n, use the recurrence relation
  T n_val = n;
  if (n_val > T(0) && std::abs(n_val - std::round(static_cast<double>(n_val))) < T(1e-10)) {
    int n_int = static_cast<int>(std::round(static_cast<double>(n_val)));

    if (n_int == 0) {
      return T(1);
    }
    if (n_int == 1) {
      return x / a - T(1);
    }

    // Use recurrence: a * C_{n+1} = (x - n - a) * C_n - n * C_{n-1}
    // Or: C_{n+1} = ((x - n - a) * C_n - n * C_{n-1}) / a
    T C_prev = T(1);           // C_0
    T C_curr = x / a - T(1);   // C_1

    for (int k = 1; k < n_int; ++k) {
      T k_val = static_cast<T>(k);
      // C_{k+1} = ((x - k - a) * C_k - k * C_{k-1}) / a
      T C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
      C_prev = C_curr;
      C_curr = C_next;
    }

    return C_curr;
  }

  // For non-integer n, use finite differences to approximate
  // This is a fallback; integer degrees are the primary use case
  // We interpolate between floor(n) and ceil(n)
  int n_floor = static_cast<int>(std::floor(static_cast<double>(n_val)));
  int n_ceil = n_floor + 1;

  if (n_floor < 0) {
    // For negative n, return 0 (undefined in standard definition)
    return T(0);
  }

  // Compute C_{n_floor} and C_{n_ceil}
  T C_floor, C_ceil;

  if (n_floor == 0) {
    C_floor = T(1);
  } else if (n_floor == 1) {
    C_floor = x / a - T(1);
  } else {
    T C_prev = T(1);
    T C_curr = x / a - T(1);
    for (int k = 1; k < n_floor; ++k) {
      T k_val = static_cast<T>(k);
      T C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
      C_prev = C_curr;
      C_curr = C_next;
    }
    C_floor = C_curr;
  }

  // C_ceil = C_{n_floor + 1}
  if (n_ceil == 1) {
    C_ceil = x / a - T(1);
  } else {
    T C_prev = T(1);
    T C_curr = x / a - T(1);
    for (int k = 1; k < n_ceil; ++k) {
      T k_val = static_cast<T>(k);
      T C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
      C_prev = C_curr;
      C_curr = C_next;
    }
    C_ceil = C_curr;
  }

  // Linear interpolation
  T frac = n_val - static_cast<T>(n_floor);
  return (T(1) - frac) * C_floor + frac * C_ceil;
}

// Complex version
template <typename T>
c10::complex<T> charlier_polynomial_c(c10::complex<T> n, c10::complex<T> x, c10::complex<T> a) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> zero(T(0), T(0));

  // C_0(x; a) = 1
  if (std::abs(n) < T(1e-10)) {
    return one;
  }

  // C_1(x; a) = x/a - 1
  if (std::abs(n - one) < T(1e-10)) {
    return x / a - one;
  }

  // For real integer n, use the recurrence relation
  if (std::abs(n.imag()) < T(1e-10) && n.real() > T(0)) {
    T n_real = n.real();
    if (std::abs(n_real - std::round(static_cast<double>(n_real))) < T(1e-10)) {
      int n_int = static_cast<int>(std::round(static_cast<double>(n_real)));

      if (n_int == 0) {
        return one;
      }
      if (n_int == 1) {
        return x / a - one;
      }

      c10::complex<T> C_prev = one;
      c10::complex<T> C_curr = x / a - one;

      for (int k = 1; k < n_int; ++k) {
        c10::complex<T> k_val(static_cast<T>(k), T(0));
        c10::complex<T> C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
        C_prev = C_curr;
        C_curr = C_next;
      }

      return C_curr;
    }
  }

  // Fallback: for non-integer n, use linear interpolation
  T n_real = n.real();
  int n_floor = static_cast<int>(std::floor(static_cast<double>(n_real)));
  int n_ceil = n_floor + 1;

  if (n_floor < 0) {
    return zero;
  }

  c10::complex<T> C_floor, C_ceil;

  if (n_floor == 0) {
    C_floor = one;
  } else if (n_floor == 1) {
    C_floor = x / a - one;
  } else {
    c10::complex<T> C_prev = one;
    c10::complex<T> C_curr = x / a - one;
    for (int k = 1; k < n_floor; ++k) {
      c10::complex<T> k_val(static_cast<T>(k), T(0));
      c10::complex<T> C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
      C_prev = C_curr;
      C_curr = C_next;
    }
    C_floor = C_curr;
  }

  if (n_ceil == 1) {
    C_ceil = x / a - one;
  } else {
    c10::complex<T> C_prev = one;
    c10::complex<T> C_curr = x / a - one;
    for (int k = 1; k < n_ceil; ++k) {
      c10::complex<T> k_val(static_cast<T>(k), T(0));
      c10::complex<T> C_next = ((x - k_val - a) * C_curr - k_val * C_prev) / a;
      C_prev = C_curr;
      C_curr = C_next;
    }
    C_ceil = C_curr;
  }

  c10::complex<T> frac = n - c10::complex<T>(static_cast<T>(n_floor), T(0));
  return (one - frac) * C_floor + frac * C_ceil;
}

} // namespace torchscience::kernel::special_functions
