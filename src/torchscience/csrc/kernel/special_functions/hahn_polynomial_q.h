#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Hahn polynomial Q_n(x; alpha, beta, N)
//
// The Hahn polynomial is a discrete orthogonal polynomial defined on the set
// {0, 1, 2, ..., N} that generalizes the Krawtchouk, Meixner, and Charlier
// polynomials.
//
// Mathematical Definition
// -----------------------
// Using the hypergeometric representation:
//
//     Q_n(x; alpha, beta, N) = 3F2(-n, n+alpha+beta+1, -x; alpha+1, -N; 1)
//
// This can be computed using the explicit sum:
//
//     Q_n(x; alpha, beta, N) = sum_{k=0}^{n} [(-n)_k * (n+alpha+beta+1)_k * (-x)_k]
//                              / [(alpha+1)_k * (-N)_k * k!]
//
// where (a)_k is the Pochhammer symbol (rising factorial).
//
// Parameters:
// - n: degree of the polynomial (typically 0 <= n <= N for classical interpretation)
// - x: argument (typically 0 <= x <= N)
// - alpha: parameter (alpha > -1)
// - beta: parameter (beta > -1)
// - N: size parameter (positive integer for classical interpretation)
//
// Special Values
// --------------
// - Q_0(x; alpha, beta, N) = 1 for all valid parameters
// - Q_1(x; alpha, beta, N) = 1 - (alpha+beta+2)*x / ((alpha+1)*N)
//
// Applications
// ------------
// - Quantum mechanics
// - Signal processing
// - Coding theory
// - Approximation theory

namespace detail {

// Pochhammer symbol (rising factorial) (a)_k = a(a+1)(a+2)...(a+k-1)
template <typename T>
T hahn_pochhammer(T a, int k) {
  if (k <= 0) return T(1);
  T result = T(1);
  for (int i = 0; i < k; ++i) {
    result *= (a + T(i));
  }
  return result;
}

// Complex version of Pochhammer
template <typename T>
c10::complex<T> hahn_pochhammer(c10::complex<T> a, int k) {
  if (k <= 0) return c10::complex<T>(T(1), T(0));
  c10::complex<T> result(T(1), T(0));
  for (int i = 0; i < k; ++i) {
    result *= (a + c10::complex<T>(T(i), T(0)));
  }
  return result;
}

// Factorial helper
template <typename T>
T hahn_factorial(int k) {
  if (k <= 0) return T(1);
  T result = T(1);
  for (int i = 2; i <= k; ++i) {
    result *= T(i);
  }
  return result;
}

} // namespace detail

template <typename T>
T hahn_polynomial_q(T n, T x, T alpha, T beta, T N) {
  int n_int = static_cast<int>(std::floor(n + T(0.5)));

  // Q_0(x; alpha, beta, N) = 1
  if (n_int <= 0) return T(1);

  // For polynomial case, sum from k=0 to n
  // Q_n(x; alpha, beta, N) = sum_{k=0}^{n} [(-n)_k * (n+alpha+beta+1)_k * (-x)_k]
  //                          / [(alpha+1)_k * (-N)_k * k!]
  T sum = T(0);

  for (int k = 0; k <= n_int; ++k) {
    // Compute Pochhammer symbols
    T neg_n_k = detail::hahn_pochhammer(-n, k);           // (-n)_k
    T n_ab1_k = detail::hahn_pochhammer(n + alpha + beta + T(1), k);  // (n+alpha+beta+1)_k
    T neg_x_k = detail::hahn_pochhammer(-x, k);           // (-x)_k
    T alpha1_k = detail::hahn_pochhammer(alpha + T(1), k); // (alpha+1)_k
    T neg_N_k = detail::hahn_pochhammer(-N, k);           // (-N)_k
    T k_factorial = detail::hahn_factorial<T>(k);          // k!

    // Compute term
    T numerator = neg_n_k * n_ab1_k * neg_x_k;
    T denominator = alpha1_k * neg_N_k * k_factorial;

    // Check for division by zero
    if (std::abs(denominator) < std::numeric_limits<T>::epsilon() * T(100)) {
      // If denominator is zero but numerator is also zero, term is 0
      if (std::abs(numerator) < std::numeric_limits<T>::epsilon() * T(100)) {
        continue;
      }
      // Otherwise we have a pole - return infinity
      return std::numeric_limits<T>::infinity();
    }

    sum += numerator / denominator;
  }

  return sum;
}

// Complex version
template <typename T>
c10::complex<T> hahn_polynomial_q(
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    c10::complex<T> N) {

  int n_int = static_cast<int>(std::floor(n.real() + T(0.5)));

  // Q_0(x; alpha, beta, N) = 1
  if (n_int <= 0) return c10::complex<T>(T(1), T(0));

  c10::complex<T> sum(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  for (int k = 0; k <= n_int; ++k) {
    // Compute Pochhammer symbols
    c10::complex<T> neg_n_k = detail::hahn_pochhammer(-n, k);
    c10::complex<T> n_ab1_k = detail::hahn_pochhammer(n + alpha + beta + one, k);
    c10::complex<T> neg_x_k = detail::hahn_pochhammer(-x, k);
    c10::complex<T> alpha1_k = detail::hahn_pochhammer(alpha + one, k);
    c10::complex<T> neg_N_k = detail::hahn_pochhammer(-N, k);
    T k_factorial = detail::hahn_factorial<T>(k);

    // Compute term
    c10::complex<T> numerator = neg_n_k * n_ab1_k * neg_x_k;
    c10::complex<T> denominator = alpha1_k * neg_N_k * c10::complex<T>(k_factorial, T(0));

    // Check for division by zero
    if (std::abs(denominator) < std::numeric_limits<T>::epsilon() * T(100)) {
      if (std::abs(numerator) < std::numeric_limits<T>::epsilon() * T(100)) {
        continue;
      }
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    sum += numerator / denominator;
  }

  return sum;
}

} // namespace torchscience::kernel::special_functions
