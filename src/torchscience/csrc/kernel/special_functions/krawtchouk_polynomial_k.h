#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

// Krawtchouk polynomial K_n(x; p, N)
//
// The Krawtchouk polynomial is a discrete orthogonal polynomial defined on
// the set {0, 1, 2, ..., N} with weight function given by the binomial distribution.
//
// Mathematical Definition
// -----------------------
// Using the hypergeometric representation:
//
//     K_n(x; p, N) = 2F1(-n, -x; -N; 1/p)
//
// Parameters:
// - n: degree of the polynomial (typically 0 <= n <= N for classical interpretation)
// - x: argument (typically 0 <= x <= N for combinatorial interpretations)
// - p: probability parameter (0 < p < 1)
// - N: size parameter (positive integer for classical interpretation)
//
// Special Values
// --------------
// - K_0(x; p, N) = 1 for all valid parameters
// - K_1(x; p, N) = 1 - x/(N*p)
// - K_n(0; p, N) = C(N, n) * ((1-p)/p)^n for integer n
//
// Recurrence Relation
// -------------------
// (n+1) * K_{n+1}(x) = (N*p - x - (2p-1)*n) * K_n(x) - (N-n+1) * p * (1-p) * K_{n-1}(x)
//
// Applications
// ------------
// - Quantum optics and quantum information
// - Coding theory (MacWilliams transform)
// - Probability theory (binomial distribution)
// - Combinatorics (enumeration problems)
// - Signal processing

template <typename T>
T krawtchouk_polynomial_k(T n, T x, T p, T N) {
  // K_n(x; p, N) = 2F1(-n, -x; -N; 1/p)
  T a = -n;
  T b = -x;
  T c = -N;
  T z = T(1) / p;

  return hypergeometric_2_f_1(a, b, c, z);
}

// Complex version
template <typename T>
c10::complex<T> krawtchouk_polynomial_k(
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> p,
    c10::complex<T> N) {
  c10::complex<T> one(T(1), T(0));

  c10::complex<T> a = -n;
  c10::complex<T> b = -x;
  c10::complex<T> c = -N;
  c10::complex<T> z = one / p;

  return hypergeometric_2_f_1(a, b, c, z);
}

} // namespace torchscience::kernel::special_functions
