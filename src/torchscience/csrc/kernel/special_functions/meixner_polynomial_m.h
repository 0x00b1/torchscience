#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

// Meixner polynomial M_n(x; beta, c)
//
// The Meixner polynomial is a discrete orthogonal polynomial defined on
// the non-negative integers with weight function given by the negative
// binomial distribution.
//
// Mathematical Definition
// -----------------------
// Using the hypergeometric representation:
//
//     M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)
//
// Parameters:
// - n: degree of the polynomial (n >= 0)
// - x: argument
// - beta: parameter (beta > 0)
// - c: parameter (0 < c < 1)
//
// Special Values
// --------------
// - M_0(x; beta, c) = 1 for all valid parameters
// - M_1(x; beta, c) = 1 + x*(c-1)/(c*beta)
//
// Recurrence Relation
// -------------------
// c*(n + beta)*M_{n+1}(x) = [(c-1)*x + (1+c)*n + c*beta] * M_n(x)
//                          - n * M_{n-1}(x)
//
// Applications
// ------------
// - Probability theory (negative binomial distribution)
// - Quantum physics
// - Coding theory
// - Combinatorics
// - Birth-death processes

template <typename T>
T meixner_polynomial_m(T n, T x, T beta, T c) {
  // M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)
  T a = -n;
  T b = -x;
  T z = T(1) - T(1) / c;

  return hypergeometric_2_f_1(a, b, beta, z);
}

// Complex version
template <typename T>
c10::complex<T> meixner_polynomial_m(
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> beta,
    c10::complex<T> c) {
  c10::complex<T> one(T(1), T(0));

  c10::complex<T> a = -n;
  c10::complex<T> b = -x;
  c10::complex<T> z = one - one / c;

  return hypergeometric_2_f_1(a, b, beta, z);
}

} // namespace torchscience::kernel::special_functions
