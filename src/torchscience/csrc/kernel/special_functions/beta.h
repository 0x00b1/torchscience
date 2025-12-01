#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>

#include "log_beta.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T beta(T a, T b) {
  return std::exp(log_beta(a, b));
}

template <typename T>
c10::complex<T> beta(c10::complex<T> a, c10::complex<T> b) {
  return std::exp(log_beta(a, b));
}

namespace detail {

template <typename T>
constexpr T beta_eps();

template <>
constexpr float beta_eps<float>() { return 1e-7f; }

template <>
constexpr double beta_eps<double>() { return 1e-15; }

template <>
inline c10::Half beta_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 beta_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int beta_max_iter() { return 200; }

template <typename T>
T beta_continued_fraction(T a, T b, T x) {
  const T eps = beta_eps<T>();

  T qab = a + b;
  T qap = a + T(1);
  T qam = a - T(1);

  T c = T(1);
  T d = T(1) - qab * x / qap;

  if (std::abs(d) < eps) {
    d = eps;
  }

  d = T(1) / d;
  T h = d;

  for (int m = 1; m <= beta_max_iter<T>(); ++m) {
    T m_t = static_cast<T>(m);
    T m2 = T(2) * m_t;

    T aa = m_t * (b - m_t) * x / ((qam + m2) * (a + m2));
    d = T(1) + aa * d;

    if (std::abs(d) < eps) {
      d = eps;
    }

    c = T(1) + aa / c;

    if (std::abs(c) < eps) {
      c = eps;
    }

    d = T(1) / d;
    h *= d * c;

    aa = -(a + m_t) * (qab + m_t) * x / ((a + m2) * (qap + m2));
    d = T(1) + aa * d;

    if (std::abs(d) < eps) {
      d = eps;
    }

    c = T(1) + aa / c;

    if (std::abs(c) < eps) {
      c = eps;
    }

    d = T(1) / d;
    T delta = d * c;
    h *= delta;

    if (std::abs(delta - T(1)) < eps) {
      break;
    }
  }

  return h;
}

template <typename T>
c10::complex<T> beta_continued_fraction(c10::complex<T> a, c10::complex<T> b, c10::complex<T> x) {
  const T eps = beta_eps<T>();
  const int max_iter = beta_max_iter<T>();

  c10::complex<T> qab = a + b;
  c10::complex<T> qap = a + c10::complex<T>(T(1), T(0));
  c10::complex<T> qam = a - c10::complex<T>(T(1), T(0));

  c10::complex<T> one(T(1), T(0));
  c10::complex<T> c = one;
  c10::complex<T> d = one - qab * x / qap;
  if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
  d = one / d;
  c10::complex<T> h = d;

  for (int m = 1; m <= max_iter; ++m) {
    c10::complex<T> m_c(static_cast<T>(m), T(0));
    c10::complex<T> m2 = c10::complex<T>(T(2), T(0)) * m_c;

    c10::complex<T> aa = m_c * (b - m_c) * x / ((qam + m2) * (a + m2));
    d = one + aa * d;
    if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
    c = one + aa / c;
    if (std::abs(c) < eps) c = c10::complex<T>(eps, T(0));
    d = one / d;
    h *= d * c;

    aa = -(a + m_c) * (qab + m_c) * x / ((a + m2) * (qap + m2));
    d = one + aa * d;
    if (std::abs(d) < eps) d = c10::complex<T>(eps, T(0));
    c = one + aa / c;
    if (std::abs(c) < eps) c = c10::complex<T>(eps, T(0));

    d = one / d;

    c10::complex<T> delta = d * c;

    h *= delta;

    if (std::abs(delta - one) < eps) break;
  }

  return h;
}

} // namespace detail

} // namespace torchscience::kernel::special_functions
