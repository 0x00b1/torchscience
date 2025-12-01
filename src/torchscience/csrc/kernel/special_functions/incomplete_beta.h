#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "beta.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
T incomplete_beta_cf(T a, T b, T x) {
  return std::exp(a * std::log(x) + b * std::log(T(1) - x) - log_beta(a, b)) * beta_continued_fraction(a, b, x) / a;
}

template <typename T>
c10::complex<T> incomplete_beta_cf(c10::complex<T> a, c10::complex<T> b, c10::complex<T> x) {
  c10::complex<T> one(T(1), T(0));

  return std::exp(a * std::log(x) + b * std::log(one - x) - log_beta(a, b)) * beta_continued_fraction(a, b, x) / a;
}

} // namespace detail

template <typename T>
T incomplete_beta(T x, T a, T b) {
  // Check for invalid parameters (non-positive a or b)
  if (a <= T(0) || b <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x <= T(0)) {
    return T(0);
  }

  if (x >= T(1)) {
    return T(1);
  }

  if (a == T(1) && b == T(1)) {
    return x;
  }

  if (a == T(1)) {
    return T(1) - std::pow(T(1) - x, b);
  }

  if (b == T(1)) {
    return std::pow(x, a);
  }

  T threshold = (a + T(1)) / (a + b + T(2));

  if (x > threshold) {
    return T(1) - incomplete_beta(T(1) - x, b, a);
  }

  return detail::incomplete_beta_cf(a, b, x);
}

template <typename T>
c10::complex<T> incomplete_beta(
  c10::complex<T> x,
  c10::complex<T> a,
  c10::complex<T> b
) {
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  // Check for invalid parameters (non-positive real parts of a or b)
  // For complex parameters, the function is undefined when Re(a) <= 0 or Re(b) <= 0
  if (a.real() <= T(0) || b.real() <= T(0)) {
    T nan_val = std::numeric_limits<T>::quiet_NaN();
    return c10::complex<T>(nan_val, nan_val);
  }

  if (std::abs(x) < detail::beta_eps<T>()) {
    return zero;
  }

  if (std::abs(x - one) < detail::beta_eps<T>()) {
    return one;
  }

  if (std::abs(a - one) < detail::beta_eps<T>() && std::abs(b - one) < detail::beta_eps<T>()) {
    return x;
  }

  if (std::abs(a - one) < detail::beta_eps<T>()) {
    return one - std::pow(one - x, b);
  }

  if (std::abs(b - one) < detail::beta_eps<T>()) {
    return std::pow(x, a);
  }

  if (x.real() > (a.real() + T(1)) / (a.real() + b.real() + T(2)) && std::abs(x.imag()) < detail::beta_eps<T>()) {
    return one - incomplete_beta(one - x, b, a);
  }

  return detail::incomplete_beta_cf(a, b, x);
}

} // namespace torchscience::kernel::special_functions
