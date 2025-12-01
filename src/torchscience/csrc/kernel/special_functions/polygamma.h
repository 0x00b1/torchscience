#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "digamma.h"
#include "trigamma.h"
#include "tetragamma.h"
#include "pentagamma.h"
#include "sin_pi.h"
#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T polygamma_general(int n, T z) {
  // For n >= 4, use recurrence relation and asymptotic expansion
  // psi^(n)(z) = (-1)^(n+1) * n! * sum_{k=0}^{inf} 1/(z+k)^(n+1)

  T result = T(0);
  T y = z;

  // Use recurrence to shift z to larger values for better convergence
  // psi^(n)(z) = psi^(n)(z+1) + (-1)^(n+1) * n! / z^(n+1)
  T factorial_n = T(1);
  for (int i = 2; i <= n; ++i) {
    factorial_n *= T(i);
  }

  T sign = (n % 2 == 0) ? T(-1) : T(1);

  while (y < T(6)) {
    T y_pow = std::pow(y, n + 1);
    result += sign * factorial_n / y_pow;
    y += T(1);
  }

  // Asymptotic expansion for psi^(n)(y) with large y
  // psi^(n)(y) ~ (-1)^(n+1) * [(n-1)!/y^n + n!/(2*y^(n+1)) + ...]
  T y_pow_n = std::pow(y, n);
  T y_pow_np1 = y_pow_n * y;
  T y_pow_np2 = y_pow_np1 * y;

  T factorial_nm1 = factorial_n / T(n);

  // Leading terms of asymptotic expansion
  result += sign * (
    factorial_nm1 / y_pow_n +
    factorial_n / (T(2) * y_pow_np1) +
    factorial_n * T(n + 1) / (T(12) * y_pow_np2)
  );

  return result;
}

template <typename T>
T polygamma(T n, T z) {
  int order = static_cast<int>(n);

  if (order < 0) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  switch (order) {
    case 0:
      return digamma(z);
    case 1:
      return trigamma(z);
    case 2:
      return tetragamma(z);
    case 3:
      return pentagamma(z);
    default:
      return polygamma_general(order, z);
  }
}

template <typename T>
c10::complex<T> polygamma_general(int n, c10::complex<T> z) {
  // Complex polygamma for n >= 4
  c10::complex<T> result(T(0), T(0));
  c10::complex<T> y = z;

  T factorial_n = T(1);
  for (int i = 2; i <= n; ++i) {
    factorial_n *= T(i);
  }

  T sign = (n % 2 == 0) ? T(-1) : T(1);

  // Reflection formula for Re(z) < 0.5
  if (y.real() < T(0.5)) {
    // Use reflection formula
    // psi^(n)(z) = (-1)^(n+1) * d^n/dz^n [pi*cot(pi*z)] + (-1)^n * psi^(n)(1-z)
    // This is complex, so we use the direct series for now
    // Fall through to direct computation with shifted argument
  }

  while (std::abs(y) < T(6)) {
    c10::complex<T> y_pow = std::pow(y, n + 1);
    result = result + c10::complex<T>(sign * factorial_n, T(0)) / y_pow;
    y = y + c10::complex<T>(T(1), T(0));
  }

  // Asymptotic expansion
  c10::complex<T> y_pow_n = std::pow(y, n);
  c10::complex<T> y_pow_np1 = y_pow_n * y;
  c10::complex<T> y_pow_np2 = y_pow_np1 * y;

  T factorial_nm1 = factorial_n / T(n);

  result = result + c10::complex<T>(sign, T(0)) * (
    c10::complex<T>(factorial_nm1, T(0)) / y_pow_n +
    c10::complex<T>(factorial_n / T(2), T(0)) / y_pow_np1 +
    c10::complex<T>(factorial_n * T(n + 1) / T(12), T(0)) / y_pow_np2
  );

  return result;
}

template <typename T>
c10::complex<T> polygamma(c10::complex<T> n, c10::complex<T> z) {
  // For complex n, only integer values are valid
  int order = static_cast<int>(n.real());

  if (n.imag() != T(0) || n.real() != T(order) || order < 0) {
    return c10::complex<T>(
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    );
  }

  switch (order) {
    case 0:
      return digamma(z);
    case 1:
      return trigamma(z);
    case 2:
      return tetragamma(z);
    case 3:
      return pentagamma(z);
    default:
      return polygamma_general(order, z);
  }
}

} // namespace torchscience::kernel::special_functions
