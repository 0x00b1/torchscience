#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Exponential integral Ein(x)
// Ein(x) = integral from 0 to x of (1 - e^(-t))/t dt
// Ein(x) is an entire function (no singularities)
// Series: Ein(x) = sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
// Relation: Ein(x) = Ei(x) - gamma - ln|x| for x > 0 (real positive x)

// The series Ein(x) = x - x^2/4 + x^3/18 - x^4/96 + ...
// converges for all x (entire function)

template <typename T>
T exponential_integral_ein(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == T(0)) {
    return T(0);
  }

  if (std::isinf(x)) {
    // Ein(+inf) = +inf, Ein(-inf) = -inf
    return x;
  }

  // Series: Ein(x) = sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
  // term_n = (-1)^{n+1} x^n / (n * n!)
  // term_1 = x / (1 * 1!) = x
  // term_n / term_{n-1} = -x / n^2 * (n-1) = -x * (n-1) / n^2

  T result = x;  // First term: n=1, x^1 / (1 * 1!) = x
  T term = x;

  const int max_iterations = 200;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 2; n <= max_iterations; ++n) {
    // term_n = term_{n-1} * (-x) * (n-1) / (n * n)
    term *= -x * T(n - 1) / (T(n) * T(n));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

// Complex exponential integral Ein(z)
template <typename T>
c10::complex<T> exponential_integral_ein(c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Handle special cases
  if (std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // z = 0: Ein(0) = 0
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(T(0), T(0));
  }

  // Series expansion: Ein(z) = sum_{n=1}^inf (-1)^{n+1} z^n / (n * n!)
  // This converges for all z since Ein is entire

  Complex result = z;  // First term: z
  Complex term = z;

  const int max_iterations = 300;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 2; n <= max_iterations; ++n) {
    // term_n = term_{n-1} * (-z) * (n-1) / (n * n)
    term *= -z * T(n - 1) / (T(n) * T(n));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

} // namespace torchscience::kernel::special_functions
