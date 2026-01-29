#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Euler-Mascheroni constant (duplicated for self-contained header)
template <typename T>
constexpr T euler_mascheroni_e1() {
  return T(0.5772156649015328606065120900824024310421593359);
}

// Series expansion for E1(x) for small positive x (0 < x <= 1)
// E1(x) = -gamma - ln(x) + sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
template <typename T>
T exponential_integral_e_1_series(T x) {
  const T gamma = euler_mascheroni_e1<T>();
  T result = -gamma - std::log(x);

  // sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
  // term_n = (-1)^{n+1} x^n / (n * n!)
  // term_1 = x
  // term_n / term_{n-1} = -x * (n-1) / (n * n)
  T term = x;  // First term: x
  result += term;

  const int max_iterations = 100;
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

// Continued fraction for E1(x) for x > 1
// Using the well-known continued fraction representation
template <typename T>
T exponential_integral_e_1_continued_fraction(T x) {
  const T tiny = std::numeric_limits<T>::min() * T(1e10);
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);
  const int max_iterations = 100;

  // E1(x) = exp(-x) / (x + 1/(1 + 1/(x + 2/(1 + 2/(x + 3/(1 + ...))))))
  // Use modified Lentz's method

  T f = x;  // Starting value b_0 = x
  T C = f;
  if (std::abs(C) < tiny) C = tiny;
  T D = T(0);

  for (int n = 1; n <= max_iterations; ++n) {
    T a_n, b_n;
    if (n % 2 == 1) {
      // Odd n: a_n = (n+1)/2, b_n = 1
      a_n = T((n + 1) / 2);
      b_n = T(1);
    } else {
      // Even n: a_n = n/2, b_n = x
      a_n = T(n / 2);
      b_n = x;
    }

    D = b_n + a_n * D;
    if (std::abs(D) < tiny) D = tiny;
    D = T(1) / D;

    C = b_n + a_n / C;
    if (std::abs(C) < tiny) C = tiny;

    T delta = C * D;
    f *= delta;

    if (std::abs(delta - T(1)) < epsilon) {
      break;
    }
  }

  return std::exp(-x) / f;
}

} // namespace detail

// Exponential integral E_1(x)
// E_1(x) = integral from x to infinity of e^(-t)/t dt for x > 0
// For x < 0, returns NaN (undefined for real inputs)
// Related to Ei: E_1(x) = -Ei(-x) for x > 0
template <typename T>
T exponential_integral_e_1(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == T(0)) {
    // E1(0) = +infinity
    return std::numeric_limits<T>::infinity();
  }

  if (std::isinf(x)) {
    if (x > T(0)) {
      // E1(+inf) = 0
      return T(0);
    } else {
      // E1(-inf) is undefined for real inputs
      return std::numeric_limits<T>::quiet_NaN();
    }
  }

  if (x < T(0)) {
    // E1 is undefined for negative real x (returns NaN)
    // The complex extension has a branch cut along the negative real axis
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For positive x, choose algorithm based on magnitude
  if (x <= T(1)) {
    return detail::exponential_integral_e_1_series(x);
  } else {
    return detail::exponential_integral_e_1_continued_fraction(x);
  }
}

// Complex exponential integral E_1(z)
// E_1(z) = integral from z to infinity of e^(-t)/t dt
// Related to Ei: E_1(z) = -Ei(-z)
template <typename T>
c10::complex<T> exponential_integral_e_1(c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Handle special cases
  if (std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // z = 0 is a logarithmic singularity
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(std::numeric_limits<T>::infinity(), T(0));
  }

  T abs_z = std::abs(z);

  // For small |z|, use series expansion
  // E_1(z) = -gamma - ln(z) + sum_{n=1}^inf (-1)^{n+1} z^n / (n * n!)
  if (abs_z <= T(40)) {
    const T gamma = detail::euler_mascheroni_e1<T>();
    Complex result = Complex(-gamma, T(0)) - std::log(z);

    // sum_{n=1}^inf (-1)^{n+1} z^n / (n * n!)
    Complex term = z;  // First term: z
    result += term;

    const int max_iterations = 150;
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

  // For large |z|, use asymptotic expansion
  // E_1(z) ~ exp(-z)/z * sum_{n=0}^N (-1)^n * n! / z^n
  Complex inv_z = Complex(T(1), T(0)) / z;
  Complex sum(T(1), T(0));
  Complex term(T(1), T(0));

  const int max_terms = 50;
  T prev_abs = T(1);

  for (int n = 1; n <= max_terms; ++n) {
    Complex new_term = -term * T(n) * inv_z;  // Note: negative sign for E1
    T curr_abs = std::abs(new_term);

    // Asymptotic series: stop when terms start growing
    if (curr_abs > prev_abs) {
      break;
    }

    term = new_term;
    sum += term;
    prev_abs = curr_abs;

    if (curr_abs < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }
  }

  return std::exp(-z) * inv_z * sum;
}

} // namespace torchscience::kernel::special_functions
