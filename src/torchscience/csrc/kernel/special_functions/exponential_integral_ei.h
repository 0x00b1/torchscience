#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Euler-Mascheroni constant
template <typename T>
constexpr T euler_mascheroni() {
  return T(0.5772156649015328606065120900824024310421593359);
}

// Series expansion for Ei(x) for small to moderate positive x
// Ei(x) = gamma + ln(x) + sum_{n=1}^inf x^n / (n * n!)
template <typename T>
T exponential_integral_ei_series(T x) {
  const T gamma = euler_mascheroni<T>();
  T result = gamma + std::log(x);

  // Compute sum_{n=1}^inf x^n / (n * n!)
  // term_n = x^n / (n * n!)
  // term_1 = x / (1 * 1!) = x
  // term_n / term_{n-1} = x * (n-1) / (n * n)
  T term = x;  // First term: x^1 / (1 * 1!) = x
  result += term;

  const int max_iterations = 100;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 2; n <= max_iterations; ++n) {
    // term_n = term_{n-1} * x * (n-1) / (n * n)
    term *= x * T(n - 1) / (T(n) * T(n));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

// Asymptotic expansion for Ei(x) for large positive x
// Ei(x) ~ exp(x)/x * sum_{n=0}^inf n!/x^n
template <typename T>
T exponential_integral_ei_asymptotic(T x) {
  T inv_x = T(1) / x;
  T sum = T(1);
  T term = T(1);

  const int max_terms = 50;

  for (int n = 1; n <= max_terms; ++n) {
    T new_term = term * T(n) * inv_x;
    // Asymptotic series: stop when terms start growing
    if (std::abs(new_term) > std::abs(term)) {
      break;
    }
    term = new_term;
    sum += term;
  }

  return std::exp(x) * inv_x * sum;
}

// Continued fraction for E1(x) for x > 0
// Using the well-known continued fraction representation
template <typename T>
T exponential_integral_e1_continued_fraction(T x) {
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

// Series for E1(x) for small positive x
// E1(x) = -gamma - ln(x) + sum_{n=1}^inf (-1)^{n+1} x^n / (n * n!)
template <typename T>
T exponential_integral_e1_series(T x) {
  const T gamma = euler_mascheroni<T>();
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

} // namespace detail

// Exponential integral Ei(x)
// Ei(x) = -PV integral from -x to infinity of e^(-t)/t dt
//       = integral from -infinity to x of e^t/t dt (principal value)
template <typename T>
T exponential_integral_ei(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (std::isinf(x)) {
    return x > T(0) ? std::numeric_limits<T>::infinity() : T(0);
  }

  if (x > T(0)) {
    // For positive x, use series for small/moderate x, asymptotic for large x
    if (x <= T(40)) {
      return detail::exponential_integral_ei_series(x);
    } else {
      return detail::exponential_integral_ei_asymptotic(x);
    }
  } else {
    // For negative x: Ei(x) = -E1(-x)
    // where E1 is the exponential integral of the first kind
    T abs_x = -x;

    if (abs_x <= T(1)) {
      // Use series for E1
      T e1 = detail::exponential_integral_e1_series(abs_x);
      return -e1;  // Ei(x) = -E1(-x)
    } else {
      // Use continued fraction for E1
      T e1 = detail::exponential_integral_e1_continued_fraction(abs_x);
      return -e1;  // Ei(x) = -E1(-x)
    }
  }
}

// Complex exponential integral Ei(z)
template <typename T>
c10::complex<T> exponential_integral_ei(c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Handle special cases
  if (std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // z = 0 is a logarithmic singularity
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(-std::numeric_limits<T>::infinity(), T(0));
  }

  T abs_z = std::abs(z);

  // For small |z|, use series expansion
  // Ei(z) = gamma + ln(z) + sum_{n=1}^inf z^n / (n * n!)
  if (abs_z <= T(40)) {
    const T gamma = detail::euler_mascheroni<T>();
    Complex result = Complex(gamma, T(0)) + std::log(z);

    Complex term = z;
    result += term;

    const int max_iterations = 150;
    const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

    for (int n = 2; n <= max_iterations; ++n) {
      // term_n = term_{n-1} * z * (n-1) / (n * n)
      term *= z * T(n - 1) / (T(n) * T(n));
      result += term;

      if (std::abs(term) < epsilon * std::abs(result)) {
        break;
      }
    }

    return result;
  }

  // For large |z|, use asymptotic expansion
  // Ei(z) ~ exp(z)/z * sum_{n=0}^N n!/z^n
  Complex inv_z = Complex(T(1), T(0)) / z;
  Complex sum(T(1), T(0));
  Complex term(T(1), T(0));

  const int max_terms = 50;
  T prev_abs = T(1);

  for (int n = 1; n <= max_terms; ++n) {
    Complex new_term = term * T(n) * inv_z;
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

  return std::exp(z) * inv_z * sum;
}

} // namespace torchscience::kernel::special_functions
