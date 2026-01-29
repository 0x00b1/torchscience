#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Euler-Mascheroni constant
template <typename T>
constexpr T euler_mascheroni() {
  return T(0.5772156649015328606065120900824024310421593359);
}

// Helper function: asymptotic expansion for auxiliary functions f and g
// Ci(x) = f(x)*sin(x) - g(x)*cos(x) for large x
// where f(x) ~ 1/x - 2!/x^3 + 4!/x^5 - ...
//       g(x) ~ 1/x^2 - 3!/x^4 + 5!/x^6 - ...
template <typename T>
void cosine_integral_auxiliary(T x, T& f, T& g) {
  T inv_x = T(1) / x;
  T inv_x2 = inv_x * inv_x;

  // Compute f(x)
  f = inv_x;  // First term: 1/x
  T f_term = inv_x;
  T prev_f_term_abs = std::abs(f_term);

  // Compute g(x)
  g = inv_x2;  // First term: 1/x^2
  T g_term = inv_x2;
  T prev_g_term_abs = std::abs(g_term);

  const int max_iterations = 50;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 1; n <= max_iterations; ++n) {
    // f term: (-1)^n (2n)! / x^(2n+1)
    // ratio: -((2n)(2n-1)) / x^2
    T f_ratio = -T(2*n) * T(2*n - 1) * inv_x2;
    T f_term_new = f_term * f_ratio;

    // g term: (-1)^n (2n+1)! / x^(2n+2)
    // ratio: -((2n+1)(2n)) / x^2
    T g_ratio = -T(2*n + 1) * T(2*n) * inv_x2;
    T g_term_new = g_term * g_ratio;

    // Check for divergence (asymptotic series eventually diverges)
    // Stop when terms start growing in magnitude
    T f_term_abs = std::abs(f_term_new);
    T g_term_abs = std::abs(g_term_new);
    if (f_term_abs > prev_f_term_abs || g_term_abs > prev_g_term_abs) {
      break;
    }

    f += f_term_new;
    g += g_term_new;
    f_term = f_term_new;
    g_term = g_term_new;
    prev_f_term_abs = f_term_abs;
    prev_g_term_abs = g_term_abs;

    if (f_term_abs < epsilon * std::abs(f) &&
        g_term_abs < epsilon * std::abs(g)) {
      break;
    }
  }
}

// Real cosine integral Ci(x)
// Ci(x) = gamma + ln(x) + integral from 0 to x of (cos(t)-1)/t dt
// Only defined for x > 0, with a logarithmic singularity at x = 0
//
// Series expansion for the integral part:
// integral_0^x (cos(t)-1)/t dt = sum_{n=1}^inf (-1)^n x^(2n) / (2n * (2n)!)
//                              = -x^2/4 + x^4/96 - x^6/4320 + ...
//
// Term ratio: term_n / term_{n-1} = -x^2 * (2n-2) / ((2n) * (2n) * (2n-1))
//                                 = -x^2 * (n-1) / (n * (2n) * (2n-1))
template <typename T>
T cosine_integral_ci(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Ci is only defined for x > 0 (has log singularity at 0)
  if (x <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (std::isinf(x)) {
    return T(0);  // Ci(+inf) = 0
  }

  const T gamma = euler_mascheroni<T>();

  // For large x, use asymptotic expansion
  // Ci(x) = f(x)*sin(x) - g(x)*cos(x) for x > 0
  if (x > T(10)) {
    T f, g;
    cosine_integral_auxiliary(x, f, g);
    T cos_x = std::cos(x);
    T sin_x = std::sin(x);
    return f * sin_x - g * cos_x;
  }

  // For small to moderate x, use series:
  // Ci(x) = gamma + ln(x) + sum_{n=1}^inf (-1)^n x^(2n) / (2n * (2n)!)
  T result = gamma + std::log(x);
  T x_sq = x * x;

  // First term: n=1, -x^2 / (2 * 2!) = -x^2/4
  T term = -x_sq / T(4);
  result += term;

  const int max_iterations = 150;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 2; n <= max_iterations; ++n) {
    int two_n = 2 * n;
    int two_n_minus_2 = two_n - 2;
    // term_n = term_{n-1} * (-x^2) * (2n-2) / ((2n) * (2n) * (2n-1))
    term *= -x_sq * T(two_n_minus_2) / (T(two_n) * T(two_n) * T(two_n - 1));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

// Complex cosine integral Ci(z)
// For complex z, use analytic continuation via power series
// Ci(z) = gamma + ln(z) + sum_{n=1}^inf (-1)^n z^(2n) / (2n * (2n)!)
template <typename T>
c10::complex<T> cosine_integral_ci(c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Handle special cases
  if (std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // z = 0: Ci(0) = -inf (log singularity)
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(-std::numeric_limits<T>::infinity(), T(0));
  }

  const T gamma = euler_mascheroni<T>();

  // Series: Ci(z) = gamma + ln(z) + sum_{n=1}^inf (-1)^n z^(2n) / (2n * (2n)!)
  Complex result = Complex(gamma, T(0)) + std::log(z);
  Complex z_sq = z * z;

  // First term: n=1, -z^2 / 4
  Complex term = -z_sq / T(4);
  result += term;

  const int max_iterations = 200;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 2; n <= max_iterations; ++n) {
    int two_n = 2 * n;
    int two_n_minus_2 = two_n - 2;
    // term_n = term_{n-1} * (-z^2) * (2n-2) / ((2n) * (2n) * (2n-1))
    term *= -z_sq * T(two_n_minus_2) / (T(two_n) * T(two_n) * T(two_n - 1));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

} // namespace torchscience::kernel::special_functions
