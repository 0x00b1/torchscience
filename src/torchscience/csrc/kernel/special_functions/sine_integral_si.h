#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Sine integral Si(x)
// Si(x) = integral from 0 to x of sin(t)/t dt
// Si(x) is an odd entire function (no singularities)
// Series: Si(x) = sum_{n=0}^inf (-1)^n x^(2n+1) / ((2n+1) * (2n+1)!)

// The series Si(x) = x - x^3/18 + x^5/600 - x^7/35280 + ...
// converges for all x (entire function)

// Helper function: asymptotic expansion for auxiliary functions f and g
// Si(x) = pi/2 - f(x)*cos(x) - g(x)*sin(x)  for x > 0
// where f(x) ~ 1/x - 2!/x^3 + 4!/x^5 - ...
//       g(x) ~ 1/x^2 - 3!/x^4 + 5!/x^6 - ...
template <typename T>
void sine_integral_auxiliary(T x, T& f, T& g) {
  // Compute f(x) and g(x) using asymptotic series
  // f(x) = sum_{n=0}^inf (-1)^n (2n)! / x^(2n+1)
  // g(x) = sum_{n=0}^inf (-1)^n (2n+1)! / x^(2n+2)

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

template <typename T>
T sine_integral_si(T x) {
  // Handle special cases
  if (std::isnan(x)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (x == T(0)) {
    return T(0);
  }

  const T pi_2 = T(1.5707963267948966192313216916397514420985846996876);

  if (std::isinf(x)) {
    // Si(+inf) = pi/2, Si(-inf) = -pi/2
    return x > T(0) ? pi_2 : -pi_2;
  }

  // Use odd function property: Si(-x) = -Si(x)
  T sign = T(1);
  T abs_x = x;
  if (x < T(0)) {
    sign = T(-1);
    abs_x = -x;
  }

  // For large |x|, use asymptotic expansion
  // Threshold determined empirically for good accuracy
  // The Taylor series suffers from catastrophic cancellation for large x,
  // especially in float32 where it fails around x > 12.
  // The asymptotic expansion becomes accurate for x > ~10.
  // Using 10 as threshold works well for both float32 and float64.
  if (abs_x > T(10)) {
    T f, g;
    sine_integral_auxiliary(abs_x, f, g);
    // Si(x) = pi/2 - f(x)*cos(x) - g(x)*sin(x)  for x > 0
    T cos_x = std::cos(abs_x);
    T sin_x = std::sin(abs_x);
    return sign * (pi_2 - f * cos_x - g * sin_x);
  }

  // Series: Si(x) = sum_{n=0}^inf (-1)^n x^(2n+1) / ((2n+1) * (2n+1)!)
  // term_0 = x / (1 * 1!) = x
  // term_n / term_{n-1} = -x^2 * (2n-1) / (2n * (2n+1)^2)
  //
  // Derivation:
  // term_{n-1} = (-1)^{n-1} x^{2n-1} / ((2n-1) * (2n-1)!)
  // term_n     = (-1)^n x^{2n+1} / ((2n+1) * (2n+1)!)
  // ratio = -x^2 * (2n-1) * (2n-1)! / ((2n+1) * (2n+1)!)
  //       = -x^2 * (2n-1) / ((2n+1) * (2n+1) * 2n * (2n-1)!)/(2n-1)!)
  //       = -x^2 * (2n-1) / (2n * (2n+1)^2)

  T result = abs_x;  // First term: n=0, x^1 / (1 * 1!) = x
  T term = abs_x;
  T x_sq = abs_x * abs_x;

  const int max_iterations = 200;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 1; n <= max_iterations; ++n) {
    int two_n = 2 * n;
    int two_n_plus_1 = two_n + 1;
    int two_n_minus_1 = two_n - 1;
    // term_n = term_{n-1} * (-x^2) * (2n-1) / (2n * (2n+1)^2)
    term *= -x_sq * T(two_n_minus_1) / (T(two_n) * T(two_n_plus_1) * T(two_n_plus_1));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return sign * result;
}

// Complex sine integral Si(z)
template <typename T>
c10::complex<T> sine_integral_si(c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // Handle special cases
  if (std::isnan(z.real()) || std::isnan(z.imag())) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  // z = 0: Si(0) = 0
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(T(0), T(0));
  }

  // Series expansion: Si(z) = sum_{n=0}^inf (-1)^n z^(2n+1) / ((2n+1) * (2n+1)!)
  // This converges for all z since Si is entire
  // term_n / term_{n-1} = -z^2 * (2n-1) / (2n * (2n+1)^2)

  Complex result = z;  // First term: z
  Complex term = z;
  Complex z_sq = z * z;

  const int max_iterations = 300;
  const T epsilon = std::numeric_limits<T>::epsilon() * T(10);

  for (int n = 1; n <= max_iterations; ++n) {
    int two_n = 2 * n;
    int two_n_plus_1 = two_n + 1;
    int two_n_minus_1 = two_n - 1;
    // term_n = term_{n-1} * (-z^2) * (2n-1) / (2n * (2n+1)^2)
    term *= -z_sq * T(two_n_minus_1) / (T(two_n) * T(two_n_plus_1) * T(two_n_plus_1));
    result += term;

    if (std::abs(term) < epsilon * std::abs(result)) {
      break;
    }
  }

  return result;
}

} // namespace torchscience::kernel::special_functions
