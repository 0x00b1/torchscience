#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Power series for Fresnel C (converges for all z, but faster for small |z|)
// C(z) = sum_{n=0}^{inf} (-1)^n * (pi/2)^{2n} * z^{4n+1} / ((2n)! * (4n+1))
template <typename T>
T fresnel_c_series(T z) {
  const T pi_over_2 = static_cast<T>(1.5707963267948966192313216916397514421);
  T z2 = z * z;
  T z4 = z2 * z2;

  // First term: z
  T term = z;
  T sum = term;
  T pi2_over_4 = pi_over_2 * pi_over_2;

  // Subsequent terms using recurrence relation
  for (int n = 1; n < 100; ++n) {
    // term_n / term_{n-1} = -pi^2/4 * z^4 * (4n-3) / ((2n-1)(2n) * (4n+1))
    term *= -pi2_over_4 * z4 * static_cast<T>(4 * n - 3) /
            (static_cast<T>(2 * n - 1) * static_cast<T>(2 * n) *
             static_cast<T>(4 * n + 1));
    T new_sum = sum + term;
    if (new_sum == sum) {  // Converged
      break;
    }
    sum = new_sum;
  }

  return sum;
}

// Complex power series for Fresnel C
template <typename T>
c10::complex<T> fresnel_c_series_complex(c10::complex<T> z) {
  const T pi_over_2 = static_cast<T>(1.5707963267948966192313216916397514421);
  c10::complex<T> z2 = z * z;
  c10::complex<T> z4 = z2 * z2;

  c10::complex<T> term = z;
  c10::complex<T> sum = term;
  T pi2_over_4 = pi_over_2 * pi_over_2;

  for (int n = 1; n < 100; ++n) {
    term *= -pi2_over_4 * z4 * static_cast<T>(4 * n - 3) /
            (static_cast<T>(2 * n - 1) * static_cast<T>(2 * n) *
             static_cast<T>(4 * n + 1));
    c10::complex<T> new_sum = sum + term;
    if (std::abs(term) < std::abs(sum) * std::numeric_limits<T>::epsilon()) {
      break;
    }
    sum = new_sum;
  }

  return sum;
}

// Asymptotic expansion for large |z| using auxiliary functions f and g
// From Abramowitz & Stegun 7.3.28-7.3.31:
// f(z) = 1/(pi*z) * [1 - 1*3/(pi*z^2)^2 + 1*3*5*7/(pi*z^2)^4 - ...]
// g(z) = 1/(pi^2*z^3) * [1 - 3*5/(pi*z^2)^2 + 3*5*7*9/(pi*z^2)^4 - ...]
// S(z) = 0.5 - f(z)*cos(pi*z^2/2) - g(z)*sin(pi*z^2/2)
// C(z) = 0.5 + f(z)*sin(pi*z^2/2) - g(z)*cos(pi*z^2/2)
template <typename T>
void fresnel_c_fg_asymptotic(T z, T& f, T& g) {
  const T pi = static_cast<T>(3.14159265358979323846264338327950288);
  T pi_z2 = pi * z * z;
  T inv_pi_z2 = static_cast<T>(1) / pi_z2;
  T inv_pi_z2_sq = inv_pi_z2 * inv_pi_z2;

  T term_f = static_cast<T>(1);
  T sum_f = term_f;

  for (int n = 1; n <= 25; ++n) {
    T coef = static_cast<T>(4 * n - 3) * static_cast<T>(4 * n - 1);
    term_f *= -coef * inv_pi_z2_sq;
    if (std::abs(term_f) > std::abs(sum_f) * T(0.5)) break;
    T new_sum = sum_f + term_f;
    if (new_sum == sum_f) break;
    sum_f = new_sum;
  }
  f = sum_f / (pi * z);

  T term_g = static_cast<T>(1);
  T sum_g = term_g;

  for (int n = 1; n <= 25; ++n) {
    T coef = static_cast<T>(4 * n - 1) * static_cast<T>(4 * n + 1);
    term_g *= -coef * inv_pi_z2_sq;
    if (std::abs(term_g) > std::abs(sum_g) * T(0.5)) break;
    T new_sum = sum_g + term_g;
    if (new_sum == sum_g) break;
    sum_g = new_sum;
  }
  g = sum_g / (pi * pi * z * z * z);
}

// Complex asymptotic expansion
template <typename T>
void fresnel_c_fg_asymptotic_complex(c10::complex<T> z, c10::complex<T>& f, c10::complex<T>& g) {
  const T pi = static_cast<T>(3.14159265358979323846264338327950288);
  c10::complex<T> pi_z2 = pi * z * z;
  c10::complex<T> inv_pi_z2 = c10::complex<T>(1, 0) / pi_z2;
  c10::complex<T> inv_pi_z2_sq = inv_pi_z2 * inv_pi_z2;

  c10::complex<T> term_f(1, 0);
  c10::complex<T> sum_f = term_f;

  for (int n = 1; n <= 25; ++n) {
    T coef = static_cast<T>(4 * n - 3) * static_cast<T>(4 * n - 1);
    term_f *= -coef * inv_pi_z2_sq;
    if (std::abs(term_f) > std::abs(sum_f) * T(0.5)) break;
    c10::complex<T> new_sum = sum_f + term_f;
    if (std::abs(new_sum - sum_f) < std::abs(sum_f) * std::numeric_limits<T>::epsilon()) break;
    sum_f = new_sum;
  }
  f = sum_f / (pi * z);

  c10::complex<T> term_g(1, 0);
  c10::complex<T> sum_g = term_g;

  for (int n = 1; n <= 25; ++n) {
    T coef = static_cast<T>(4 * n - 1) * static_cast<T>(4 * n + 1);
    term_g *= -coef * inv_pi_z2_sq;
    if (std::abs(term_g) > std::abs(sum_g) * T(0.5)) break;
    c10::complex<T> new_sum = sum_g + term_g;
    if (std::abs(new_sum - sum_g) < std::abs(sum_g) * std::numeric_limits<T>::epsilon()) break;
    sum_g = new_sum;
  }
  g = sum_g / (pi * pi * z * z * z);
}

}  // namespace detail

// Fresnel cosine integral C(z) = integral from 0 to z of cos(pi*t^2/2) dt
// Properties:
//   C(0) = 0
//   C(-z) = -C(z) (odd function)
//   C(z) -> 0.5 as z -> +inf
//   C(z) -> -0.5 as z -> -inf
template <typename T>
T fresnel_c(T z) {
  // Handle special cases
  if (std::isnan(z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (z == static_cast<T>(0)) {
    return static_cast<T>(0);
  }

  if (std::isinf(z)) {
    return (z > 0) ? static_cast<T>(0.5) : static_cast<T>(-0.5);
  }

  // Handle negative z using odd symmetry
  T sign = (z < 0) ? static_cast<T>(-1) : static_cast<T>(1);
  T az = std::abs(z);

  const T pi_over_2 = static_cast<T>(1.5707963267948966192313216916397514421);

  // Use power series for moderate z, asymptotic for large z
  T threshold = static_cast<T>(4.5);

  T result;
  if (az < threshold) {
    result = detail::fresnel_c_series(az);
  } else {
    T f, g;
    detail::fresnel_c_fg_asymptotic(az, f, g);
    T arg = pi_over_2 * az * az;
    // C(z) = 0.5 + f(z)*sin(pi*z^2/2) - g(z)*cos(pi*z^2/2)
    result = static_cast<T>(0.5) + f * std::sin(arg) - g * std::cos(arg);
  }

  return sign * result;
}

// Complex version of Fresnel C (overload for c10::complex)
template <typename T>
c10::complex<T> fresnel_c(c10::complex<T> z) {
  const T pi_over_2 = static_cast<T>(1.5707963267948966192313216916397514421);

  // Handle z = 0
  if (z.real() == T(0) && z.imag() == T(0)) {
    return c10::complex<T>(0, 0);
  }

  T az = std::abs(z);
  T threshold = static_cast<T>(4.5);

  if (az < threshold) {
    return detail::fresnel_c_series_complex(z);
  } else {
    c10::complex<T> f, g;
    detail::fresnel_c_fg_asymptotic_complex(z, f, g);
    c10::complex<T> arg = pi_over_2 * z * z;
    // C(z) = 0.5 + f(z)*sin(pi*z^2/2) - g(z)*cos(pi*z^2/2)
    return c10::complex<T>(0.5, 0) + f * c10_complex_math::sin(arg) - g * c10_complex_math::cos(arg);
  }
}

}  // namespace torchscience::kernel::special_functions
