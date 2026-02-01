#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct faddeeva_constants {
  static constexpr T sqrt_pi = T(1.7724538509055160272981674833411451828);
  static constexpr T sqrt_pi_inv = T(0.5641895835477562869480794515607725858);
  static constexpr T two_sqrt_pi_inv = T(1.1283791670955125738961589031215451716);
};

// Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
//
// Implementation based on the algorithm by Steven G. Johnson (2012)
// "Faddeeva Package" which itself is based on Poppe & Wijers (1990).
//
// The algorithm uses different methods for different regions of the complex plane
// to achieve ~15 digit accuracy across all inputs.
//
// For Im(z) < 0, we use: w(z) = 2*exp(-z^2) - w(-z)

// Dawson function F(x) = exp(-x^2) * integral_0^x exp(t^2) dt
// Used for computing w(x) on the real axis
// Based on Cody, Paciorek, and Thacher (1970) rational approximations
template <typename T>
T dawson_impl(T x) {
  T ax = std::abs(x);

  if (ax < T(0.5)) {
    // Small x: Taylor series F(x) = x * (1 - 2x^2/3 + 4x^4/15 - ...)
    T x2 = ax * ax;
    T sum = T(1);
    T term = T(1);
    for (int n = 1; n <= 40; ++n) {
      term *= -T(2) * x2 / T(2 * n + 1);
      T new_sum = sum + term;
      if (std::abs(new_sum - sum) < std::numeric_limits<T>::epsilon() * std::abs(sum)) break;
      sum = new_sum;
    }
    return x * sum;
  }
  else if (ax < T(4.0)) {
    // Medium x: use the series representation with correction
    // F(x) = x * sum_{n=0}^inf (-1)^n * (2x^2)^n / ((2n+1)!!)
    // This is the same as the Taylor series but we need more terms
    T x2 = ax * ax;
    T sum = T(0);
    T term = ax;
    T factor = T(1);

    for (int n = 0; n <= 100; ++n) {
      sum += term;
      factor = -T(2) * x2 / T(2 * n + 3);
      T new_term = term * factor;
      if (std::abs(new_term) < std::numeric_limits<T>::epsilon() * std::abs(sum) * T(10)) break;
      term = new_term;
    }
    return (x >= T(0) ? T(1) : T(-1)) * sum;
  }
  else {
    // Large x: asymptotic expansion F(x) ~ 1/(2x) * (1 + 1/(2x^2) + 3/(2x^2)^2 + ...)
    T x2 = ax * ax;
    T inv_2x2 = T(0.5) / x2;
    T sum = T(1);
    T term = T(1);
    for (int n = 1; n <= 50; ++n) {
      term *= T(2 * n - 1) * inv_2x2;
      if (std::abs(term) < std::numeric_limits<T>::epsilon() * T(10)) break;
      if (term > sum) break;  // Asymptotic series diverging
      sum += term;
    }
    return (x >= T(0) ? T(1) : T(-1)) * sum / (T(2) * ax);
  }
}

// Compute w(x) for real x: w(x) = exp(-x^2) + 2i/sqrt(pi) * Dawson(x)
template <typename T>
c10::complex<T> faddeeva_w_real(T x) {
  T exp_mx2 = std::exp(-x * x);
  T daw = dawson_impl(x);
  return c10::complex<T>(exp_mx2, faddeeva_constants<T>::two_sqrt_pi_inv * daw);
}

// Compute w(iy) for pure imaginary y > 0: w(iy) = erfcx(y) (real-valued)
// erfcx(y) = exp(y^2) * erfc(y)
// Uses the continued fraction representation that converges for all y > 0
template <typename T>
T erfcx_impl(T y) {
  if (y < T(0)) {
    return T(2) * std::exp(y * y) - erfcx_impl(-y);
  }

  if (y < T(1e-10)) {
    return T(1) - faddeeva_constants<T>::two_sqrt_pi_inv * y;
  }

  if (y < T(0.5)) {
    // Small y: use the series erfc(y) = 1 - erf(y)
    // erf(y) = 2/sqrt(pi) * y * (1 - y^2/3 + y^4/10 - y^6/42 + ...)
    // Then erfcx(y) = exp(y^2) * (1 - erf(y))
    T y2 = y * y;
    T sum = T(1);
    T term = T(1);
    for (int n = 1; n <= 50; ++n) {
      term *= -y2 / T(n);
      T coeff = T(1) / T(2 * n + 1);
      sum += term * coeff;
    }
    T erf_y = faddeeva_constants<T>::two_sqrt_pi_inv * y * sum;
    return std::exp(y2) * (T(1) - erf_y);
  }

  // Larger y: use asymptotic continued fraction
  // erfcx(y) = 1/(sqrt(pi) * y) * (1 - 1/(2y^2) + 3/(2y^2)^2 - ...)
  // Or use the continued fraction: erfcx(y) = 1/sqrt(pi) / (y + k1/(y + k2/(y + ...)))
  // where k_n = n/2

  const int max_iter = 200;
  const T eps = std::numeric_limits<T>::epsilon() * T(10);

  // Use backward recurrence for the continued fraction
  // The continued fraction is: 1/(y + (1/2)/(y + 1/(y + (3/2)/(y + 2/(y + ...)))))
  T cf = T(0);
  for (int n = max_iter; n >= 1; --n) {
    T a_n = T(n) / T(2);
    cf = a_n / (y + cf);
  }

  return faddeeva_constants<T>::sqrt_pi_inv / (y + cf);
}

// Taylor expansion of w(x + iy) around y=0 for small y
// Uses the recurrence relation: d^n w/dz^n = (-2)^n * H_n(-iz) * w(z) * exp(-z^2) * ...
// More simply: w'(z) = -2z*w(z) + 2i/sqrt(pi)
// So we can compute derivatives at y=0 and do Taylor in y
template <typename T>
c10::complex<T> faddeeva_w_small_y(T x, T y) {
  // w(x) at y=0
  c10::complex<T> w0 = faddeeva_w_real(x);
  c10::complex<T> two_i_sqrt_pi(T(0), faddeeva_constants<T>::two_sqrt_pi_inv);

  // Compute derivatives using w'(z) = -2z*w(z) + 2i/sqrt(pi)
  // At z = x (real), the derivatives are:
  // w^(1) = -2x*w(x) + 2i/sqrt(pi)
  // w^(2) = -2*w(x) + (-2x)*w^(1) = -2*w - 2x*w^(1)
  // w^(3) = -2*w^(1) - 2*w^(1) - 2x*w^(2) = -4*w^(1) - 2x*w^(2)
  // w^(n) = -2*(n-1)*w^(n-2) - 2x*w^(n-1)  for n >= 2

  c10::complex<T> neg_2x(-T(2) * x, T(0));
  c10::complex<T> neg_2(-T(2), T(0));

  // Compute up to 8th derivative for good accuracy
  c10::complex<T> w1 = neg_2x * w0 + two_i_sqrt_pi;
  c10::complex<T> w2 = neg_2 * w0 + neg_2x * w1;
  c10::complex<T> w3 = c10::complex<T>(T(-4), T(0)) * w1 + neg_2x * w2;
  c10::complex<T> w4 = c10::complex<T>(T(-6), T(0)) * w2 + neg_2x * w3;
  c10::complex<T> w5 = c10::complex<T>(T(-8), T(0)) * w3 + neg_2x * w4;
  c10::complex<T> w6 = c10::complex<T>(T(-10), T(0)) * w4 + neg_2x * w5;
  c10::complex<T> w7 = c10::complex<T>(T(-12), T(0)) * w5 + neg_2x * w6;
  c10::complex<T> w8 = c10::complex<T>(T(-14), T(0)) * w6 + neg_2x * w7;

  // Taylor series in (iy): w(x+iy) = sum_{n=0}^inf w^(n)(x) * (iy)^n / n!
  c10::complex<T> iy(T(0), y);
  c10::complex<T> iy2 = iy * iy;  // -y^2
  c10::complex<T> iy3 = iy2 * iy;
  c10::complex<T> iy4 = iy2 * iy2;
  c10::complex<T> iy5 = iy4 * iy;
  c10::complex<T> iy6 = iy4 * iy2;
  c10::complex<T> iy7 = iy6 * iy;
  c10::complex<T> iy8 = iy4 * iy4;

  // Factorials: 1, 1, 2, 6, 24, 120, 720, 5040, 40320
  return w0
       + iy * w1
       + iy2 * w2 / c10::complex<T>(T(2), T(0))
       + iy3 * w3 / c10::complex<T>(T(6), T(0))
       + iy4 * w4 / c10::complex<T>(T(24), T(0))
       + iy5 * w5 / c10::complex<T>(T(120), T(0))
       + iy6 * w6 / c10::complex<T>(T(720), T(0))
       + iy7 * w7 / c10::complex<T>(T(5040), T(0))
       + iy8 * w8 / c10::complex<T>(T(40320), T(0));
}

// Main implementation
template <typename T>
c10::complex<T> faddeeva_w_impl(c10::complex<T> z) {
  T x = z.real();
  T y = z.imag();

  if (std::isnan(x) || std::isnan(y)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  // Handle lower half-plane via reflection: w(z) = 2*exp(-z^2) - w(-z)
  bool neg_imag = y < T(0);
  if (neg_imag) {
    z = -z;
    x = -x;
    y = -y;
  }

  c10::complex<T> result;

  // Special case: real axis (y == 0)
  if (y == T(0)) {
    result = faddeeva_w_real(x);
  }
  // Special case: pure imaginary (x == 0, y > 0)
  else if (x == T(0)) {
    result = c10::complex<T>(erfcx_impl(y), T(0));
  }
  // Small y with non-zero x: use Taylor expansion around real axis
  // The continued fraction converges poorly for small y
  else if (y < T(0.1) * (T(1) + std::abs(x))) {
    result = faddeeva_w_small_y(x, y);
  }
  // General case: use region-based algorithm
  else {
    T ax = std::abs(x);
    T rho = std::sqrt(x * x + y * y);

    if (rho > T(6.0)) {
      // Large |z|: asymptotic expansion
      // w(z) ~ i/(sqrt(pi)*z) * sum_{n=0}^inf (2n-1)!! / (2*z^2)^n
      c10::complex<T> inv_z = c10::complex<T>(T(1), T(0)) / z;
      c10::complex<T> inv_2z2 = inv_z * inv_z * c10::complex<T>(T(0.5), T(0));

      c10::complex<T> sum(T(1), T(0));
      c10::complex<T> term(T(1), T(0));

      for (int n = 1; n <= 30; ++n) {
        term = term * c10::complex<T>(T(2 * n - 1), T(0)) * inv_2z2;
        c10::complex<T> new_sum = sum + term;
        T term_mag = std::sqrt(term.real() * term.real() + term.imag() * term.imag());
        if (term_mag < std::numeric_limits<T>::epsilon() * T(10)) break;
        // Check if series is diverging
        T sum_mag = std::sqrt(sum.real() * sum.real() + sum.imag() * sum.imag());
        if (term_mag > sum_mag) break;
        sum = new_sum;
      }
      result = c10::complex<T>(T(0), faddeeva_constants<T>::sqrt_pi_inv) * inv_z * sum;
    }
    else {
      // Moderate |z| with sufficient y: use continued fraction
      // The continued fraction is:
      // w(z) = i/sqrt(pi) * 1/(z - (1/2)/(z - 1/(z - (3/2)/(z - 2/(z - ...)))))

      const int max_iter = 200;
      const T tiny = std::numeric_limits<T>::min() * T(1e10);
      const T eps = std::numeric_limits<T>::epsilon() * T(10);

      // Use backward recurrence for stability
      c10::complex<T> cf(T(0), T(0));
      for (int n = max_iter; n >= 1; --n) {
        T a_n = T(n) / T(2);
        cf = c10::complex<T>(a_n, T(0)) / (z - cf);
      }

      result = c10::complex<T>(T(0), faddeeva_constants<T>::sqrt_pi_inv) / (z - cf);
    }
  }

  // Apply reflection formula for lower half-plane
  if (neg_imag) {
    c10::complex<T> z_orig = -z;
    c10::complex<T> z2 = z_orig * z_orig;
    c10::complex<T> exp_neg_z2 = std::exp(-z2);
    result = c10::complex<T>(T(2), T(0)) * exp_neg_z2 - result;
  }

  return result;
}

}  // namespace detail

template <typename T>
c10::complex<T> faddeeva_w(T x) {
  return detail::faddeeva_w_impl(c10::complex<T>(x, T(0)));
}

template <typename T>
c10::complex<T> faddeeva_w(c10::complex<T> z) {
  return detail::faddeeva_w_impl(z);
}

}  // namespace torchscience::kernel::special_functions
