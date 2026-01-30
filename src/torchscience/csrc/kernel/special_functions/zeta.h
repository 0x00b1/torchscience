#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Bernoulli numbers B_2n for Euler-Maclaurin formula
// These are B_2, B_4, B_6, ..., B_24
constexpr double bernoulli_2n[] = {
    1.0 / 6.0,                           // B_2
    -1.0 / 30.0,                         // B_4
    1.0 / 42.0,                          // B_6
    -1.0 / 30.0,                         // B_8
    5.0 / 66.0,                          // B_10
    -691.0 / 2730.0,                     // B_12
    7.0 / 6.0,                           // B_14
    -3617.0 / 510.0,                     // B_16
    43867.0 / 798.0,                     // B_18
    -174611.0 / 330.0,                   // B_20
    854513.0 / 138.0,                    // B_22
    -236364091.0 / 2730.0,               // B_24
};

constexpr int num_bernoulli = 12;

} // namespace detail

// Riemann zeta function for s > 1 using Euler-Maclaurin summation
// For s <= 1, returns NaN (pole at s=1, analytic continuation not implemented)
template <typename T>
T zeta(T s) {
  // Handle special cases
  if (std::isnan(s)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Check domain: only valid for s > 1
  if (s <= T(1)) {
    if (s == T(1)) {
      // Pole at s = 1
      return std::numeric_limits<T>::infinity();
    }
    // For s < 1, analytic continuation not implemented
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For very large s, zeta(s) -> 1
  if (s > T(50)) {
    return T(1);
  }

  // Euler-Maclaurin formula:
  // zeta(s) = sum_{n=1}^{N-1} 1/n^s + 1/((s-1)*N^(s-1)) + 1/(2*N^s)
  //         + sum_{k=1}^{p} B_{2k}/(2k)! * s*(s+1)*...*(s+2k-2) / N^(s+2k-1) + R
  //
  // where N is the cutoff for direct summation and p is the number of
  // Bernoulli terms. The remainder R can be bounded.

  // Choose N based on s for good accuracy
  int N;
  if (s < T(5)) {
    N = 20;
  } else if (s < T(10)) {
    N = 15;
  } else {
    N = 10;
  }

  // Direct summation: sum_{n=1}^{N-1} 1/n^s
  T sum = T(0);
  for (int n = 1; n < N; ++n) {
    sum += std::pow(static_cast<T>(n), -s);
  }

  // Integral term: 1/((s-1)*N^(s-1))
  T N_pow = std::pow(static_cast<T>(N), T(1) - s);
  sum += N_pow / (s - T(1));

  // Boundary term: 1/(2*N^s)
  T N_pow_s = std::pow(static_cast<T>(N), -s);
  sum += N_pow_s / T(2);

  // Euler-Maclaurin correction terms using Bernoulli numbers
  // The k-th term is: B_{2k}/(2k)! * prod_{j=0}^{2k-2}(s+j) / N^(s+2k-1)
  T term_coeff = s / static_cast<T>(N);  // s/N
  T power = N_pow_s / static_cast<T>(N);  // 1/N^(s+1)

  for (int k = 0; k < detail::num_bernoulli; ++k) {
    int two_k = 2 * (k + 1);  // 2, 4, 6, ...

    // Compute coefficient: B_{2k} / (2k)! * s*(s+1)*...*(s+2k-2)
    // We build this incrementally
    T coeff = detail::bernoulli_2n[k];

    // Falling factorial s*(s+1)*...*(s+2k-2) divided by (2k)!
    // For k=1: s/2! = s/2
    // For k=2: s*(s+1)*(s+2)/4! = s*(s+1)*(s+2)/24
    // etc.

    T falling = T(1);
    for (int j = 0; j < two_k - 1; ++j) {
      falling *= (s + static_cast<T>(j)) / static_cast<T>(two_k - j);
    }

    T correction = coeff * falling * power;
    sum += correction;

    // Update power for next iteration: multiply by 1/N^2
    power /= static_cast<T>(N * N);

    // Check for convergence
    if (std::abs(correction) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }
  }

  return sum;
}

// Complex zeta function for Re(s) > 1
template <typename T>
c10::complex<T> zeta(c10::complex<T> s) {
  // For complex s, we need Re(s) > 1
  if (std::isnan(s.real()) || std::isnan(s.imag())) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  if (s.real() <= T(1)) {
    if (s.real() == T(1) && s.imag() == T(0)) {
      // Pole at s = 1
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
    // For Re(s) <= 1, analytic continuation not implemented
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  // For very large Re(s), zeta(s) -> 1
  if (s.real() > T(50)) {
    return c10::complex<T>(T(1), T(0));
  }

  // Similar Euler-Maclaurin approach for complex s
  int N = 20;
  if (s.real() >= T(10)) {
    N = 15;
  }

  // Direct summation
  c10::complex<T> sum(T(0), T(0));
  for (int n = 1; n < N; ++n) {
    sum += std::pow(c10::complex<T>(static_cast<T>(n), T(0)), -s);
  }

  // Integral term
  c10::complex<T> N_complex(static_cast<T>(N), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> N_pow = std::pow(N_complex, one - s);
  sum += N_pow / (s - one);

  // Boundary term
  c10::complex<T> N_pow_s = std::pow(N_complex, -s);
  sum += N_pow_s / c10::complex<T>(T(2), T(0));

  // Euler-Maclaurin correction terms
  c10::complex<T> power = N_pow_s / N_complex;

  for (int k = 0; k < detail::num_bernoulli; ++k) {
    int two_k = 2 * (k + 1);

    T coeff = detail::bernoulli_2n[k];

    c10::complex<T> falling(T(1), T(0));
    for (int j = 0; j < two_k - 1; ++j) {
      falling *= (s + c10::complex<T>(static_cast<T>(j), T(0))) /
                 c10::complex<T>(static_cast<T>(two_k - j), T(0));
    }

    c10::complex<T> correction = c10::complex<T>(coeff, T(0)) * falling * power;
    sum += correction;

    // Update power
    power /= c10::complex<T>(static_cast<T>(N * N), T(0));

    // Check for convergence
    T mag_correction = std::abs(correction);
    T mag_sum = std::abs(sum);
    if (mag_correction < std::numeric_limits<T>::epsilon() * mag_sum) {
      break;
    }
  }

  return sum;
}

} // namespace torchscience::kernel::special_functions
