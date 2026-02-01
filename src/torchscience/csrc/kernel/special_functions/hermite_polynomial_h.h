#pragma once

#include <cmath>
#include <type_traits>

#include <c10/util/complex.h>

#include "confluent_hypergeometric_m.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for Hermite polynomial
template <typename T>
struct hermite_is_complex_type : std::false_type {};

template <typename T>
struct hermite_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hermite_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool hermite_is_complex_v = hermite_is_complex_type<T>::value;

template <typename T>
struct hermite_real_type { using type = T; };

template <typename T>
struct hermite_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hermite_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using hermite_real_type_t = typename hermite_real_type<T>::type;

template <typename T>
constexpr auto hermite_epsilon() {
  using real_t = hermite_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-6);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-10);
  } else {
    return float(1e-6);
  }
}

// Check if n is a non-negative integer
template <typename T>
bool hermite_is_nonneg_integer(T n) {
  if constexpr (hermite_is_complex_v<T>) {
    using real_t = hermite_real_type_t<T>;
    auto re = static_cast<real_t>(n.real());
    auto im = static_cast<real_t>(n.imag());
    return std::abs(im) < hermite_epsilon<T>() &&
           re >= real_t(0) &&
           std::abs(re - std::round(re)) < hermite_epsilon<T>();
  } else {
    return n >= T(0) && std::abs(n - std::round(n)) < hermite_epsilon<T>();
  }
}

template <typename T>
int hermite_get_integer(T n) {
  if constexpr (hermite_is_complex_v<T>) {
    using real_t = hermite_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(n.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(n)));
  }
}

// Compute H_n(z) for non-negative integer n using recurrence relation
// H_{n+1}(z) = 2z * H_n(z) - 2n * H_{n-1}(z)
template <typename T>
T hermite_integer_recurrence(int n, T z) {
  if (n == 0) {
    return T(1);
  }
  if (n == 1) {
    return T(2) * z;
  }

  T H_prev = T(1);       // H_0
  T H_curr = T(2) * z;   // H_1
  for (int k = 2; k <= n; ++k) {
    T H_next = T(2) * z * H_curr - T(2) * T(k - 1) * H_prev;
    H_prev = H_curr;
    H_curr = H_next;
  }
  return H_curr;
}

// Compute H_n(z) for non-integer n using confluent hypergeometric functions
template <typename T>
T hermite_hypergeometric(T n, T z) {
  const T pi = T(3.14159265358979323846);
  const T sqrt_pi = std::sqrt(pi);

  T two_pow_n = std::pow(T(2), n);
  T z_sq = z * z;

  // Term 1: 2^n * sqrt(pi) / Gamma((1-n)/2) * 1F1(-n/2; 1/2; z^2)
  T a1 = -n / T(2);
  T b1 = T(0.5);
  T gamma1 = gamma((T(1) - n) / T(2));
  T hyp1 = confluent_hypergeometric_m(a1, b1, z_sq);
  T term1 = two_pow_n * sqrt_pi / gamma1 * hyp1;

  // Term 2: 2^{n+1} * sqrt(pi) * z / Gamma(-n/2) * 1F1((1-n)/2; 3/2; z^2)
  T a2 = (T(1) - n) / T(2);
  T b2 = T(1.5);
  T gamma2 = gamma(-n / T(2));
  T hyp2 = confluent_hypergeometric_m(a2, b2, z_sq);
  T term2 = std::pow(T(2), n + T(1)) * sqrt_pi * z / gamma2 * hyp2;

  return term1 + term2;
}

} // namespace detail

// Physicists' Hermite polynomial H_n(z)
//
// For non-negative integer n: uses the recurrence relation
// H_{n+1}(z) = 2z * H_n(z) - 2n * H_{n-1}(z)
// with H_0(z) = 1, H_1(z) = 2z
//
// For non-integer n: uses the confluent hypergeometric representation
// H_n(z) = 2^n * sqrt(pi) / Gamma((1-n)/2) * 1F1(-n/2; 1/2; z^2)
//        + 2^{n+1} * sqrt(pi) * z / Gamma(-n/2) * 1F1((1-n)/2; 3/2; z^2)
template <typename T>
T hermite_polynomial_h(T n, T z) {
  // For non-negative integer n, use the fast recurrence relation
  if (detail::hermite_is_nonneg_integer(n)) {
    int n_int = detail::hermite_get_integer(n);
    return detail::hermite_integer_recurrence(n_int, z);
  }

  // For non-integer n or negative n, use the hypergeometric formula
  return detail::hermite_hypergeometric(n, z);
}

} // namespace torchscience::kernel::special_functions
