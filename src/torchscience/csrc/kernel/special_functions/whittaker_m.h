#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

#include "confluent_hypergeometric_m.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for Whittaker M (use whit_m_ prefix to avoid ODR conflicts)
template <typename T>
struct whit_m_is_complex_type : std::false_type {};

template <typename T>
struct whit_m_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct whit_m_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool whit_m_is_complex_v = whit_m_is_complex_type<T>::value;

template <typename T>
struct whit_m_real_type { using type = T; };

template <typename T>
struct whit_m_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct whit_m_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using whit_m_real_type_t = typename whit_m_real_type<T>::type;

template <typename T>
constexpr auto whit_m_epsilon() {
  using real_t = whit_m_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-6);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-14);
  } else {
    return float(1e-6);
  }
}

// Check if mu is a negative half-integer (e.g., -1/2, -3/2, -5/2, ...)
// These cause poles in the Whittaker M function due to Gamma(2*mu + 1)
template <typename T>
bool whit_m_is_negative_half_integer(T mu) {
  if constexpr (whit_m_is_complex_v<T>) {
    using real_t = whit_m_real_type_t<T>;
    auto re = static_cast<real_t>(mu.real());
    auto im = static_cast<real_t>(mu.imag());
    if (std::abs(im) > whit_m_epsilon<T>()) {
      return false;
    }
    // Check if 2*mu + 1 is a non-positive integer
    double two_mu_plus_1 = 2.0 * static_cast<double>(re) + 1.0;
    return two_mu_plus_1 <= 0.0 &&
           std::abs(two_mu_plus_1 - std::round(two_mu_plus_1)) < whit_m_epsilon<T>();
  } else {
    double two_mu_plus_1 = 2.0 * static_cast<double>(mu) + 1.0;
    return two_mu_plus_1 <= 0.0 &&
           std::abs(two_mu_plus_1 - std::round(two_mu_plus_1)) < whit_m_epsilon<T>();
  }
}

} // namespace detail

// Whittaker M function: M_kappa,mu(z)
//
// The Whittaker M function is defined as:
//   M_kappa,mu(z) = exp(-z/2) * z^(mu + 1/2) * M(mu - kappa + 1/2, 2*mu + 1, z)
//
// where M(a, b, z) is the confluent hypergeometric function of the first kind.
//
// Parameters:
//   kappa: First parameter
//   mu: Second parameter
//   z: Argument
//
// Special cases:
//   - M_kappa,mu(0) = 0 for Re(mu) > -1/2
//   - M_kappa,mu(0) is undefined (pole) for mu = -1/2, -3/2, -5/2, ...
//
template <typename T>
T whittaker_m(T kappa, T mu, T z) {
  using detail::whit_m_epsilon;
  using detail::whit_m_is_complex_v;
  using detail::whit_m_is_negative_half_integer;

  // Handle z = 0 case
  // M_kappa,mu(0) = exp(0) * 0^(mu + 1/2) * M(...)
  // For Re(mu + 1/2) > 0, this is 0
  // For Re(mu + 1/2) <= 0, this involves 0^negative or 0^0
  if (std::abs(z) < whit_m_epsilon<T>()) {
    double mu_plus_half_real;
    if constexpr (whit_m_is_complex_v<T>) {
      mu_plus_half_real = static_cast<double>(mu.real()) + 0.5;
    } else {
      mu_plus_half_real = static_cast<double>(mu) + 0.5;
    }

    if (mu_plus_half_real > whit_m_epsilon<T>()) {
      // z^(mu + 1/2) -> 0 as z -> 0 for Re(mu + 1/2) > 0
      return T(0);
    } else if (std::abs(mu_plus_half_real) < whit_m_epsilon<T>()) {
      // mu + 1/2 = 0, i.e., mu = -1/2
      // z^0 = 1, but M(a, 0, z) has a pole at b = 0
      // 2*mu + 1 = 0 when mu = -1/2
      return std::numeric_limits<T>::infinity();
    } else {
      // Re(mu + 1/2) < 0: z^(mu + 1/2) -> infinity as z -> 0
      return std::numeric_limits<T>::infinity();
    }
  }

  // Check for poles at mu = -1/2, -3/2, -5/2, ... (when 2*mu + 1 is non-positive integer)
  // These cause Gamma(2*mu + 1) to have poles in the confluent hypergeometric M
  if (whit_m_is_negative_half_integer(mu)) {
    // The function M(a, b, z) has a pole when b is a non-positive integer
    // unless a is also a non-positive integer with |a| < |b|
    // For simplicity, return infinity for these cases
    return std::numeric_limits<T>::infinity();
  }

  // Compute the parameters for confluent hypergeometric M
  // a = mu - kappa + 1/2
  // b = 2*mu + 1
  T half = T(0.5);
  T one = T(1);
  T two = T(2);

  T a = mu - kappa + half;
  T b = two * mu + one;

  // Compute the confluent hypergeometric function M(a, b, z)
  T M_val = confluent_hypergeometric_m(a, b, z);

  // Handle case where M returns infinity or NaN
  if (std::isinf(M_val) || std::isnan(M_val)) {
    return M_val;
  }

  // Compute exp(-z/2)
  T exp_factor = std::exp(-z / two);

  // Compute z^(mu + 1/2)
  // Using z^(mu + 1/2) = exp((mu + 1/2) * log(z))
  T mu_plus_half = mu + half;
  T z_power = std::exp(mu_plus_half * std::log(z));

  // Final result: exp(-z/2) * z^(mu + 1/2) * M(a, b, z)
  return exp_factor * z_power * M_val;
}

// Complex version of Whittaker M function
template <typename T>
c10::complex<T> whittaker_m(c10::complex<T> kappa, c10::complex<T> mu, c10::complex<T> z) {
  using detail::whit_m_epsilon;
  using detail::whit_m_is_negative_half_integer;

  // Handle z = 0 case
  if (std::abs(z) < whit_m_epsilon<c10::complex<T>>()) {
    T mu_plus_half_real = mu.real() + T(0.5);

    if (mu_plus_half_real > whit_m_epsilon<c10::complex<T>>()) {
      return c10::complex<T>(T(0), T(0));
    } else if (std::abs(mu_plus_half_real) < whit_m_epsilon<c10::complex<T>>() &&
               std::abs(mu.imag()) < whit_m_epsilon<c10::complex<T>>()) {
      // mu = -1/2
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    } else {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
  }

  // Check for poles
  if (whit_m_is_negative_half_integer(mu)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // Compute parameters
  c10::complex<T> half(T(0.5), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> a = mu - kappa + half;
  c10::complex<T> b = two * mu + one;

  // Compute confluent hypergeometric M
  c10::complex<T> M_val = confluent_hypergeometric_m(a, b, z);

  // Handle infinity or NaN
  if (std::isinf(M_val.real()) || std::isinf(M_val.imag()) ||
      std::isnan(M_val.real()) || std::isnan(M_val.imag())) {
    return M_val;
  }

  // Compute exp(-z/2)
  c10::complex<T> exp_factor = std::exp(-z / two);

  // Compute z^(mu + 1/2)
  c10::complex<T> mu_plus_half = mu + half;
  c10::complex<T> z_power = std::exp(mu_plus_half * std::log(z));

  // Final result
  return exp_factor * z_power * M_val;
}

} // namespace torchscience::kernel::special_functions
