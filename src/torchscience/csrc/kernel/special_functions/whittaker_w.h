#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

#include "confluent_hypergeometric_u.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for Whittaker W (use whit_w_ prefix to avoid ODR conflicts)
template <typename T>
struct whit_w_is_complex_type : std::false_type {};

template <typename T>
struct whit_w_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct whit_w_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool whit_w_is_complex_v = whit_w_is_complex_type<T>::value;

template <typename T>
struct whit_w_real_type { using type = T; };

template <typename T>
struct whit_w_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct whit_w_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using whit_w_real_type_t = typename whit_w_real_type<T>::type;

template <typename T>
constexpr auto whit_w_epsilon() {
  using real_t = whit_w_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-6);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-14);
  } else {
    return float(1e-6);
  }
}

// Check if mu is a negative half-integer (e.g., -1/2, -3/2, -5/2, ...)
// These may cause special behavior in the Whittaker W function
template <typename T>
bool whit_w_is_negative_half_integer(T mu) {
  if constexpr (whit_w_is_complex_v<T>) {
    using real_t = whit_w_real_type_t<T>;
    auto re = static_cast<real_t>(mu.real());
    auto im = static_cast<real_t>(mu.imag());
    if (std::abs(im) > whit_w_epsilon<T>()) {
      return false;
    }
    // Check if 2*mu + 1 is a non-positive integer
    double two_mu_plus_1 = 2.0 * static_cast<double>(re) + 1.0;
    return two_mu_plus_1 <= 0.0 &&
           std::abs(two_mu_plus_1 - std::round(two_mu_plus_1)) < whit_w_epsilon<T>();
  } else {
    double two_mu_plus_1 = 2.0 * static_cast<double>(mu) + 1.0;
    return two_mu_plus_1 <= 0.0 &&
           std::abs(two_mu_plus_1 - std::round(two_mu_plus_1)) < whit_w_epsilon<T>();
  }
}

} // namespace detail

// Whittaker W function: W_kappa,mu(z)
//
// The Whittaker W function is defined as:
//   W_kappa,mu(z) = exp(-z/2) * z^(mu + 1/2) * U(mu - kappa + 1/2, 2*mu + 1, z)
//
// where U(a, b, z) is the confluent hypergeometric function of the second kind
// (Tricomi function).
//
// Parameters:
//   kappa: First parameter
//   mu: Second parameter
//   z: Argument
//
// Properties:
//   - W_kappa,mu(z) is entire in kappa and mu
//   - W_kappa,mu(z) = W_kappa,-mu(z) (symmetric in mu)
//   - As z -> infinity: W_kappa,mu(z) ~ exp(-z/2) * z^kappa
//
// Special cases:
//   - W_kappa,mu(0) = 0 for Re(mu + 1/2) > 0 and the U term is finite
//   - W_kappa,mu(0) may have poles depending on the parameters
//
template <typename T>
T whittaker_w(T kappa, T mu, T z) {
  using detail::whit_w_epsilon;
  using detail::whit_w_is_complex_v;

  // Handle z = 0 case
  // W_kappa,mu(0) = exp(0) * 0^(mu + 1/2) * U(...)
  // The behavior depends on mu and the U term
  if (std::abs(z) < whit_w_epsilon<T>()) {
    double mu_plus_half_real;
    if constexpr (whit_w_is_complex_v<T>) {
      mu_plus_half_real = static_cast<double>(mu.real()) + 0.5;
    } else {
      mu_plus_half_real = static_cast<double>(mu) + 0.5;
    }

    // U(a, b, z) at z=0:
    // - For Re(b) <= 1: U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1)
    // - For Re(b) > 1: U(a, b, z) ~ z^(1-b) as z -> 0 (pole)
    //
    // Here b = 2*mu + 1, so Re(b) = 2*Re(mu) + 1
    // Re(b) <= 1 iff Re(mu) <= 0
    // Re(b) > 1 iff Re(mu) > 0

    double mu_real;
    if constexpr (whit_w_is_complex_v<T>) {
      mu_real = static_cast<double>(mu.real());
    } else {
      mu_real = static_cast<double>(mu);
    }

    if (mu_plus_half_real > whit_w_epsilon<T>()) {
      // z^(mu + 1/2) -> 0 as z -> 0 for Re(mu + 1/2) > 0
      // If U is bounded at z=0 (Re(2*mu+1) <= 1, i.e., Re(mu) <= 0),
      // then W -> 0
      // If U has a pole (Re(mu) > 0), the behavior is more complex
      if (mu_real <= 0.0) {
        // U is bounded at z=0, z^(mu+1/2) -> 0, so W -> 0
        return T(0);
      } else {
        // U ~ z^(1-b) = z^(-2*mu) as z -> 0
        // W ~ z^(mu+1/2) * z^(-2*mu) = z^(-mu + 1/2)
        // For Re(mu) > 0, Re(-mu + 1/2) could be positive or negative
        if (mu_plus_half_real > mu_real) {
          // z term dominates, W -> 0
          return T(0);
        } else {
          // U pole dominates
          return std::numeric_limits<T>::infinity();
        }
      }
    } else if (std::abs(mu_plus_half_real) < whit_w_epsilon<T>()) {
      // mu + 1/2 = 0, i.e., mu = -1/2
      // z^0 = 1, and b = 2*(-1/2) + 1 = 0
      // U(a, 0, z) has special behavior
      // For b = 0, U is typically infinite unless a is specific
      return std::numeric_limits<T>::infinity();
    } else {
      // Re(mu + 1/2) < 0: z^(mu + 1/2) -> infinity as z -> 0
      return std::numeric_limits<T>::infinity();
    }
  }

  // Compute the parameters for confluent hypergeometric U
  // a = mu - kappa + 1/2
  // b = 2*mu + 1
  T half = T(0.5);
  T one = T(1);
  T two = T(2);

  T a = mu - kappa + half;
  T b = two * mu + one;

  // Compute the confluent hypergeometric function U(a, b, z)
  T U_val = confluent_hypergeometric_u(a, b, z);

  // Handle case where U returns infinity or NaN
  if (std::isinf(U_val) || std::isnan(U_val)) {
    return U_val;
  }

  // Compute exp(-z/2)
  T exp_factor = std::exp(-z / two);

  // Compute z^(mu + 1/2)
  // Using z^(mu + 1/2) = exp((mu + 1/2) * log(z))
  T mu_plus_half = mu + half;
  T z_power = std::exp(mu_plus_half * std::log(z));

  // Final result: exp(-z/2) * z^(mu + 1/2) * U(a, b, z)
  return exp_factor * z_power * U_val;
}

// Complex version of Whittaker W function
template <typename T>
c10::complex<T> whittaker_w(c10::complex<T> kappa, c10::complex<T> mu, c10::complex<T> z) {
  using detail::whit_w_epsilon;

  // Handle z = 0 case
  if (std::abs(z) < whit_w_epsilon<c10::complex<T>>()) {
    T mu_plus_half_real = mu.real() + T(0.5);
    T mu_real = mu.real();

    if (mu_plus_half_real > whit_w_epsilon<c10::complex<T>>()) {
      // z^(mu + 1/2) -> 0 as z -> 0 for Re(mu + 1/2) > 0
      if (mu_real <= T(0)) {
        // U is bounded at z=0
        return c10::complex<T>(T(0), T(0));
      } else {
        // U has a pole, analyze the combined behavior
        if (mu_plus_half_real > mu_real) {
          return c10::complex<T>(T(0), T(0));
        } else {
          return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
      }
    } else if (std::abs(mu_plus_half_real) < whit_w_epsilon<c10::complex<T>>() &&
               std::abs(mu.imag()) < whit_w_epsilon<c10::complex<T>>()) {
      // mu = -1/2
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    } else {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }
  }

  // Compute parameters
  c10::complex<T> half(T(0.5), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> a = mu - kappa + half;
  c10::complex<T> b = two * mu + one;

  // Compute confluent hypergeometric U
  c10::complex<T> U_val = confluent_hypergeometric_u(a, b, z);

  // Handle infinity or NaN
  if (std::isinf(U_val.real()) || std::isinf(U_val.imag()) ||
      std::isnan(U_val.real()) || std::isnan(U_val.imag())) {
    return U_val;
  }

  // Compute exp(-z/2)
  c10::complex<T> exp_factor = std::exp(-z / two);

  // Compute z^(mu + 1/2)
  c10::complex<T> mu_plus_half = mu + half;
  c10::complex<T> z_power = std::exp(mu_plus_half * std::log(z));

  // Final result
  return exp_factor * z_power * U_val;
}

} // namespace torchscience::kernel::special_functions
