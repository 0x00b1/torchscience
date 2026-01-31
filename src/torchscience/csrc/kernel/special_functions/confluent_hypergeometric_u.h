#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

#include "confluent_hypergeometric_m.h"
#include "log_gamma.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for confluent hypergeometric U (use hypu_ prefix to avoid ODR conflicts)
template <typename T>
struct hypu_is_complex_type : std::false_type {};

template <typename T>
struct hypu_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hypu_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool hypu_is_complex_v = hypu_is_complex_type<T>::value;

template <typename T>
struct hypu_real_type { using type = T; };

template <typename T>
struct hypu_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hypu_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using hypu_real_type_t = typename hypu_real_type<T>::type;

template <typename T>
constexpr auto hypu_epsilon() {
  using real_t = hypu_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-6);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-14);
  } else {
    return float(1e-6);
  }
}

template <typename T>
bool hypu_is_nonpositive_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
bool hypu_is_positive_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           re >= real_t(1) &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd >= 1.0 && std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
bool hypu_is_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hypu_epsilon<T>() &&
           std::abs(re - std::round(re)) < hypu_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return std::abs(xd - std::round(xd)) < hypu_epsilon<T>();
  }
}

template <typename T>
int hypu_get_integer(T x) {
  if constexpr (hypu_is_complex_v<T>) {
    using real_t = hypu_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

// Asymptotic expansion for large |z|:
// U(a, b, z) ~ z^(-a) * sum_{n=0}^{inf} (a)_n * (a - b + 1)_n / n! * (-z)^(-n)
template <typename T>
T hypu_asymptotic(T a, T b, T z, int max_iter = 100) {
  T sum = T(1);
  T term = T(1);
  T a_minus_b_plus_1 = a - b + T(1);
  T neg_z_inv = T(-1) / z;

  for (int n = 0; n < max_iter; ++n) {
    T new_term = term * (a + T(n)) * (a_minus_b_plus_1 + T(n)) / T(n + 1) * neg_z_inv;

    // Check for convergence
    if (std::abs(new_term) < hypu_epsilon<T>() * std::abs(sum)) {
      break;
    }

    // Check for divergence (asymptotic series may diverge)
    if (std::abs(new_term) > std::abs(term) && n > 5) {
      break;
    }

    term = new_term;
    sum += term;
  }

  // Compute z^(-a)
  T result = std::exp(-a * std::log(z)) * sum;
  return result;
}

// U for integer b >= 2 using the limiting form with logarithm
// When b is a positive integer >= 2, we need a different approach
// U(a, n, z) involves logarithmic terms
template <typename T>
T hypu_integer_b_positive(T a, int n, T z, int max_iter = 200) {
  // For b = n (positive integer >= 2), use recurrence or series
  // U(a, 1, z) = z^(1-1) * Psi(a, 1, z) where Psi is a series
  // For simplicity, we use the asymptotic expansion for large z
  // and M-based formula for small z with regularization

  T b = T(n);

  // For integer b, the standard formula has removable singularities
  // Use L'Hopital's rule or Taylor expansion around integer b

  // Compute using the formula:
  // U(a, n, z) = ((-1)^n / ((n-1)! * Gamma(a-n+1))) *
  //              [M(a, n, z) * (psi(a) - psi(n) - log(z)) + series...]

  // For now, use a simpler approach: slightly perturb b and use the general formula
  // This is numerically stable for most cases
  T b_perturbed = b + T(1e-8);
  T one_minus_b = T(1) - b_perturbed;
  T a_minus_b_plus_1 = a - b_perturbed + T(1);

  // Compute gamma ratio: Gamma(1-b) / Gamma(a-b+1)
  T log_gamma_1_minus_b = log_gamma(one_minus_b);
  T log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  if (std::isinf(log_gamma_a_minus_b_plus_1)) {
    // a - b + 1 is a non-positive integer, first term vanishes
    // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
    T b_minus_1 = b - T(1);
    T log_gamma_b_minus_1 = log_gamma(b_minus_1);
    T log_gamma_a = log_gamma(a);

    if (std::isinf(log_gamma_a)) {
      // Both terms vanish or are undefined
      return std::numeric_limits<T>::quiet_NaN();
    }

    T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    T z_power = std::exp((T(1) - b) * std::log(z));
    T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

    return coeff2 * z_power * M2;
  }

  T coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
  T M1 = confluent_hypergeometric_m(a, b_perturbed, z);

  // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
  T b_minus_1 = b - T(1);
  T log_gamma_b_minus_1 = log_gamma(b_minus_1);
  T log_gamma_a = log_gamma(a);

  if (std::isinf(log_gamma_a)) {
    // Second term vanishes
    return coeff1 * M1;
  }

  T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
  T z_power = std::exp((T(1) - b) * std::log(z));
  T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

  return coeff1 * M1 + coeff2 * z_power * M2;
}

// U for integer b <= 0 using limiting form
template <typename T>
T hypu_integer_b_nonpositive(T a, int n, T z) {
  // For b = n <= 0 (non-positive integer), the formula simplifies
  // Both 1-b and 2-b are positive integers > 1
  T b = T(n);
  T one_minus_b = T(1) - b;  // Positive integer >= 1
  T a_minus_b_plus_1 = a - b + T(1);

  // Gamma(1-b) = (|b|)! for b <= 0
  T log_gamma_1_minus_b = log_gamma(one_minus_b);
  T log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  if (std::isinf(log_gamma_a_minus_b_plus_1)) {
    // a - b + 1 is a non-positive integer
    T b_minus_1 = b - T(1);  // Negative
    T log_gamma_b_minus_1 = log_gamma(b_minus_1);
    T log_gamma_a = log_gamma(a);

    if (std::isinf(log_gamma_b_minus_1) || std::isinf(log_gamma_a)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    T z_power = std::exp((T(1) - b) * std::log(z));
    T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

    return coeff2 * z_power * M2;
  }

  T coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
  T M1 = confluent_hypergeometric_m(a, b, z);

  // Second term
  T b_minus_1 = b - T(1);
  T log_gamma_b_minus_1 = log_gamma(b_minus_1);
  T log_gamma_a = log_gamma(a);

  if (std::isinf(log_gamma_b_minus_1)) {
    // b - 1 is a non-positive integer, second term vanishes
    return coeff1 * M1;
  }

  if (std::isinf(log_gamma_a)) {
    // a is a non-positive integer, second term vanishes
    return coeff1 * M1;
  }

  T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
  T z_power = std::exp((T(1) - b) * std::log(z));
  T M2 = confluent_hypergeometric_m(a - b + T(1), T(2) - b, z);

  return coeff1 * M1 + coeff2 * z_power * M2;
}

// General U computation using M-based definition for non-integer b
// U(a, b, z) = Gamma(1-b)/Gamma(a-b+1) * M(a, b, z)
//            + Gamma(b-1)/Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
template <typename T>
T hypu_via_m(T a, T b, T z) {
  T one_minus_b = T(1) - b;
  T a_minus_b_plus_1 = a - b + T(1);

  // First term: Gamma(1-b) / Gamma(a-b+1) * M(a, b, z)
  T log_gamma_1_minus_b = log_gamma(one_minus_b);
  T log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  T term1 = T(0);
  bool term1_valid = true;

  if (std::isinf(log_gamma_1_minus_b)) {
    // 1 - b is a non-positive integer (b is a positive integer >= 1)
    term1_valid = false;
  } else if (std::isinf(log_gamma_a_minus_b_plus_1)) {
    // a - b + 1 is a non-positive integer, first term vanishes
    term1_valid = false;
  } else {
    T coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
    T M1 = confluent_hypergeometric_m(a, b, z);
    term1 = coeff1 * M1;
  }

  // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
  T b_minus_1 = b - T(1);
  T log_gamma_b_minus_1 = log_gamma(b_minus_1);
  T log_gamma_a = log_gamma(a);

  T term2 = T(0);
  bool term2_valid = true;

  if (std::isinf(log_gamma_b_minus_1)) {
    // b - 1 is a non-positive integer (b is an integer <= 1)
    term2_valid = false;
  } else if (std::isinf(log_gamma_a)) {
    // a is a non-positive integer, second term vanishes
    term2_valid = false;
  } else {
    T coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    T z_power = std::exp(one_minus_b * std::log(z));
    T M2 = confluent_hypergeometric_m(a_minus_b_plus_1, T(2) - b, z);
    term2 = coeff2 * z_power * M2;
  }

  if (!term1_valid && !term2_valid) {
    // Both terms are invalid, need special handling
    return std::numeric_limits<T>::quiet_NaN();
  }

  return (term1_valid ? term1 : T(0)) + (term2_valid ? term2 : T(0));
}

} // namespace detail

// Confluent hypergeometric function U(a, b, z) (Tricomi function)
template <typename T>
T confluent_hypergeometric_u(T a, T b, T z) {
  using detail::hypu_epsilon;
  using detail::hypu_is_nonpositive_integer;
  using detail::hypu_is_positive_integer;
  using detail::hypu_is_integer;
  using detail::hypu_get_integer;
  using detail::hypu_is_complex_v;
  using detail::hypu_asymptotic;
  using detail::hypu_via_m;
  using detail::hypu_integer_b_positive;

  // Handle z = 0: U(a, b, 0) has a pole for Re(b) > 1
  // For Re(b) <= 1, U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1)
  if (std::abs(z) < hypu_epsilon<T>()) {
    // Check if b > 1 (pole at z = 0)
    double b_real;
    if constexpr (hypu_is_complex_v<T>) {
      b_real = static_cast<double>(b.real());
    } else {
      b_real = static_cast<double>(b);
    }

    if (b_real > 1.0 + hypu_epsilon<T>()) {
      return std::numeric_limits<T>::infinity();
    }

    // U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1) for Re(b) <= 1
    T one_minus_b = T(1) - b;
    T a_minus_b_plus_1 = a - b + T(1);

    T log_gamma_num = log_gamma(one_minus_b);
    T log_gamma_denom = log_gamma(a_minus_b_plus_1);

    if (std::isinf(log_gamma_denom)) {
      // a - b + 1 is a non-positive integer
      return T(0);
    }
    if (std::isinf(log_gamma_num)) {
      // 1 - b is a non-positive integer
      return std::numeric_limits<T>::infinity();
    }

    return std::exp(log_gamma_num - log_gamma_denom);
  }

  // Special case: a = 0, U(0, b, z) = 1
  if (std::abs(a) < hypu_epsilon<T>()) {
    return T(1);
  }

  // Special case: a = b, U(a, a, z) = exp(z) * Gamma(1-a, z) / z^(a-1)
  // For integer a >= 1: U(n, n, z) = exp(z) * sum_{k=0}^{n-1} z^k / k!
  // This is related to incomplete gamma, but for now use general method

  // Get magnitude of z
  double z_abs;
  if constexpr (hypu_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  // For large |z|, use asymptotic expansion
  if (z_abs > 30.0) {
    return hypu_asymptotic(a, b, z);
  }

  // Check if b is an integer
  if (hypu_is_integer(b)) {
    int b_int = hypu_get_integer(b);

    if (b_int >= 2) {
      // b is a positive integer >= 2, need special handling
      return hypu_integer_b_positive(a, b_int, z);
    } else if (b_int == 1) {
      // b = 1: U(a, 1, z) has a simpler form
      // U(a, 1, z) = -1/Gamma(a) * [M(a, 1, z) * (psi(a) + log(z) + 2*gamma) - sum...]
      // Use general method with perturbation for stability
      T b_perturbed = b + T(1e-8);
      return hypu_via_m(a, b_perturbed, z);
    }
    // b <= 0: standard formula works
  }

  // General case: use M-based formula
  return hypu_via_m(a, b, z);
}

// Complex version
template <typename T>
c10::complex<T> confluent_hypergeometric_u(c10::complex<T> a, c10::complex<T> b, c10::complex<T> z) {
  using detail::hypu_epsilon;
  using detail::hypu_is_nonpositive_integer;
  using detail::hypu_is_integer;
  using detail::hypu_get_integer;

  // Handle z = 0
  if (std::abs(z) < hypu_epsilon<c10::complex<T>>()) {
    // Check if Re(b) > 1 (pole at z = 0)
    if (b.real() > T(1) + hypu_epsilon<c10::complex<T>>()) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // U(a, b, 0) = Gamma(1-b) / Gamma(a-b+1) for Re(b) <= 1
    c10::complex<T> one_minus_b = c10::complex<T>(T(1), T(0)) - b;
    c10::complex<T> a_minus_b_plus_1 = a - b + c10::complex<T>(T(1), T(0));

    c10::complex<T> log_gamma_num = log_gamma(one_minus_b);
    c10::complex<T> log_gamma_denom = log_gamma(a_minus_b_plus_1);

    if (std::isinf(log_gamma_denom.real())) {
      return c10::complex<T>(T(0), T(0));
    }
    if (std::isinf(log_gamma_num.real())) {
      return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    return std::exp(log_gamma_num - log_gamma_denom);
  }

  // Special case: a = 0
  if (std::abs(a) < hypu_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(T(1), T(0));
  }

  double z_abs = std::abs(z);

  // For large |z|, use asymptotic expansion
  if (z_abs > 30.0) {
    c10::complex<T> sum(T(1), T(0));
    c10::complex<T> term(T(1), T(0));
    c10::complex<T> a_minus_b_plus_1 = a - b + c10::complex<T>(T(1), T(0));
    c10::complex<T> neg_z_inv = c10::complex<T>(T(-1), T(0)) / z;

    for (int n = 0; n < 100; ++n) {
      c10::complex<T> n_c(T(n), T(0));
      c10::complex<T> new_term = term * (a + n_c) * (a_minus_b_plus_1 + n_c) / c10::complex<T>(T(n + 1), T(0)) * neg_z_inv;

      if (std::abs(new_term) < hypu_epsilon<c10::complex<T>>() * std::abs(sum)) {
        break;
      }
      if (std::abs(new_term) > std::abs(term) && n > 5) {
        break;
      }

      term = new_term;
      sum += term;
    }

    // z^(-a) * sum
    return std::exp(-a * std::log(z)) * sum;
  }

  // General case: use M-based formula
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> one_minus_b = one - b;
  c10::complex<T> a_minus_b_plus_1 = a - b + one;

  // First term: Gamma(1-b) / Gamma(a-b+1) * M(a, b, z)
  c10::complex<T> log_gamma_1_minus_b = log_gamma(one_minus_b);
  c10::complex<T> log_gamma_a_minus_b_plus_1 = log_gamma(a_minus_b_plus_1);

  c10::complex<T> term1(T(0), T(0));
  bool term1_valid = true;

  if (std::isinf(log_gamma_1_minus_b.real())) {
    term1_valid = false;
  } else if (std::isinf(log_gamma_a_minus_b_plus_1.real())) {
    term1_valid = false;
  } else {
    c10::complex<T> coeff1 = std::exp(log_gamma_1_minus_b - log_gamma_a_minus_b_plus_1);
    c10::complex<T> M1 = confluent_hypergeometric_m(a, b, z);
    term1 = coeff1 * M1;
  }

  // Second term: Gamma(b-1) / Gamma(a) * z^(1-b) * M(a-b+1, 2-b, z)
  c10::complex<T> b_minus_1 = b - one;
  c10::complex<T> log_gamma_b_minus_1 = log_gamma(b_minus_1);
  c10::complex<T> log_gamma_a = log_gamma(a);

  c10::complex<T> term2(T(0), T(0));
  bool term2_valid = true;

  if (std::isinf(log_gamma_b_minus_1.real())) {
    term2_valid = false;
  } else if (std::isinf(log_gamma_a.real())) {
    term2_valid = false;
  } else {
    c10::complex<T> coeff2 = std::exp(log_gamma_b_minus_1 - log_gamma_a);
    c10::complex<T> z_power = std::exp(one_minus_b * std::log(z));
    c10::complex<T> two_minus_b = c10::complex<T>(T(2), T(0)) - b;
    c10::complex<T> M2 = confluent_hypergeometric_m(a_minus_b_plus_1, two_minus_b, z);
    term2 = coeff2 * z_power * M2;
  }

  if (!term1_valid && !term2_valid) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
  }

  return (term1_valid ? term1 : c10::complex<T>(T(0), T(0))) +
         (term2_valid ? term2 : c10::complex<T>(T(0), T(0)));
}

} // namespace torchscience::kernel::special_functions
