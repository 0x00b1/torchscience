#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for hypergeometric 0F1 (use hyp0f1_ prefix to avoid ODR conflicts)
template <typename T>
struct hyp0f1_is_complex_type : std::false_type {};

template <typename T>
struct hyp0f1_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hyp0f1_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool hyp0f1_is_complex_v = hyp0f1_is_complex_type<T>::value;

template <typename T>
struct hyp0f1_real_type { using type = T; };

template <typename T>
struct hyp0f1_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hyp0f1_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using hyp0f1_real_type_t = typename hyp0f1_real_type<T>::type;

template <typename T>
constexpr auto hyp0f1_epsilon() {
  using real_t = hyp0f1_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    return float(1e-7);
  }
}

template <typename T>
bool hyp0f1_is_nonpositive_integer(T x) {
  if constexpr (hyp0f1_is_complex_v<T>) {
    using real_t = hyp0f1_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hyp0f1_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hyp0f1_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hyp0f1_epsilon<T>();
  }
}

// Series expansion: 0F1(;b;z) = sum_{n=0}^{inf} z^n / ((b)_n * n!)
// where (b)_n = b(b+1)...(b+n-1) is the Pochhammer symbol
template <typename T>
T hyp0f1_series(T b, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    T denom = (b + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp0f1_epsilon<T>()) {
      // b + n is near zero, potential pole
      break;
    }
    term *= z / denom;
    sum += term;

    if (std::abs(term) < hyp0f1_epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

// Asymptotic expansion for large |z|
// For large |z|, 0F1(;b;z) ~ Gamma(b) * (
//   exp(2*sqrt(z)) * (2*sqrt(z))^(1/2-b) / (2*sqrt(pi))
//   + exp(-2*sqrt(z)) * (-2*sqrt(z))^(1/2-b) / (2*sqrt(pi))
// )
// This is related to Bessel function asymptotics
template <typename T>
T hyp0f1_asymptotic(T b, T z, int max_terms = 20) {
  // For real positive z, use relation to Bessel I:
  // 0F1(;b;z) = Gamma(b) * (sqrt(z))^(1-b) * I_{b-1}(2*sqrt(z))
  //
  // For real negative z, use relation to Bessel J:
  // 0F1(;b;-z) = Gamma(b) * (sqrt(z))^(1-b) * J_{b-1}(2*sqrt(z))
  //
  // For now, use the series for all z (works well for moderate z)
  // The asymptotic expansion requires careful handling of branch cuts
  return hyp0f1_series(b, z, 1000);
}

} // namespace detail

// Hypergeometric function 0F1(;b;z)
//
// The confluent hypergeometric limit function is defined as:
//   0F1(;b;z) = sum_{n=0}^{inf} z^n / ((b)_n * n!)
//
// where (b)_n = b(b+1)(b+2)...(b+n-1) is the Pochhammer symbol (rising factorial).
//
// Properties:
//   - 0F1(;b;0) = 1
//   - Poles when b is a non-positive integer
//   - Related to Bessel functions:
//     J_v(z) = (z/2)^v / Gamma(v+1) * 0F1(;v+1;-z^2/4)
//     I_v(z) = (z/2)^v / Gamma(v+1) * 0F1(;v+1;z^2/4)
//
// Parameters:
//   b: Parameter (must not be a non-positive integer)
//   z: Argument
//
template <typename T>
T hypergeometric_0_f_1(T b, T z) {
  using detail::hyp0f1_epsilon;
  using detail::hyp0f1_is_complex_v;
  using detail::hyp0f1_is_nonpositive_integer;
  using detail::hyp0f1_series;
  using detail::hyp0f1_asymptotic;

  // Handle z = 0: 0F1(;b;0) = 1
  if (std::abs(z) < hyp0f1_epsilon<T>()) {
    return T(1);
  }

  // Handle b being a non-positive integer (pole)
  if (hyp0f1_is_nonpositive_integer(b)) {
    return std::numeric_limits<T>::infinity();
  }

  // Get magnitude of z
  double z_abs;
  if constexpr (hyp0f1_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  // For large |z|, could use asymptotic expansion
  // For now, series works reasonably well for moderate z
  if (z_abs > 100.0) {
    return hyp0f1_asymptotic(b, z);
  }

  // General case: use series expansion
  return hyp0f1_series(b, z);
}

// Complex version
template <typename T>
c10::complex<T> hypergeometric_0_f_1(c10::complex<T> b, c10::complex<T> z) {
  using detail::hyp0f1_epsilon;
  using detail::hyp0f1_is_nonpositive_integer;
  using detail::hyp0f1_series;

  // Handle z = 0
  if (std::abs(z) < hyp0f1_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(T(1), T(0));
  }

  // Handle b being a non-positive integer (pole)
  if (hyp0f1_is_nonpositive_integer(b)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // Use series expansion
  return hyp0f1_series(b, z);
}

} // namespace torchscience::kernel::special_functions
