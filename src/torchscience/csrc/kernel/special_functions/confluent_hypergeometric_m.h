#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for confluent hypergeometric (use hyp1f1_ prefix to avoid conflicts)
template <typename T>
struct hyp1f1_is_complex_type : std::false_type {};

template <typename T>
struct hyp1f1_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hyp1f1_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool hyp1f1_is_complex_v = hyp1f1_is_complex_type<T>::value;

template <typename T>
struct hyp1f1_real_type { using type = T; };

template <typename T>
struct hyp1f1_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hyp1f1_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using hyp1f1_real_type_t = typename hyp1f1_real_type<T>::type;

template <typename T>
constexpr auto hyp1f1_epsilon() {
  using real_t = hyp1f1_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    return float(1e-7);
  }
}

template <typename T>
bool hyp1f1_is_nonpositive_integer(T x) {
  if constexpr (hyp1f1_is_complex_v<T>) {
    using real_t = hyp1f1_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hyp1f1_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hyp1f1_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hyp1f1_epsilon<T>();
  }
}

template <typename T>
int hyp1f1_get_nonpositive_int(T x) {
  if constexpr (hyp1f1_is_complex_v<T>) {
    using real_t = hyp1f1_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

// Series expansion: M(a, b, z) = sum_{n=0}^{inf} (a)_n / (b)_n * z^n / n!
template <typename T>
T hyp1f1_series(T a, T b, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    T denom = (b + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp1f1_epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) / denom * z;
    sum += term;

    if (std::abs(term) < hyp1f1_epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

} // namespace detail

// Confluent hypergeometric function M(a, b, z) = 1F1(a; b; z)
template <typename T>
T confluent_hypergeometric_m(T a, T b, T z) {
  using detail::hyp1f1_epsilon;
  using detail::hyp1f1_is_nonpositive_integer;
  using detail::hyp1f1_get_nonpositive_int;
  using detail::hyp1f1_series;
  using detail::hyp1f1_is_complex_v;

  // M(a, b, 0) = 1
  if (std::abs(z) < hyp1f1_epsilon<T>()) {
    return T(1);
  }

  // M(0, b, z) = 1
  if (std::abs(a) < hyp1f1_epsilon<T>()) {
    return T(1);
  }

  // Pole when b is a non-positive integer (unless a cancels it)
  if (hyp1f1_is_nonpositive_integer(b)) {
    int b_int = hyp1f1_get_nonpositive_int(b);
    bool a_cancels = hyp1f1_is_nonpositive_integer(a) &&
                     hyp1f1_get_nonpositive_int(a) > b_int;
    if (!a_cancels) {
      return std::numeric_limits<T>::infinity();
    }
  }

  // M(a, a, z) = exp(z)
  if (std::abs(a - b) < hyp1f1_epsilon<T>()) {
    return std::exp(z);
  }

  // When a is a non-positive integer, series terminates (polynomial)
  if (hyp1f1_is_nonpositive_integer(a)) {
    int n_terms = -hyp1f1_get_nonpositive_int(a) + 1;
    return hyp1f1_series(a, b, z, n_terms);
  }

  double z_abs;
  if constexpr (hyp1f1_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  // For small |z|, use direct series
  if (z_abs < 50.0) {
    return hyp1f1_series(a, b, z);
  }

  // For large |z| with Re(z) < 0, use Kummer transformation:
  // M(a, b, z) = exp(z) * M(b-a, b, -z)
  double z_real;
  if constexpr (hyp1f1_is_complex_v<T>) {
    z_real = static_cast<double>(z.real());
  } else {
    z_real = static_cast<double>(z);
  }

  if (z_real < 0) {
    return std::exp(z) * hyp1f1_series(b - a, b, -z);
  }

  // For large positive z, series may converge slowly
  // Use more iterations
  return hyp1f1_series(a, b, z, 2000);
}

} // namespace torchscience::kernel::special_functions
