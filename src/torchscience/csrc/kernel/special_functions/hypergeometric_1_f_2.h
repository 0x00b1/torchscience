#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for hypergeometric 1F2 (use hyp1f2_ prefix to avoid ODR conflicts)
template <typename T>
struct hyp1f2_is_complex_type : std::false_type {};

template <typename T>
struct hyp1f2_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct hyp1f2_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool hyp1f2_is_complex_v = hyp1f2_is_complex_type<T>::value;

template <typename T>
struct hyp1f2_real_type { using type = T; };

template <typename T>
struct hyp1f2_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct hyp1f2_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using hyp1f2_real_type_t = typename hyp1f2_real_type<T>::type;

template <typename T>
constexpr auto hyp1f2_epsilon() {
  using real_t = hyp1f2_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    return float(1e-7);
  }
}

template <typename T>
bool hyp1f2_is_nonpositive_integer(T x) {
  if constexpr (hyp1f2_is_complex_v<T>) {
    using real_t = hyp1f2_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hyp1f2_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hyp1f2_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hyp1f2_epsilon<T>();
  }
}

template <typename T>
int hyp1f2_get_nonpositive_int(T x) {
  if constexpr (hyp1f2_is_complex_v<T>) {
    using real_t = hyp1f2_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

// Series expansion: 1F2(a; b1, b2; z) = sum_{n=0}^{inf} (a)_n / ((b1)_n (b2)_n) * z^n / n!
template <typename T>
T hyp1f2_series(T a, T b1, T b2, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    T denom = (b1 + T(n)) * (b2 + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp1f2_epsilon<T>()) {
      // b1 + n or b2 + n is near zero, potential pole
      break;
    }
    term *= (a + T(n)) * z / denom;
    sum += term;

    if (std::abs(term) < hyp1f2_epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

// For a being a non-positive integer -m, the series terminates (polynomial)
template <typename T>
T hyp1f2_polynomial(int m, T b1, T b2, T z) {
  // m is the absolute value of a (a = -m where m >= 0)
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < m; ++n) {
    T a_term = T(-m + n);  // (a)_n where a = -m
    T denom = (b1 + T(n)) * (b2 + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp1f2_epsilon<T>()) {
      break;
    }
    term *= a_term * z / denom;
    sum += term;
  }

  return sum;
}

} // namespace detail

// Hypergeometric function 1F2(a; b1, b2; z)
//
// The generalized hypergeometric function 1F2 is defined as:
//   1F2(a; b1, b2; z) = sum_{n=0}^{inf} (a)_n / ((b1)_n (b2)_n) * z^n / n!
//
// where (x)_n = x(x+1)(x+2)...(x+n-1) is the Pochhammer symbol.
//
// Properties:
//   - 1F2(a; b1, b2; 0) = 1
//   - Poles when b1 or b2 is a non-positive integer
//   - When a is a non-positive integer -m, the series terminates (polynomial of degree m)
//   - Related to Bessel functions and other special functions
//
// Parameters:
//   a: Upper parameter
//   b1: First lower parameter (must not be a non-positive integer)
//   b2: Second lower parameter (must not be a non-positive integer)
//   z: Argument
//
template <typename T>
T hypergeometric_1_f_2(T a, T b1, T b2, T z) {
  using detail::hyp1f2_epsilon;
  using detail::hyp1f2_is_complex_v;
  using detail::hyp1f2_is_nonpositive_integer;
  using detail::hyp1f2_get_nonpositive_int;
  using detail::hyp1f2_series;
  using detail::hyp1f2_polynomial;

  // Handle z = 0: 1F2(a; b1, b2; 0) = 1
  if (std::abs(z) < hyp1f2_epsilon<T>()) {
    return T(1);
  }

  // Handle b1 or b2 being a non-positive integer (pole)
  if (hyp1f2_is_nonpositive_integer(b1) || hyp1f2_is_nonpositive_integer(b2)) {
    // Check if a is also a non-positive integer that would cancel the pole
    if (hyp1f2_is_nonpositive_integer(a)) {
      int a_int = hyp1f2_get_nonpositive_int(a);
      int b1_int = hyp1f2_is_nonpositive_integer(b1) ? hyp1f2_get_nonpositive_int(b1) : -1000;
      int b2_int = hyp1f2_is_nonpositive_integer(b2) ? hyp1f2_get_nonpositive_int(b2) : -1000;

      // If a >= b1 or a >= b2 (both non-positive), the series terminates before the pole
      if (a_int >= b1_int || a_int >= b2_int) {
        return hyp1f2_polynomial(-a_int, b1, b2, z);
      }
    }
    return std::numeric_limits<T>::infinity();
  }

  // Handle a being a non-positive integer (polynomial case)
  if (hyp1f2_is_nonpositive_integer(a)) {
    int m = -hyp1f2_get_nonpositive_int(a);
    return hyp1f2_polynomial(m, b1, b2, z);
  }

  // General case: use series expansion
  return hyp1f2_series(a, b1, b2, z);
}

// Complex version
template <typename T>
c10::complex<T> hypergeometric_1_f_2(c10::complex<T> a, c10::complex<T> b1, c10::complex<T> b2, c10::complex<T> z) {
  using detail::hyp1f2_epsilon;
  using detail::hyp1f2_is_nonpositive_integer;
  using detail::hyp1f2_series;

  // Handle z = 0
  if (std::abs(z) < hyp1f2_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(T(1), T(0));
  }

  // Handle b1 or b2 being a non-positive integer (pole)
  if (hyp1f2_is_nonpositive_integer(b1) || hyp1f2_is_nonpositive_integer(b2)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // Use series expansion
  return hyp1f2_series(a, b1, b2, z);
}

} // namespace torchscience::kernel::special_functions
