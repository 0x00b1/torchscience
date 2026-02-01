#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Type traits for hypergeometric pFq (use pfq_ prefix to avoid ODR conflicts)
template <typename T>
struct pfq_is_complex_type : std::false_type {};

template <typename T>
struct pfq_is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct pfq_is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool pfq_is_complex_v = pfq_is_complex_type<T>::value;

template <typename T>
struct pfq_real_type { using type = T; };

template <typename T>
struct pfq_real_type<std::complex<T>> { using type = T; };

template <typename T>
struct pfq_real_type<c10::complex<T>> { using type = T; };

template <typename T>
using pfq_real_type_t = typename pfq_real_type<T>::type;

template <typename T>
constexpr auto pfq_epsilon() {
  using real_t = pfq_real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    return float(1e-7);
  }
}

template <typename T>
bool pfq_is_nonpositive_integer(T x) {
  if constexpr (pfq_is_complex_v<T>) {
    using real_t = pfq_real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < pfq_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < pfq_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < pfq_epsilon<T>();
  }
}

template <typename T>
int pfq_get_nonpositive_int(T x) {
  if constexpr (pfq_is_complex_v<T>) {
    using real_t = pfq_real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

// Series expansion for pFq with variable-length parameter arrays
// pFq(a[]; b[]; z) = sum_{n=0}^{inf} [prod_i(a[i])_n / prod_j(b[j])_n] * z^n / n!
template <typename T>
T pfq_series(const T* a, int p, const T* b, int q, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    // Compute numerator: product of (a[i] + n) for all i
    T numer = T(1);
    for (int i = 0; i < p; ++i) {
      numer *= (a[i] + T(n));
    }

    // Compute denominator: product of (b[j] + n) for all j, times (n+1)
    T denom = T(n + 1);
    for (int j = 0; j < q; ++j) {
      T bj_plus_n = b[j] + T(n);
      if (std::abs(bj_plus_n) < pfq_epsilon<T>()) {
        // b[j] + n is near zero, potential pole
        return std::numeric_limits<pfq_real_type_t<T>>::infinity();
      }
      denom *= bj_plus_n;
    }

    if (std::abs(denom) < pfq_epsilon<T>()) {
      break;
    }

    term *= numer * z / denom;
    sum += term;

    if (std::abs(term) < pfq_epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

// Check if any upper parameter is a non-positive integer (polynomial case)
template <typename T>
int pfq_find_terminating_a(const T* a, int p) {
  int min_neg = 0;  // The smallest (most negative) non-positive integer
  bool found = false;

  for (int i = 0; i < p; ++i) {
    if (pfq_is_nonpositive_integer(a[i])) {
      int val = pfq_get_nonpositive_int(a[i]);
      if (!found || val > min_neg) {
        min_neg = val;
        found = true;
      }
    }
  }

  return found ? -min_neg : -1;  // Return polynomial degree or -1 if not terminating
}

// Check if any lower parameter is a non-positive integer (pole)
template <typename T>
bool pfq_has_pole(const T* b, int q) {
  for (int j = 0; j < q; ++j) {
    if (pfq_is_nonpositive_integer(b[j])) {
      return true;
    }
  }
  return false;
}

// Polynomial evaluation for terminating series
template <typename T>
T pfq_polynomial(const T* a, int p, const T* b, int q, T z, int degree) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < degree; ++n) {
    T numer = T(1);
    for (int i = 0; i < p; ++i) {
      numer *= (a[i] + T(n));
    }

    T denom = T(n + 1);
    for (int j = 0; j < q; ++j) {
      denom *= (b[j] + T(n));
    }

    if (std::abs(denom) < pfq_epsilon<T>()) {
      break;
    }

    term *= numer * z / denom;
    sum += term;
  }

  return sum;
}

} // namespace detail

// Generalized hypergeometric function pFq(a[]; b[]; z)
//
// The generalized hypergeometric function pFq is defined as:
//   pFq(a1,...,ap; b1,...,bq; z) = sum_{n=0}^{inf} [prod_i(ai)_n / prod_j(bj)_n] * z^n / n!
//
// where (x)_n = x(x+1)(x+2)...(x+n-1) is the Pochhammer symbol.
//
// Convergence:
//   - p <= q: converges for all z (entire function)
//   - p = q + 1: converges for |z| < 1, conditionally on |z| = 1
//   - p > q + 1: diverges except for polynomial cases
//
// Parameters:
//   a: Array of upper parameters (length p)
//   p: Number of upper parameters
//   b: Array of lower parameters (length q, must not contain non-positive integers)
//   q: Number of lower parameters
//   z: Argument
//
template <typename T>
T hypergeometric_p_f_q(const T* a, int p, const T* b, int q, T z) {
  using detail::pfq_epsilon;
  using detail::pfq_is_complex_v;
  using detail::pfq_real_type_t;
  using detail::pfq_series;
  using detail::pfq_polynomial;
  using detail::pfq_find_terminating_a;
  using detail::pfq_has_pole;

  // Handle z = 0: pFq(a[]; b[]; 0) = 1
  if (std::abs(z) < pfq_epsilon<T>()) {
    return T(1);
  }

  // Check for poles (b[j] is a non-positive integer)
  if (pfq_has_pole(b, q)) {
    // Check if any a[i] is a more negative non-positive integer that would cancel
    int poly_degree = pfq_find_terminating_a(a, p);
    if (poly_degree >= 0) {
      // Series terminates, might avoid the pole
      // For simplicity, just evaluate the polynomial
      return pfq_polynomial(a, p, b, q, z, poly_degree);
    }
    return std::numeric_limits<pfq_real_type_t<T>>::infinity();
  }

  // Check for polynomial case (a[i] is a non-positive integer)
  int poly_degree = pfq_find_terminating_a(a, p);
  if (poly_degree >= 0) {
    return pfq_polynomial(a, p, b, q, z, poly_degree);
  }

  // Check convergence for p > q + 1
  if (p > q + 1) {
    // Series diverges for non-polynomial cases
    // Return NaN to indicate divergence
    return std::numeric_limits<pfq_real_type_t<T>>::quiet_NaN();
  }

  // Check convergence for p = q + 1 when |z| >= 1
  if (p == q + 1) {
    double z_abs;
    if constexpr (pfq_is_complex_v<T>) {
      z_abs = std::abs(z);
    } else {
      z_abs = std::abs(static_cast<double>(z));
    }

    if (z_abs > 1.0 + pfq_epsilon<T>()) {
      // Series diverges for |z| > 1 when p = q + 1
      return std::numeric_limits<pfq_real_type_t<T>>::quiet_NaN();
    }
  }

  // General case: use series expansion
  return pfq_series(a, p, b, q, z);
}

// Complex version
template <typename T>
c10::complex<T> hypergeometric_p_f_q(const c10::complex<T>* a, int p, const c10::complex<T>* b, int q, c10::complex<T> z) {
  using detail::pfq_epsilon;
  using detail::pfq_series;
  using detail::pfq_has_pole;

  // Handle z = 0
  if (std::abs(z) < pfq_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(T(1), T(0));
  }

  // Check for poles
  if (pfq_has_pole(b, q)) {
    return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
  }

  // Check convergence for p > q + 1
  if (p > q + 1) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
  }

  // Check convergence for p = q + 1 when |z| >= 1
  if (p == q + 1 && std::abs(z) > T(1) + pfq_epsilon<c10::complex<T>>()) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
  }

  // Use series expansion
  return pfq_series(a, p, b, q, z);
}

} // namespace torchscience::kernel::special_functions
