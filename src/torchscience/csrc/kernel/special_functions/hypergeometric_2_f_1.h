#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct is_complex_type : std::false_type {};

template <typename T>
struct is_complex_type<std::complex<T>> : std::true_type {};

template <typename T>
struct is_complex_type<c10::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex_type<T>::value;

template <typename T>
struct real_type { using type = T; };

template <typename T>
struct real_type<std::complex<T>> { using type = T; };

template <typename T>
struct real_type<c10::complex<T>> { using type = T; };

template <typename T>
using real_type_t = typename real_type<T>::type;

template <typename T>
constexpr auto hyp2f1_epsilon() {
  using real_t = real_type_t<T>;
  if constexpr (std::is_same_v<real_t, float>) {
    return float(1e-7);
  } else if constexpr (std::is_same_v<real_t, double>) {
    return double(1e-15);
  } else {
    return float(1e-7);
  }
}

template <typename T>
bool hyp2f1_is_nonpositive_integer(T x) {
  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    auto re = static_cast<real_t>(x.real());
    auto im = static_cast<real_t>(x.imag());
    return std::abs(im) < hyp2f1_epsilon<T>() &&
           re <= real_t(0) &&
           std::abs(re - std::round(re)) < hyp2f1_epsilon<T>();
  } else {
    double xd = static_cast<double>(x);
    return xd <= 0.0 && std::abs(xd - std::round(xd)) < hyp2f1_epsilon<T>();
  }
}

template <typename T>
int hyp2f1_get_nonpositive_int(T x) {
  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    return static_cast<int>(std::round(static_cast<real_t>(x.real())));
  } else {
    return static_cast<int>(std::round(static_cast<double>(x)));
  }
}

template <typename T>
T hyp2f1_series(T a, T b, T c, T z, int max_iter = 500) {
  T sum = T(1);
  T term = T(1);

  for (int n = 0; n < max_iter; ++n) {
    T denom = (c + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp2f1_epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) * (b + T(n)) / denom * z;
    sum += term;

    if (std::abs(term) < hyp2f1_epsilon<T>() * std::abs(sum)) {
      return sum;
    }
  }

  return sum;
}

template <typename T>
T hyp2f1_near_one(T a, T b, T c, T z) {
  T z_transformed = z / (z - T(1));

  double zt_abs;
  if constexpr (is_complex_v<T>) {
    zt_abs = std::abs(z_transformed);
  } else {
    zt_abs = std::abs(static_cast<double>(z_transformed));
  }

  if (zt_abs < 0.5) {
    return std::pow(T(1) - z, -a) * hyp2f1_series(a, c - b, c, z_transformed);
  }

  if (zt_abs < 0.9) {
    return std::pow(T(1) - z, -b) * hyp2f1_series(b, c - a, c, z_transformed);
  }

  return hyp2f1_series(a, b, c, z, 2000);
}

template <typename T>
T hyp2f1_at_one(T a, T b, T c) {
  T s = c - a - b;

  double s_real;
  if constexpr (is_complex_v<T>) {
    s_real = static_cast<double>(s.real());
  } else {
    s_real = static_cast<double>(s);
  }

  if (s_real <= 0) {
    if constexpr (is_complex_v<T>) {
      using real_t = real_type_t<T>;
      return T(std::numeric_limits<real_t>::quiet_NaN(), real_t(0));
    } else {
      return std::numeric_limits<T>::quiet_NaN();
    }
  }

  if constexpr (is_complex_v<T>) {
    using real_t = real_type_t<T>;
    return T(std::numeric_limits<real_t>::quiet_NaN(), real_t(0));
  } else {
    T gamma_c = std::tgamma(c);
    T gamma_s = std::tgamma(s);
    T gamma_c_minus_a = std::tgamma(c - a);
    T gamma_c_minus_b = std::tgamma(c - b);

    return gamma_c * gamma_s / (gamma_c_minus_a * gamma_c_minus_b);
  }
}

template <typename T>
T hyp2f1_negative_z(T a, T b, T c, T z) {
  T w = z / (z - T(1));
  T prefactor = std::pow(T(1) - z, -a);

  double w_abs = std::abs(static_cast<double>(w));

  if (w_abs < 0.5) {
    return prefactor * hyp2f1_series(a, c - b, c, w);
  }

  return prefactor * hyp2f1_series(a, c - b, c, w, 1000);
}

template <typename T>
bool hyp2f1_on_unit_circle(T z) {
  double abs_z = std::abs(z);
  return std::abs(abs_z - 1.0) < 1e-10;
}

template <typename T>
bool hyp2f1_series_diverges_on_unit_circle(T a, T b, T c, T z) {
  if (!hyp2f1_on_unit_circle(z)) {
    return false;
  }

  double z_real, z_imag_abs;
  if constexpr (is_complex_v<T>) {
    z_real = static_cast<double>(z.real());
    z_imag_abs = std::abs(static_cast<double>(z.imag()));
  } else {
    z_real = static_cast<double>(z);
    z_imag_abs = 0.0;
  }

  if (std::abs(z_real - 1.0) < 1e-10 && z_imag_abs < 1e-10) {
    return false;
  }

  T s = c - a - b;
  double s_real;
  if constexpr (is_complex_v<T>) {
    s_real = static_cast<double>(s.real());
  } else {
    s_real = static_cast<double>(s);
  }

  return s_real <= -1.0;
}

} // namespace detail

template <typename T>
T hypergeometric_2_f_1(T a, T b, T c, T z) {
  using detail::hyp2f1_epsilon;
  using detail::hyp2f1_is_nonpositive_integer;
  using detail::hyp2f1_get_nonpositive_int;
  using detail::hyp2f1_series;
  using detail::hyp2f1_near_one;
  using detail::hyp2f1_at_one;
  using detail::hyp2f1_negative_z;
  using detail::hyp2f1_series_diverges_on_unit_circle;
  using detail::is_complex_v;

  if (std::abs(z) < hyp2f1_epsilon<T>()) {
    return T(1);
  }

  if (std::abs(a) < hyp2f1_epsilon<T>() || std::abs(b) < hyp2f1_epsilon<T>()) {
    return T(1);
  }

  if (hyp2f1_is_nonpositive_integer(c)) {
    int c_int = hyp2f1_get_nonpositive_int(c);
    bool a_cancels = hyp2f1_is_nonpositive_integer(a) && hyp2f1_get_nonpositive_int(a) > c_int;
    bool b_cancels = hyp2f1_is_nonpositive_integer(b) && hyp2f1_get_nonpositive_int(b) > c_int;
    if (!a_cancels && !b_cancels) {
      return std::numeric_limits<T>::infinity();
    }
  }

  if (hyp2f1_series_diverges_on_unit_circle(a, b, c, z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (std::abs(c - b) < hyp2f1_epsilon<T>()) {
    return std::pow(T(1) - z, -a);
  }

  if (std::abs(c - a) < hyp2f1_epsilon<T>()) {
    return std::pow(T(1) - z, -b);
  }

  if (hyp2f1_is_nonpositive_integer(a)) {
    int n_terms = -hyp2f1_get_nonpositive_int(a) + 1;
    return hyp2f1_series(a, b, c, z, n_terms);
  }
  if (hyp2f1_is_nonpositive_integer(b)) {
    int n_terms = -hyp2f1_get_nonpositive_int(b) + 1;
    return hyp2f1_series(a, b, c, z, n_terms);
  }

  double z_real, z_imag_abs;
  if constexpr (is_complex_v<T>) {
    z_real = static_cast<double>(z.real());
    z_imag_abs = std::abs(static_cast<double>(z.imag()));
  } else {
    z_real = static_cast<double>(z);
    z_imag_abs = 0.0;
  }

  if (std::abs(z_real - 1.0) < 1e-10 && z_imag_abs < 1e-10) {
    return hyp2f1_at_one(a, b, c);
  }

  if constexpr (is_complex_v<T>) {
    using real_t = detail::real_type_t<T>;
    if (std::abs(z) < real_t(0.5)) {
      return hyp2f1_series(a, b, c, z);
    }

    if (std::abs(z) < real_t(1)) {
      return hyp2f1_near_one(a, b, c, z);
    }

    T w = z / (z - T(1));
    return std::pow(T(1) - z, -a) * hyp2f1_series(a, c - b, c, w, 2000);
  } else {
    double zd = static_cast<double>(z);

    if (zd < 0.0) {
      return hyp2f1_negative_z(a, b, c, z);
    }

    if (std::abs(zd) < 0.5) {
      return hyp2f1_series(a, b, c, z);
    }

    if (std::abs(zd) < 1.0) {
      return hyp2f1_near_one(a, b, c, z);
    }

    return std::numeric_limits<T>::quiet_NaN();
  }
}

} // namespace torchscience::kernel::special_functions
