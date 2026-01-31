#pragma once

#include <cmath>
#include <tuple>

#include "confluent_hypergeometric_m.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct Hyp1F1WithGrads {
  T value;
  T grad_a;
  T grad_b;
};

// Series with gradient accumulation for parameter derivatives
template <typename T>
Hyp1F1WithGrads<T> hyp1f1_series_with_grads(T a, T b, T z, int max_iter = 500) {
  T sum = T(1);
  T da_sum = T(0);
  T db_sum = T(0);

  T term = T(1);
  T H_a = T(0);  // sum of 1/(a+k) for k=0..n-1
  T H_b = T(0);  // sum of 1/(b+k) for k=0..n-1

  for (int n = 0; n < max_iter; ++n) {
    if (n > 0) {
      H_a += T(1) / (a + T(n - 1));
      H_b += T(1) / (b + T(n - 1));

      da_sum += term * H_a;
      db_sum -= term * H_b;
    }

    T denom = (b + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp1f1_epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) / denom * z;
    sum += term;

    if (std::abs(term) < hyp1f1_epsilon<T>() * std::abs(sum)) {
      break;
    }
  }

  return {sum, da_sum, db_sum};
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> confluent_hypergeometric_m_backward(T grad, T a, T b, T z) {
  using detail::hyp1f1_epsilon;
  using detail::hyp1f1_is_nonpositive_integer;
  using detail::hyp1f1_series_with_grads;
  using detail::hyp1f1_is_complex_v;
  using detail::hyp1f1_real_type_t;

  // d/dz M(a,b,z) = (a/b) * M(a+1, b+1, z)
  T dfdz = (a / b) * confluent_hypergeometric_m(a + T(1), b + T(1), z);

  double z_abs;
  if constexpr (hyp1f1_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  bool use_analytical = z_abs < 30.0 &&
                        !hyp1f1_is_nonpositive_integer(a) &&
                        !hyp1f1_is_nonpositive_integer(b);

  T dfda, dfdb;

  if (use_analytical) {
    auto result = hyp1f1_series_with_grads(a, b, z);
    dfda = result.grad_a;
    dfdb = result.grad_b;
  } else {
    // Fallback to finite differences
    using real_t = hyp1f1_real_type_t<T>;
    real_t eps_real = std::sqrt(hyp1f1_epsilon<T>());
    T eps = T(eps_real);

    T f_a_plus = confluent_hypergeometric_m(a + eps, b, z);
    T f_a_minus = confluent_hypergeometric_m(a - eps, b, z);
    dfda = (f_a_plus - f_a_minus) / (T(2) * eps);

    T f_b_plus = confluent_hypergeometric_m(a, b + eps, z);
    T f_b_minus = confluent_hypergeometric_m(a, b - eps, z);
    dfdb = (f_b_plus - f_b_minus) / (T(2) * eps);
  }

  // For complex types, PyTorch expects grad * conj(derivative)
  if constexpr (hyp1f1_is_complex_v<T>) {
    return {
      grad * std::conj(dfda),
      grad * std::conj(dfdb),
      grad * std::conj(dfdz)
    };
  } else {
    return {grad * dfda, grad * dfdb, grad * dfdz};
  }
}

} // namespace torchscience::kernel::special_functions
