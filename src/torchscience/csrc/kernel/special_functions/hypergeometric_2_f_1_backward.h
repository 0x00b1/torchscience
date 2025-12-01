#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct Hyp2F1WithGrads {
  T value;
  T grad_a;
  T grad_b;
  T grad_c;
};

template <typename T>
Hyp2F1WithGrads<T> hyp2f1_series_with_grads(T a, T b, T c, T z, int max_iter = 500) {
  T sum = T(1);
  T da_sum = T(0);
  T db_sum = T(0);
  T dc_sum = T(0);

  T term = T(1);
  T H_a = T(0);
  T H_b = T(0);
  T H_c = T(0);

  for (int n = 0; n < max_iter; ++n) {
    if (n > 0) {
      H_a += T(1) / (a + T(n - 1));
      H_b += T(1) / (b + T(n - 1));
      H_c += T(1) / (c + T(n - 1));

      da_sum += term * H_a;
      db_sum += term * H_b;
      dc_sum -= term * H_c;
    }

    T denom = (c + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp2f1_epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) * (b + T(n)) / denom * z;
    sum += term;

    if (std::abs(term) < hyp2f1_epsilon<T>() * std::abs(sum)) {
      break;
    }
  }

  return {sum, da_sum, db_sum, dc_sum};
}

} // namespace detail

template <typename T>
std::tuple<T, T, T, T> hypergeometric_2_f_1_backward(T grad, T a, T b, T c, T z) {
  using detail::hyp2f1_epsilon;
  using detail::hyp2f1_is_nonpositive_integer;
  using detail::hyp2f1_series_with_grads;
  using detail::is_complex_v;

  // d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
  T dfdz = (a * b / c) * hypergeometric_2_f_1(a + T(1), b + T(1), c + T(1), z);

  double z_abs;
  if constexpr (is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }
  bool use_analytical = z_abs < 0.5 &&
                        !hyp2f1_is_nonpositive_integer(a) &&
                        !hyp2f1_is_nonpositive_integer(b);

  T dfda, dfdb, dfdc;

  if (use_analytical) {
    auto result = hyp2f1_series_with_grads(a, b, c, z);
    dfda = result.grad_a;
    dfdb = result.grad_b;
    dfdc = result.grad_c;
  } else {
    // Fallback to finite differences
    using real_t = detail::real_type_t<T>;
    real_t eps_real = std::sqrt(hyp2f1_epsilon<T>());
    T eps = T(eps_real);

    T f_a_plus = hypergeometric_2_f_1(a + eps, b, c, z);
    T f_a_minus = hypergeometric_2_f_1(a - eps, b, c, z);
    dfda = (f_a_plus - f_a_minus) / (T(2) * eps);

    T f_b_plus = hypergeometric_2_f_1(a, b + eps, c, z);
    T f_b_minus = hypergeometric_2_f_1(a, b - eps, c, z);
    dfdb = (f_b_plus - f_b_minus) / (T(2) * eps);

    T f_c_plus = hypergeometric_2_f_1(a, b, c + eps, z);
    T f_c_minus = hypergeometric_2_f_1(a, b, c - eps, z);
    dfdc = (f_c_plus - f_c_minus) / (T(2) * eps);
  }

  // For complex types, PyTorch expects grad * conj(derivative) for holomorphic functions
  // This handles both real and imaginary parts of the output correctly
  if constexpr (is_complex_v<T>) {
    return {
      grad * std::conj(dfda),
      grad * std::conj(dfdb),
      grad * std::conj(dfdc),
      grad * std::conj(dfdz)
    };
  } else {
    return {grad * dfda, grad * dfdb, grad * dfdc, grad * dfdz};
  }
}

} // namespace torchscience::kernel::special_functions
