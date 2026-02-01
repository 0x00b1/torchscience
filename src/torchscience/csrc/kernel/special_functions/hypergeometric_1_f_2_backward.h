#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_1_f_2.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct Hyp1F2WithGrads {
  T value;
  T grad_a;
  T grad_b1;
  T grad_b2;
};

// Series with gradient accumulation for parameter derivatives
// 1F2(a; b1, b2; z) = sum_{n=0}^inf (a)_n / ((b1)_n (b2)_n) * z^n / n!
template <typename T>
Hyp1F2WithGrads<T> hyp1f2_series_with_grads(T a, T b1, T b2, T z, int max_iter = 500) {
  T sum = T(1);
  T da_sum = T(0);
  T db1_sum = T(0);
  T db2_sum = T(0);

  T term = T(1);
  T H_a = T(0);   // sum of 1/(a+k) for k=0..n-1
  T H_b1 = T(0);  // sum of 1/(b1+k) for k=0..n-1
  T H_b2 = T(0);  // sum of 1/(b2+k) for k=0..n-1

  for (int n = 0; n < max_iter; ++n) {
    if (n > 0) {
      H_a += T(1) / (a + T(n - 1));
      H_b1 += T(1) / (b1 + T(n - 1));
      H_b2 += T(1) / (b2 + T(n - 1));

      da_sum += term * H_a;
      db1_sum -= term * H_b1;
      db2_sum -= term * H_b2;
    }

    T denom = (b1 + T(n)) * (b2 + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp1f2_epsilon<T>()) {
      break;
    }
    term *= (a + T(n)) * z / denom;
    sum += term;

    if (std::abs(term) < hyp1f2_epsilon<T>() * std::abs(sum)) {
      break;
    }
  }

  return {sum, da_sum, db1_sum, db2_sum};
}

} // namespace detail

// Backward pass for hypergeometric 1F2 function
// Returns (grad_a, grad_b1, grad_b2, grad_z)
template <typename T>
std::tuple<T, T, T, T> hypergeometric_1_f_2_backward(T grad, T a, T b1, T b2, T z) {
  using detail::hyp1f2_epsilon;
  using detail::hyp1f2_is_complex_v;
  using detail::hyp1f2_is_nonpositive_integer;
  using detail::hyp1f2_real_type_t;
  using detail::hyp1f2_series_with_grads;

  // d/dz[1F2(a; b1, b2; z)] = (a / (b1 * b2)) * 1F2(a+1; b1+1, b2+1; z)
  T dfdz = (a / (b1 * b2)) * hypergeometric_1_f_2(a + T(1), b1 + T(1), b2 + T(1), z);

  double z_abs;
  if constexpr (hyp1f2_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  bool use_analytical = z_abs < 50.0 &&
                        !hyp1f2_is_nonpositive_integer(a) &&
                        !hyp1f2_is_nonpositive_integer(b1) &&
                        !hyp1f2_is_nonpositive_integer(b2);

  T dfda, dfdb1, dfdb2;

  if (use_analytical) {
    auto result = hyp1f2_series_with_grads(a, b1, b2, z);
    dfda = result.grad_a;
    dfdb1 = result.grad_b1;
    dfdb2 = result.grad_b2;
  } else {
    // Fallback to finite differences
    using real_t = hyp1f2_real_type_t<T>;
    real_t eps_real = std::sqrt(hyp1f2_epsilon<T>());
    T eps = T(eps_real);

    T f_a_plus = hypergeometric_1_f_2(a + eps, b1, b2, z);
    T f_a_minus = hypergeometric_1_f_2(a - eps, b1, b2, z);
    dfda = (f_a_plus - f_a_minus) / (T(2) * eps);

    T f_b1_plus = hypergeometric_1_f_2(a, b1 + eps, b2, z);
    T f_b1_minus = hypergeometric_1_f_2(a, b1 - eps, b2, z);
    dfdb1 = (f_b1_plus - f_b1_minus) / (T(2) * eps);

    T f_b2_plus = hypergeometric_1_f_2(a, b1, b2 + eps, z);
    T f_b2_minus = hypergeometric_1_f_2(a, b1, b2 - eps, z);
    dfdb2 = (f_b2_plus - f_b2_minus) / (T(2) * eps);
  }

  // For complex types, PyTorch expects grad * conj(derivative)
  if constexpr (hyp1f2_is_complex_v<T>) {
    return {
      grad * std::conj(dfda),
      grad * std::conj(dfdb1),
      grad * std::conj(dfdb2),
      grad * std::conj(dfdz)
    };
  } else {
    return {
      grad * dfda,
      grad * dfdb1,
      grad * dfdb2,
      grad * dfdz
    };
  }
}

} // namespace torchscience::kernel::special_functions
