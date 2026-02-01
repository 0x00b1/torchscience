#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_0_f_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
struct Hyp0F1WithGrad {
  T value;
  T grad_b;
};

// Series with gradient accumulation for parameter b derivative
// 0F1(;b;z) = sum_{n=0}^inf z^n / ((b)_n * n!)
// d/db[0F1(;b;z)] = -sum_{n=1}^inf z^n / ((b)_n * n!) * H_b(n)
// where H_b(n) = sum_{k=0}^{n-1} 1/(b+k)
template <typename T>
Hyp0F1WithGrad<T> hyp0f1_series_with_grad(T b, T z, int max_iter = 500) {
  T sum = T(1);
  T db_sum = T(0);

  T term = T(1);
  T H_b = T(0);  // sum of 1/(b+k) for k=0..n-1

  for (int n = 0; n < max_iter; ++n) {
    if (n > 0) {
      H_b += T(1) / (b + T(n - 1));
      db_sum -= term * H_b;
    }

    T denom = (b + T(n)) * T(n + 1);
    if (std::abs(denom) < hyp0f1_epsilon<T>()) {
      break;
    }
    term *= z / denom;
    sum += term;

    if (std::abs(term) < hyp0f1_epsilon<T>() * std::abs(sum)) {
      break;
    }
  }

  return {sum, db_sum};
}

} // namespace detail

// Backward pass for hypergeometric 0F1 function
// Returns (grad_b, grad_z)
//
// Derivatives:
//   d/dz[0F1(;b;z)] = 0F1(;b+1;z) / b
//   d/db requires series with gradient accumulation or finite differences
template <typename T>
std::tuple<T, T> hypergeometric_0_f_1_backward(T grad, T b, T z) {
  using detail::hyp0f1_epsilon;
  using detail::hyp0f1_is_complex_v;
  using detail::hyp0f1_is_nonpositive_integer;
  using detail::hyp0f1_real_type_t;
  using detail::hyp0f1_series_with_grad;

  // d/dz[0F1(;b;z)] = 0F1(;b+1;z) / b
  T dfdz = hypergeometric_0_f_1(b + T(1), z) / b;

  double z_abs;
  if constexpr (hyp0f1_is_complex_v<T>) {
    z_abs = std::abs(z);
  } else {
    z_abs = std::abs(static_cast<double>(z));
  }

  bool use_analytical = z_abs < 50.0 && !hyp0f1_is_nonpositive_integer(b);

  T dfdb;

  if (use_analytical) {
    auto result = hyp0f1_series_with_grad(b, z);
    dfdb = result.grad_b;
  } else {
    // Fallback to finite differences
    using real_t = hyp0f1_real_type_t<T>;
    real_t eps_real = std::sqrt(hyp0f1_epsilon<T>());
    T eps = T(eps_real);

    T f_b_plus = hypergeometric_0_f_1(b + eps, z);
    T f_b_minus = hypergeometric_0_f_1(b - eps, z);
    dfdb = (f_b_plus - f_b_minus) / (T(2) * eps);
  }

  // For complex types, PyTorch expects grad * conj(derivative)
  if constexpr (hyp0f1_is_complex_v<T>) {
    return {
      grad * std::conj(dfdb),
      grad * std::conj(dfdz)
    };
  } else {
    return {
      grad * dfdb,
      grad * dfdz
    };
  }
}

} // namespace torchscience::kernel::special_functions
