#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_1_f_2.h"
#include "hypergeometric_1_f_2_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for hypergeometric 1F2 function
// Returns (grad_grad, grad_a, grad_b1, grad_b2, grad_z)
template <typename T>
std::tuple<T, T, T, T, T> hypergeometric_1_f_2_backward_backward(
    T gg_a, T gg_b1, T gg_b2, T gg_z,
    T grad, T a, T b1, T b2, T z) {
  using detail::hyp1f2_epsilon;
  using detail::hyp1f2_is_complex_v;
  using detail::hyp1f2_real_type_t;

  // For second-order derivatives, use finite differences
  using real_t = hyp1f2_real_type_t<T>;
  real_t eps_real = std::cbrt(hyp1f2_epsilon<T>());
  T eps = T(eps_real);

  // Get first-order derivatives at current point
  auto [da, db1, db2, dz] = hypergeometric_1_f_2_backward(T(1), a, b1, b2, z);

  // Compute second derivatives via finite differences
  // d²f/da²
  auto [da_plus, db1_a_plus, db2_a_plus, dz_a_plus] =
      hypergeometric_1_f_2_backward(T(1), a + eps, b1, b2, z);
  auto [da_minus, db1_a_minus, db2_a_minus, dz_a_minus] =
      hypergeometric_1_f_2_backward(T(1), a - eps, b1, b2, z);
  T d2f_da2 = (da_plus - da_minus) / (T(2) * eps);
  T d2f_db1_da = (db1_a_plus - db1_a_minus) / (T(2) * eps);
  T d2f_db2_da = (db2_a_plus - db2_a_minus) / (T(2) * eps);
  T d2f_dz_da = (dz_a_plus - dz_a_minus) / (T(2) * eps);

  // d²f/db1²
  auto [da_b1_plus, db1_plus, db2_b1_plus, dz_b1_plus] =
      hypergeometric_1_f_2_backward(T(1), a, b1 + eps, b2, z);
  auto [da_b1_minus, db1_minus, db2_b1_minus, dz_b1_minus] =
      hypergeometric_1_f_2_backward(T(1), a, b1 - eps, b2, z);
  T d2f_db1_db1 = (db1_plus - db1_minus) / (T(2) * eps);
  T d2f_db2_db1 = (db2_b1_plus - db2_b1_minus) / (T(2) * eps);
  T d2f_dz_db1 = (dz_b1_plus - dz_b1_minus) / (T(2) * eps);

  // d²f/db2²
  auto [da_b2_plus, db1_b2_plus, db2_plus, dz_b2_plus] =
      hypergeometric_1_f_2_backward(T(1), a, b1, b2 + eps, z);
  auto [da_b2_minus, db1_b2_minus, db2_minus, dz_b2_minus] =
      hypergeometric_1_f_2_backward(T(1), a, b1, b2 - eps, z);
  T d2f_db2_db2 = (db2_plus - db2_minus) / (T(2) * eps);
  T d2f_dz_db2 = (dz_b2_plus - dz_b2_minus) / (T(2) * eps);

  // d²f/dz²
  auto [da_z_plus, db1_z_plus, db2_z_plus, dz_plus] =
      hypergeometric_1_f_2_backward(T(1), a, b1, b2, z + eps);
  auto [da_z_minus, db1_z_minus, db2_z_minus, dz_minus] =
      hypergeometric_1_f_2_backward(T(1), a, b1, b2, z - eps);
  T d2f_dz2 = (dz_plus - dz_minus) / (T(2) * eps);

  // grad_grad = sum of gg_i * df/di
  T grad_grad = gg_a * da + gg_b1 * db1 + gg_b2 * db2 + gg_z * dz;

  // grad_a = grad * (gg_a * d²f/da² + gg_b1 * d²f/db1da + gg_b2 * d²f/db2da + gg_z * d²f/dzda)
  T grad_a = grad * (gg_a * d2f_da2 + gg_b1 * d2f_db1_da + gg_b2 * d2f_db2_da + gg_z * d2f_dz_da);

  // grad_b1 = grad * (gg_a * d²f/dadb1 + gg_b1 * d²f/db1² + gg_b2 * d²f/db2db1 + gg_z * d²f/dzdb1)
  T grad_b1 = grad * (gg_a * d2f_db1_da + gg_b1 * d2f_db1_db1 + gg_b2 * d2f_db2_db1 + gg_z * d2f_dz_db1);

  // grad_b2 = grad * (gg_a * d²f/dadb2 + gg_b1 * d²f/db1db2 + gg_b2 * d²f/db2² + gg_z * d²f/dzdb2)
  T grad_b2 = grad * (gg_a * d2f_db2_da + gg_b1 * d2f_db2_db1 + gg_b2 * d2f_db2_db2 + gg_z * d2f_dz_db2);

  // grad_z = grad * (gg_a * d²f/dadz + gg_b1 * d²f/db1dz + gg_b2 * d²f/db2dz + gg_z * d²f/dz²)
  T grad_z = grad * (gg_a * d2f_dz_da + gg_b1 * d2f_dz_db1 + gg_b2 * d2f_dz_db2 + gg_z * d2f_dz2);

  // For complex types, PyTorch expects grad * conj(derivative)
  if constexpr (hyp1f2_is_complex_v<T>) {
    return {
      std::conj(grad_grad),
      std::conj(grad_a),
      std::conj(grad_b1),
      std::conj(grad_b2),
      std::conj(grad_z)
    };
  } else {
    return {grad_grad, grad_a, grad_b1, grad_b2, grad_z};
  }
}

} // namespace torchscience::kernel::special_functions
