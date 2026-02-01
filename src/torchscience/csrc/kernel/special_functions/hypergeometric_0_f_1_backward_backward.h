#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_0_f_1.h"
#include "hypergeometric_0_f_1_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for hypergeometric 0F1 function
// Returns (gg_out, new_grad_b, new_grad_z)
template <typename T>
std::tuple<T, T, T> hypergeometric_0_f_1_backward_backward(
  T gg_b,
  T gg_z,
  T grad,
  T b,
  T z
) {
  using detail::hyp0f1_epsilon;
  using detail::hyp0f1_is_complex_v;
  using real_t = detail::hyp0f1_real_type_t<T>;

  // d²/dz²[0F1(;b;z)] = 0F1(;b+2;z) / (b * (b+1))
  T d2f_dz2 = hypergeometric_0_f_1(b + T(2), z) / (b * (b + T(1)));

  // First derivatives
  T df_dz = hypergeometric_0_f_1(b + T(1), z) / b;

  // Use larger step for second derivatives (4th root of epsilon)
  real_t eps_real = std::pow(hyp0f1_epsilon<T>(), real_t(0.25));
  T eps = T(eps_real);

  // First derivative w.r.t. b via finite difference
  T f_b_plus = hypergeometric_0_f_1(b + eps, z);
  T f_b_minus = hypergeometric_0_f_1(b - eps, z);
  T df_db = (f_b_plus - f_b_minus) / (T(2) * eps);

  // gg_out = sum of gg_i * d(grad_i)/d(grad)
  T gg_out;
  if constexpr (hyp0f1_is_complex_v<T>) {
    gg_out = gg_b * std::conj(df_db) + gg_z * std::conj(df_dz);
  } else {
    gg_out = gg_b * df_db + gg_z * df_dz;
  }

  // Second derivatives via finite differences on backward
  auto [db_b_plus, dz_b_plus] = hypergeometric_0_f_1_backward(T(1), b + eps, z);
  auto [db_b_minus, dz_b_minus] = hypergeometric_0_f_1_backward(T(1), b - eps, z);
  T d2f_db2 = (db_b_plus - db_b_minus) / (T(2) * eps);

  // Cross derivative d²f/(db dz)
  T d2f_dbdz = (dz_b_plus - dz_b_minus) / (T(2) * eps);

  // Mixed with z
  auto [db_z_plus, dz_z_plus] = hypergeometric_0_f_1_backward(T(1), b, z + eps);
  auto [db_z_minus, dz_z_minus] = hypergeometric_0_f_1_backward(T(1), b, z - eps);
  T d2f_dzdb = (db_z_plus - db_z_minus) / (T(2) * eps);

  // Compute new gradients using Hessian
  // new_grad_i = grad * sum_j(gg_j * d2f/(di dj))
  T new_grad_b = grad * (gg_b * d2f_db2 + gg_z * d2f_dbdz);
  T new_grad_z = grad * (gg_b * d2f_dzdb + gg_z * d2f_dz2);

  return {gg_out, new_grad_b, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
