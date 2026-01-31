#pragma once

#include <cmath>
#include <tuple>

#include "confluent_hypergeometric_m.h"
#include "confluent_hypergeometric_m_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> confluent_hypergeometric_m_backward_backward(
  T gg_a,
  T gg_b,
  T gg_z,
  T grad,
  T a,
  T b,
  T z
) {
  using detail::hyp1f1_epsilon;
  using detail::hyp1f1_is_complex_v;
  using real_t = detail::hyp1f1_real_type_t<T>;

  // d²/dz² M(a,b,z) = (a/b) * ((a+1)/(b+1)) * M(a+2, b+2, z)
  T dz_coef = a / b;
  T d2z_coef = dz_coef * (a + T(1)) / (b + T(1));

  T f_shifted = confluent_hypergeometric_m(a + T(1), b + T(1), z);
  T f_double_shifted = confluent_hypergeometric_m(a + T(2), b + T(2), z);

  // Use larger step for second derivatives (4th root of epsilon)
  real_t eps_real = std::pow(hyp1f1_epsilon<T>(), real_t(0.25));
  T eps = T(eps_real);

  // First derivatives via finite difference
  T f_a_plus = confluent_hypergeometric_m(a + eps, b, z);
  T f_a_minus = confluent_hypergeometric_m(a - eps, b, z);
  T df_da = (f_a_plus - f_a_minus) / (T(2) * eps);

  T f_b_plus = confluent_hypergeometric_m(a, b + eps, z);
  T f_b_minus = confluent_hypergeometric_m(a, b - eps, z);
  T df_db = (f_b_plus - f_b_minus) / (T(2) * eps);

  T df_dz = dz_coef * f_shifted;

  // gg_out = sum of gg_i * d(grad_i)/d(grad)
  T gg_out;
  if constexpr (hyp1f1_is_complex_v<T>) {
    gg_out = gg_a * std::conj(df_da) + gg_b * std::conj(df_db) + gg_z * std::conj(df_dz);
  } else {
    gg_out = gg_a * df_da + gg_b * df_db + gg_z * df_dz;
  }

  // Second derivative w.r.t. z (exact)
  T d2f_dz2 = d2z_coef * f_double_shifted;

  // Second derivatives via finite differences on backward
  auto [da_a_plus, db_a_plus, dz_a_plus] =
    confluent_hypergeometric_m_backward(T(1), a + eps, b, z);
  auto [da_a_minus, db_a_minus, dz_a_minus] =
    confluent_hypergeometric_m_backward(T(1), a - eps, b, z);
  T d2f_da2 = (da_a_plus - da_a_minus) / (T(2) * eps);

  auto [da_b_plus, db_b_plus, dz_b_plus] =
    confluent_hypergeometric_m_backward(T(1), a, b + eps, z);
  auto [da_b_minus, db_b_minus, dz_b_minus] =
    confluent_hypergeometric_m_backward(T(1), a, b - eps, z);
  T d2f_db2 = (db_b_plus - db_b_minus) / (T(2) * eps);

  // Cross derivatives
  T d2f_dadb = (da_b_plus - da_b_minus) / (T(2) * eps);

  // Mixed with z
  auto [da_z_plus, db_z_plus, dz_z_plus] =
    confluent_hypergeometric_m_backward(T(1), a, b, z + eps);
  auto [da_z_minus, db_z_minus, dz_z_minus] =
    confluent_hypergeometric_m_backward(T(1), a, b, z - eps);
  T d2f_dadz = (da_z_plus - da_z_minus) / (T(2) * eps);
  T d2f_dbdz = (db_z_plus - db_z_minus) / (T(2) * eps);

  // Compute new gradients using Hessian
  T new_grad_a = grad * (gg_a * d2f_da2 + gg_b * d2f_dadb + gg_z * d2f_dadz);
  T new_grad_b = grad * (gg_a * d2f_dadb + gg_b * d2f_db2 + gg_z * d2f_dbdz);
  T new_grad_z = grad * (gg_a * d2f_dadz + gg_b * d2f_dbdz + gg_z * d2f_dz2);

  return {gg_out, new_grad_a, new_grad_b, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
