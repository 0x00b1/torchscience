#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_2_f_1.h"
#include "hypergeometric_2_f_1_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T, T> hypergeometric_2_f_1_backward_backward(
  T gg_a,
  T gg_b,
  T gg_c,
  T gg_z,
  T grad,
  T a,
  T b,
  T c,
  T z
) {
  using detail::hyp2f1_epsilon;
  using detail::is_complex_v;
  using real_t = detail::real_type_t<T>;

  // Second derivative w.r.t. z:
  // d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
  // d²/dz² 2F1(a,b;c;z) = (a*b/c) * ((a+1)*(b+1)/(c+1)) * 2F1(a+2, b+2; c+2; z)
  T dz_coef = a * b / c;
  T d2z_coef = dz_coef * (a + T(1)) * (b + T(1)) / (c + T(1));

  T f_shifted = hypergeometric_2_f_1(a + T(1), b + T(1), c + T(1), z);
  T f_double_shifted = hypergeometric_2_f_1(a + T(2), b + T(2), c + T(2), z);

  // gg_out: gradient of backward output w.r.t. upstream grad
  // This is sum of: gg_a * d(da)/d(grad) + gg_b * d(db)/d(grad) + ...
  // Since da = grad * df/da, we have d(da)/d(grad) = df/da
  // We compute this via finite differences on the forward function
  // Use larger step for second derivatives (4th root of epsilon is optimal)
  real_t eps_real = std::pow(hyp2f1_epsilon<T>(), real_t(0.25));
  T eps = T(eps_real);

  // df/da via finite difference
  T f_a_plus = hypergeometric_2_f_1(a + eps, b, c, z);
  T f_a_minus = hypergeometric_2_f_1(a - eps, b, c, z);
  T df_da = (f_a_plus - f_a_minus) / (T(2) * eps);

  // df/db via finite difference
  T f_b_plus = hypergeometric_2_f_1(a, b + eps, c, z);
  T f_b_minus = hypergeometric_2_f_1(a, b - eps, c, z);
  T df_db = (f_b_plus - f_b_minus) / (T(2) * eps);

  // df/dc via finite difference
  T f_c_plus = hypergeometric_2_f_1(a, b, c + eps, z);
  T f_c_minus = hypergeometric_2_f_1(a, b, c - eps, z);
  T df_dc = (f_c_plus - f_c_minus) / (T(2) * eps);

  // df/dz = dz_coef * f_shifted (exact)
  T df_dz = dz_coef * f_shifted;

  // gg_out = sum over outputs of gg_i * d(grad_i)/d(grad)
  // For complex: grad_a = grad * conj(df/da), so d(grad_a)/d(grad) = conj(df/da)
  T gg_out;
  if constexpr (is_complex_v<T>) {
    gg_out = gg_a * std::conj(df_da) + gg_b * std::conj(df_db) +
             gg_c * std::conj(df_dc) + gg_z * std::conj(df_dz);
  } else {
    gg_out = gg_a * df_da + gg_b * df_db + gg_c * df_dc + gg_z * df_dz;
  }

  // d²f/dz² = (a*b/c) * ((a+1)*(b+1)/(c+1)) * 2F1(a+2, b+2; c+2; z) (exact)
  T d2f_dz2 = d2z_coef * f_double_shifted;

  // Second derivatives for parameters via finite differences on backward
  // The backward returns grad * conj(df/d*) for complex, so we need to account for that
  // We pass grad=1 to get the raw derivative (which already has conj applied for complex)

  // d²f/da² = d/da(df/da)
  auto [da_a_plus, db_a_plus, dc_a_plus, dz_a_plus] =
    hypergeometric_2_f_1_backward(T(1), a + eps, b, c, z);
  auto [da_a_minus, db_a_minus, dc_a_minus, dz_a_minus] =
    hypergeometric_2_f_1_backward(T(1), a - eps, b, c, z);
  T d2f_da2 = (da_a_plus - da_a_minus) / (T(2) * eps);

  // d²f/db² = d/db(df/db)
  auto [da_b_plus, db_b_plus, dc_b_plus, dz_b_plus] =
    hypergeometric_2_f_1_backward(T(1), a, b + eps, c, z);
  auto [da_b_minus, db_b_minus, dc_b_minus, dz_b_minus] =
    hypergeometric_2_f_1_backward(T(1), a, b - eps, c, z);
  T d2f_db2 = (db_b_plus - db_b_minus) / (T(2) * eps);

  // d²f/dc² = d/dc(df/dc)
  auto [da_c_plus, db_c_plus, dc_c_plus, dz_c_plus] =
    hypergeometric_2_f_1_backward(T(1), a, b, c + eps, z);
  auto [da_c_minus, db_c_minus, dc_c_minus, dz_c_minus] =
    hypergeometric_2_f_1_backward(T(1), a, b, c - eps, z);
  T d2f_dc2 = (dc_c_plus - dc_c_minus) / (T(2) * eps);

  // Cross derivatives for mixed terms
  // d²f/dadb = d/db(df/da)
  T d2f_dadb = (da_b_plus - da_b_minus) / (T(2) * eps);
  // d²f/dadc = d/dc(df/da)
  T d2f_dadc = (da_c_plus - da_c_minus) / (T(2) * eps);
  // d²f/dbdc = d/dc(df/db)
  T d2f_dbdc = (db_c_plus - db_c_minus) / (T(2) * eps);

  // d²f/dadz = d/dz(df/da) via finite difference on z
  auto [da_z_plus, db_z_plus, dc_z_plus, dz_z_plus] =
    hypergeometric_2_f_1_backward(T(1), a, b, c, z + eps);
  auto [da_z_minus, db_z_minus, dc_z_minus, dz_z_minus] =
    hypergeometric_2_f_1_backward(T(1), a, b, c, z - eps);
  T d2f_dadz = (da_z_plus - da_z_minus) / (T(2) * eps);
  T d2f_dbdz = (db_z_plus - db_z_minus) / (T(2) * eps);
  T d2f_dcdz = (dc_z_plus - dc_z_minus) / (T(2) * eps);

  // new_grad_* = grad * sum_i(gg_i * d²f/di d*)
  // Uses Hessian symmetry: d²f/dadz = d²f/dzda, etc.
  T new_grad_a = grad * (gg_a * d2f_da2 + gg_b * d2f_dadb + gg_c * d2f_dadc + gg_z * d2f_dadz);
  T new_grad_b = grad * (gg_a * d2f_dadb + gg_b * d2f_db2 + gg_c * d2f_dbdc + gg_z * d2f_dbdz);
  T new_grad_c = grad * (gg_a * d2f_dadc + gg_b * d2f_dbdc + gg_c * d2f_dc2 + gg_z * d2f_dcdz);
  T new_grad_z = grad * (gg_a * d2f_dadz + gg_b * d2f_dbdz + gg_c * d2f_dcdz + gg_z * d2f_dz2);

  return {gg_out, new_grad_a, new_grad_b, new_grad_c, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
