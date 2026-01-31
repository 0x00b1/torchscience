#pragma once

#include <cmath>
#include <tuple>

#include "whittaker_w_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Whittaker W function
// W_kappa,mu(z) = exp(-z/2) * z^(mu+1/2) * U(a, b, z)
// where a = mu - kappa + 1/2, b = 2*mu + 1
//
// Arguments:
//   gg_kappa, gg_mu, gg_z: gradients of the loss w.r.t. grad_kappa, grad_mu, grad_z
//   grad_output: upstream gradient from forward pass
//   kappa, mu, z: input values
//
// Returns (gg_out, new_grad_kappa, new_grad_mu, new_grad_z)
template <typename T>
std::tuple<T, T, T, T> whittaker_w_backward_backward(
  T gg_kappa,
  T gg_mu,
  T gg_z,
  T grad_output,
  T kappa,
  T mu,
  T z
) {
  using detail::whit_w_epsilon;
  using detail::whit_w_is_complex_v;
  using detail::whit_w_real_type_t;

  using real_t = whit_w_real_type_t<T>;

  // Handle z = 0 case: all gradients are zero
  if (std::abs(z) < whit_w_epsilon<T>()) {
    return {T(0), T(0), T(0), T(0)};
  }

  // Use 4th root of epsilon for second derivatives (larger step for stability)
  real_t eps_real = std::pow(whit_w_epsilon<T>(), real_t(0.25));
  T eps = T(eps_real);

  // Get first derivatives via backward function
  auto [grad_kappa_base, grad_mu_base, grad_z_base] =
    whittaker_w_backward(T(1), kappa, mu, z);

  // ===== Compute gg_out =====
  // gg_out = sum of gg_i * d(grad_i)/d(grad_output)
  // Since grad_i = grad_output * df_di, we have d(grad_i)/d(grad_output) = df_di
  T gg_out;
  if constexpr (whit_w_is_complex_v<T>) {
    gg_out = gg_kappa * std::conj(grad_kappa_base) +
             gg_mu * std::conj(grad_mu_base) +
             gg_z * std::conj(grad_z_base);
  } else {
    gg_out = gg_kappa * grad_kappa_base +
             gg_mu * grad_mu_base +
             gg_z * grad_z_base;
  }

  // ===== Compute second derivatives via finite differences on backward =====

  // Derivatives w.r.t. kappa
  auto [dkappa_kappa_plus, dmu_kappa_plus, dz_kappa_plus] =
    whittaker_w_backward(T(1), kappa + eps, mu, z);
  auto [dkappa_kappa_minus, dmu_kappa_minus, dz_kappa_minus] =
    whittaker_w_backward(T(1), kappa - eps, mu, z);

  T d2W_dkappa2 = (dkappa_kappa_plus - dkappa_kappa_minus) / (T(2) * eps);
  T d2W_dkappa_dmu = (dmu_kappa_plus - dmu_kappa_minus) / (T(2) * eps);
  T d2W_dkappa_dz = (dz_kappa_plus - dz_kappa_minus) / (T(2) * eps);

  // Derivatives w.r.t. mu
  auto [dkappa_mu_plus, dmu_mu_plus, dz_mu_plus] =
    whittaker_w_backward(T(1), kappa, mu + eps, z);
  auto [dkappa_mu_minus, dmu_mu_minus, dz_mu_minus] =
    whittaker_w_backward(T(1), kappa, mu - eps, z);

  T d2W_dmu2 = (dmu_mu_plus - dmu_mu_minus) / (T(2) * eps);
  T d2W_dmu_dkappa = (dkappa_mu_plus - dkappa_mu_minus) / (T(2) * eps);
  T d2W_dmu_dz = (dz_mu_plus - dz_mu_minus) / (T(2) * eps);

  // Derivatives w.r.t. z
  auto [dkappa_z_plus, dmu_z_plus, dz_z_plus] =
    whittaker_w_backward(T(1), kappa, mu, z + eps);
  auto [dkappa_z_minus, dmu_z_minus, dz_z_minus] =
    whittaker_w_backward(T(1), kappa, mu, z - eps);

  T d2W_dz2 = (dz_z_plus - dz_z_minus) / (T(2) * eps);
  T d2W_dz_dkappa = (dkappa_z_plus - dkappa_z_minus) / (T(2) * eps);
  T d2W_dz_dmu = (dmu_z_plus - dmu_z_minus) / (T(2) * eps);

  // ===== Compute new gradients using Hessian =====
  // new_grad_i = grad_output * sum_j(gg_j * d2W/(di dj))
  T new_grad_kappa = grad_output * (
    gg_kappa * d2W_dkappa2 +
    gg_mu * d2W_dkappa_dmu +
    gg_z * d2W_dkappa_dz
  );

  T new_grad_mu = grad_output * (
    gg_kappa * d2W_dmu_dkappa +
    gg_mu * d2W_dmu2 +
    gg_z * d2W_dmu_dz
  );

  T new_grad_z = grad_output * (
    gg_kappa * d2W_dz_dkappa +
    gg_mu * d2W_dz_dmu +
    gg_z * d2W_dz2
  );

  return {gg_out, new_grad_kappa, new_grad_mu, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
