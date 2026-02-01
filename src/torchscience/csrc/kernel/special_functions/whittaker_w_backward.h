#pragma once

#include <cmath>
#include <tuple>

#include "whittaker_w.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute Whittaker W function value
// W_kappa,mu(z) = exp(-z/2) * z^(mu+1/2) * U(a, b, z)
// where a = mu - kappa + 1/2, b = 2*mu + 1
template <typename T>
T whit_w_value(T kappa, T mu, T z) {
  T a = mu - kappa + T(0.5);
  T b = T(2) * mu + T(1);

  T exp_term = std::exp(-z / T(2));
  T pow_term = std::pow(z, mu + T(0.5));
  T U_val = confluent_hypergeometric_u(a, b, z);

  return exp_term * pow_term * U_val;
}

} // namespace detail

// Backward pass for Whittaker W function
// W_kappa,mu(z) = exp(-z/2) * z^(mu+1/2) * U(a, b, z)
// where a = mu - kappa + 1/2, b = 2*mu + 1
//
// Returns (grad_kappa, grad_mu, grad_z)
template <typename T>
std::tuple<T, T, T> whittaker_w_backward(T grad_output, T kappa, T mu, T z) {
  using detail::whit_w_epsilon;
  using detail::whit_w_is_complex_v;
  using detail::whit_w_real_type_t;
  using detail::whit_w_value;

  using real_t = whit_w_real_type_t<T>;

  // Handle z = 0 case: gradients are zero
  if (std::abs(z) < whit_w_epsilon<T>()) {
    return {T(0), T(0), T(0)};
  }

  // Transformation parameters
  T a = mu - kappa + T(0.5);
  T b = T(2) * mu + T(1);

  // Prefactors
  T exp_term = std::exp(-z / T(2));
  T pow_term = std::pow(z, mu + T(0.5));
  T U_val = confluent_hypergeometric_u(a, b, z);

  // ===== Gradient w.r.t. z (analytical via product rule) =====
  // W_kappa,mu(z) = exp(-z/2) * z^(mu+1/2) * U(a, b, z)
  // d/dz = d(exp(-z/2))/dz * z^(mu+1/2) * U + exp(-z/2) * d(z^(mu+1/2))/dz * U
  //        + exp(-z/2) * z^(mu+1/2) * dU/dz
  //
  // d(exp(-z/2))/dz = -1/2 * exp(-z/2)
  // d(z^(mu+1/2))/dz = (mu+1/2) * z^(mu-1/2) = (mu+1/2)/z * z^(mu+1/2)
  // dU(a, b, z)/dz = -a * U(a+1, b+1, z)

  T U_shifted = confluent_hypergeometric_u(a + T(1), b + T(1), z);
  T dU_dz = -a * U_shifted;

  T dW_dz = exp_term * pow_term * (
    T(-0.5) * U_val +           // from d(exp(-z/2))/dz
    (mu + T(0.5)) / z * U_val + // from d(z^(mu+1/2))/dz
    dU_dz                       // from dU/dz
  );

  // ===== Gradients w.r.t. kappa and mu (finite differences) =====
  // These are more complex due to parameter derivatives of the hypergeometric
  real_t eps_real = std::sqrt(whit_w_epsilon<T>());
  T eps = T(eps_real);

  // Gradient w.r.t. kappa (a = mu - kappa + 1/2, so da/dkappa = -1)
  T W_kappa_plus = whit_w_value(kappa + eps, mu, z);
  T W_kappa_minus = whit_w_value(kappa - eps, mu, z);
  T dW_dkappa = (W_kappa_plus - W_kappa_minus) / (T(2) * eps);

  // Gradient w.r.t. mu
  // mu affects both the prefactor z^(mu+1/2) and the hypergeometric params
  T W_mu_plus = whit_w_value(kappa, mu + eps, z);
  T W_mu_minus = whit_w_value(kappa, mu - eps, z);
  T dW_dmu = (W_mu_plus - W_mu_minus) / (T(2) * eps);

  // Apply chain rule with upstream gradient
  if constexpr (whit_w_is_complex_v<T>) {
    return {
      grad_output * std::conj(dW_dkappa),
      grad_output * std::conj(dW_dmu),
      grad_output * std::conj(dW_dz)
    };
  } else {
    return {
      grad_output * dW_dkappa,
      grad_output * dW_dmu,
      grad_output * dW_dz
    };
  }
}

} // namespace torchscience::kernel::special_functions
