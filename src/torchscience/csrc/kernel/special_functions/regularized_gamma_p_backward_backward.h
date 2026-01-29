#pragma once

#include <cmath>
#include <tuple>
#include "regularized_gamma_p.h"
#include "regularized_gamma_p_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order gradients for regularized_gamma_p
// Uses numerical differentiation
template <typename T>
std::tuple<T, T, T> regularized_gamma_p_backward_backward(
    T grad_grad_a, T grad_grad_x, T grad, T a, T x) {

  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(a)));
  T eps_x = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(x)));

  // d^2P/da^2 (numerical)
  auto [grad_a_plus, grad_x_plus] = regularized_gamma_p_backward(T(1), a + eps, x);
  auto [grad_a_minus, grad_x_minus] = regularized_gamma_p_backward(T(1), a - eps, x);
  T d2P_da2 = (grad_a_plus - grad_a_minus) / (T(2) * eps);

  // d^2P/dx^2 (numerical)
  auto [grad_a_px, grad_x_px] = regularized_gamma_p_backward(T(1), a, x + eps_x);
  auto [grad_a_mx, grad_x_mx] = regularized_gamma_p_backward(T(1), a, x - eps_x);
  T d2P_dx2 = (grad_x_px - grad_x_mx) / (T(2) * eps_x);

  // d^2P/dadx (numerical)
  T d2P_dadx = (grad_x_plus - grad_x_minus) / (T(2) * eps);

  // Compute gradients
  T gg_output = grad_grad_a * regularized_gamma_p_grad_a(a, x) +
                grad_grad_x * regularized_gamma_p_grad_x(a, x);

  return {
    gg_output,
    grad * (grad_grad_a * d2P_da2 + grad_grad_x * d2P_dadx),
    grad * (grad_grad_a * d2P_dadx + grad_grad_x * d2P_dx2)
  };
}

}  // namespace torchscience::kernel::special_functions
