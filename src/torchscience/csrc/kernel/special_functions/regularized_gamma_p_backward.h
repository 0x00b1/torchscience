#pragma once

#include <cmath>
#include <tuple>
#include "regularized_gamma_p.h"

namespace torchscience::kernel::special_functions {

// Gradient of P(a, x) with respect to x:
// dP/dx = x^(a-1) * exp(-x) / Gamma(a)
template <typename T>
T regularized_gamma_p_grad_x(T a, T x) {
  if (x <= T(0)) return T(0);
  return std::exp((a - T(1)) * std::log(x) - x - std::lgamma(a));
}

// Gradient of P(a, x) with respect to a:
// This requires numerical differentiation for now
template <typename T>
T regularized_gamma_p_grad_a(T a, T x) {
  T eps = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), static_cast<T>(std::abs(a)));
  T p_plus = regularized_gamma_p(a + eps, x);
  T p_minus = regularized_gamma_p(a - eps, x);
  return (p_plus - p_minus) / (T(2) * eps);
}

template <typename T>
std::tuple<T, T> regularized_gamma_p_backward(T grad_output, T a, T x) {
  return {
    grad_output * regularized_gamma_p_grad_a(a, x),
    grad_output * regularized_gamma_p_grad_x(a, x)
  };
}

}  // namespace torchscience::kernel::special_functions
