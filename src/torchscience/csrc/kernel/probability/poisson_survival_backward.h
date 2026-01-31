#pragma once

#include <cmath>
#include <tuple>
#include "../special_functions/regularized_gamma_p_backward.h"

namespace torchscience::kernel::probability {

// Backward for Poisson survival function
// S(k; rate) = P(k+1, rate) where P is regularized_gamma_p(a, x)
// dS/drate = dP/dx (since rate = x in the gamma function)
//
// k is discrete, so its gradient is 0
template <typename T>
std::tuple<T, T> poisson_survival_backward(T gradient, T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0) || rate <= T(0)) {
    return {T(0), T(0)};
  }

  // a = k+1, x = rate
  T a = k + T(1);

  // Get gradients from regularized_gamma_p_backward
  auto [grad_a, grad_x] = special_functions::regularized_gamma_p_backward(gradient, a, rate);

  // We only need grad_rate = grad_x (grad_k = 0 since k is discrete)
  return {T(0), grad_x};
}

}  // namespace torchscience::kernel::probability
