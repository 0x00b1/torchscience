#pragma once

#include <cmath>
#include <tuple>
#include "../special_functions/regularized_gamma_q_backward.h"

namespace torchscience::kernel::probability {

// Backward for Poisson CDF
// CDF = Q(k+1, rate) where Q is regularized_gamma_q(a, x)
// dCDF/drate = dQ/dx (since rate = x in the gamma function)
//
// k is discrete, so its gradient is 0
template <typename T>
std::tuple<T, T> poisson_cumulative_distribution_backward(T gradient, T k, T rate) {
  k = std::floor(k);

  // Boundary case
  if (k < T(0)) {
    return {T(0), T(0)};
  }

  // a = k+1, x = rate
  T a = k + T(1);

  // Get gradients from regularized_gamma_q_backward
  auto [grad_a, grad_x] = special_functions::regularized_gamma_q_backward(gradient, a, rate);

  // We only need grad_rate = grad_x (grad_k = 0 since k is discrete)
  return {T(0), grad_x};
}

}  // namespace torchscience::kernel::probability
