#pragma once

#include <cmath>
#include <tuple>
#include "../special_functions/incomplete_beta_backward.h"

namespace torchscience::kernel::probability {

// Backward for binomial survival function
// S(k; n, p) = I_p(k+1, n-k) where I is incomplete_beta(x, a, b)
// dS/dp = dI/dx (since p = x in the beta function)
//
// k and n are discrete, so their gradients are 0
template <typename T>
std::tuple<T, T, T> binomial_survival_backward(T gradient, T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases: gradient is 0
  if (k < T(0) || k >= n || p <= T(0) || p >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  // x = p, a = k+1, b = n-k
  T x = p;
  T a = k + T(1);
  T b = n - k;

  // Get gradient w.r.t. x from incomplete_beta_backward
  auto [grad_x, grad_a, grad_b] = special_functions::incomplete_beta_backward(gradient, x, a, b);

  // grad_p = grad_x (direct mapping, no chain rule transformation)
  // k and n are discrete, so gradients are 0
  return {T(0), T(0), grad_x};
}

}  // namespace torchscience::kernel::probability
