#pragma once

#include <cmath>
#include <tuple>
#include "../special_functions/incomplete_beta_backward.h"

namespace torchscience::kernel::probability {

// Backward for binomial CDF
// CDF = I_{1-p}(n-k, k+1) where I is incomplete_beta(x, a, b)
// dCDF/dp = dI/dx * dx/dp = dI/dx * (-1) = -pdf(1-p; n-k, k+1)
// where pdf is the beta distribution pdf
//
// k and n are discrete, so their gradients are 0
template <typename T>
std::tuple<T, T, T> binomial_cumulative_distribution_backward(T gradient, T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases: gradient is 0
  if (k < T(0) || k >= n) {
    return {T(0), T(0), T(0)};
  }

  // x = 1-p, a = n-k, b = k+1
  T x = T(1) - p;
  T a = n - k;
  T b = k + T(1);

  // Get gradient w.r.t. x from incomplete_beta_backward
  auto [grad_x, grad_a, grad_b] = special_functions::incomplete_beta_backward(gradient, x, a, b);

  // Chain rule: dx/dp = -1, so grad_p = grad_x * (-1)
  T grad_p = -grad_x;

  // k and n are discrete, so gradients are 0
  return {T(0), T(0), grad_p};
}

}  // namespace torchscience::kernel::probability
