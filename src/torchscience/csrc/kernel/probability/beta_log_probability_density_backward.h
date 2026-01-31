#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Gradients of beta log probability density function
// logpdf(x; a, b) = (a - 1) * log(x) + (b - 1) * log(1 - x) - log_beta(a, b)
//
// d(logpdf)/dx = (a - 1)/x - (b - 1)/(1 - x)
// d(logpdf)/da = log(x) - digamma(a) + digamma(a + b)
// d(logpdf)/db = log(1 - x) - digamma(b) + digamma(a + b)
template <typename T>
std::tuple<T, T, T> beta_log_probability_density_backward(T gradient, T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  // d(logpdf)/dx = (a - 1)/x - (b - 1)/(1 - x)
  T grad_x = gradient * ((a - T(1)) / x - (b - T(1)) / (T(1) - x));

  // Precompute digamma values
  T psi_a = special_functions::digamma(a);
  T psi_b = special_functions::digamma(b);
  T psi_ab = special_functions::digamma(a + b);

  // d(logpdf)/da = log(x) - digamma(a) + digamma(a + b)
  T grad_a = gradient * (std::log(x) - psi_a + psi_ab);

  // d(logpdf)/db = log(1 - x) - digamma(b) + digamma(a + b)
  T grad_b = gradient * (std::log(T(1) - x) - psi_b + psi_ab);

  return {grad_x, grad_a, grad_b};
}

}  // namespace torchscience::kernel::probability
