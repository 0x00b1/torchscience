#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Backward for Poisson log probability mass function
// log(PMF) = k * log(rate) - rate - log_gamma(k + 1)
// d/dk = log(rate) - digamma(k + 1)  (gradient w.r.t. continuous k)
// d/drate = k/rate - 1
template <typename T>
std::tuple<T, T> poisson_log_probability_mass_backward(T gradient, T k, T rate) {
  k = std::floor(k);

  // Boundary cases: k < 0 or rate <= 0
  if (k < T(0) || rate <= T(0)) {
    return {T(0), T(0)};
  }

  // d(log_pmf)/dk = log(rate) - digamma(k + 1)
  T grad_k = gradient * (std::log(rate) - special_functions::digamma(k + T(1)));

  // d(log_pmf)/drate = k/rate - 1
  T grad_rate = gradient * (k / rate - T(1));

  return {grad_k, grad_rate};
}

}  // namespace torchscience::kernel::probability
