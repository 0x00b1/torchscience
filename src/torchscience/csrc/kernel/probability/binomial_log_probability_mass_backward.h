#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Backward for binomial log probability mass function
// log(PMF) = log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1)
//          + k * log(p) + (n - k) * log(1 - p)
//
// d/dk = -digamma(k + 1) + digamma(n - k + 1) + log(p) - log(1 - p)
// d/dn = digamma(n + 1) - digamma(n - k + 1) + log(1 - p)
// d/dp = k/p - (n - k)/(1 - p)
template <typename T>
std::tuple<T, T, T> binomial_log_probability_mass_backward(T gradient, T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases: k < 0 or k > n
  if (k < T(0) || k > n) {
    return {T(0), T(0), T(0)};
  }

  // Edge cases for p
  if (p <= T(0) || p >= T(1)) {
    return {T(0), T(0), T(0)};
  }

  // d(log_pmf)/dk = -digamma(k + 1) + digamma(n - k + 1) + log(p) - log(1 - p)
  T grad_k = gradient * (-special_functions::digamma(k + T(1))
                       + special_functions::digamma(n - k + T(1))
                       + std::log(p)
                       - std::log(T(1) - p));

  // d(log_pmf)/dn = digamma(n + 1) - digamma(n - k + 1) + log(1 - p)
  T grad_n = gradient * (special_functions::digamma(n + T(1))
                       - special_functions::digamma(n - k + T(1))
                       + std::log(T(1) - p));

  // d(log_pmf)/dp = k/p - (n - k)/(1 - p)
  T grad_p = gradient * (k / p - (n - k) / (T(1) - p));

  return {grad_k, grad_n, grad_p};
}

}  // namespace torchscience::kernel::probability
