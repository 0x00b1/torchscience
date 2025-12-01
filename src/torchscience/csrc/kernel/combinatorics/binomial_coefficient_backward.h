#pragma once

#include <tuple>
#include <cmath>
#include "binomial_coefficient.h"
#include "../special_functions/digamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> binomial_coefficient_backward(T grad_output, T n, T k) {
  // For edge cases where C(n,k) = 0, gradients are 0
  if (k < T(0) || (n >= T(0) && k > n && std::floor(n) == n)) {
    return {T(0), T(0)};
  }

  // For k = 0, C(n,0) = 1 constant, so gradients are 0
  if (k == T(0)) {
    return {T(0), T(0)};
  }

  T c_nk = binomial_coefficient(n, k);

  T psi_n_plus_1 = special_functions::digamma(n + T(1));
  T psi_k_plus_1 = special_functions::digamma(k + T(1));
  T psi_n_minus_k_plus_1 = special_functions::digamma(n - k + T(1));

  // d/dn C(n,k) = C(n,k) * (psi(n+1) - psi(n-k+1))
  T grad_n = grad_output * c_nk * (psi_n_plus_1 - psi_n_minus_k_plus_1);

  // d/dk C(n,k) = C(n,k) * (-psi(k+1) + psi(n-k+1))
  T grad_k = grad_output * c_nk * (-psi_k_plus_1 + psi_n_minus_k_plus_1);

  return {grad_n, grad_k};
}

} // namespace torchscience::kernel::special_functions
