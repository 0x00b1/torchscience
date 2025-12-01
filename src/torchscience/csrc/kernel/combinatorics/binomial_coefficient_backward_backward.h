#pragma once

#include <tuple>
#include <cmath>
#include "binomial_coefficient.h"
#include "../special_functions/digamma.h"
#include "../special_functions/trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> binomial_coefficient_backward_backward(
  T gg_n,
  T gg_k,
  T grad_output,
  T n,
  T k
) {
  // For edge cases where C(n,k) = 0 or constant, second derivatives are 0
  if (k < T(0) || (n >= T(0) && k > n && std::floor(n) == n) || k == T(0)) {
    return {T(0), T(0), T(0)};
  }

  T c_nk = binomial_coefficient(n, k);

  T psi_n1 = special_functions::digamma(n + T(1));
  T psi_k1 = special_functions::digamma(k + T(1));
  T psi_nmk1 = special_functions::digamma(n - k + T(1));

  T psi1_n1 = special_functions::trigamma(n + T(1));
  T psi1_k1 = special_functions::trigamma(k + T(1));
  T psi1_nmk1 = special_functions::trigamma(n - k + T(1));

  T dn = psi_n1 - psi_nmk1;
  T dk = -psi_k1 + psi_nmk1;

  // Second derivatives:
  // d2/dn2 C = C * (dn^2 + psi1(n+1) - psi1(n-k+1))
  // d2/dk2 C = C * (dk^2 - psi1(k+1) - psi1(n-k+1))
  // d2/dndk C = C * (dn*dk + psi1(n-k+1))

  T d2_nn = c_nk * (dn * dn + psi1_n1 - psi1_nmk1);
  T d2_kk = c_nk * (dk * dk - psi1_k1 - psi1_nmk1);
  T d2_nk = c_nk * (dn * dk + psi1_nmk1);

  // grad_grad_output = gg_n * d(grad_n)/d(grad_output) + gg_k * d(grad_k)/d(grad_output)
  //                  = gg_n * C * dn + gg_k * C * dk
  T grad_grad_output = gg_n * c_nk * dn + gg_k * c_nk * dk;

  // grad_n from backward_backward:
  // d/dn of (grad_output * C * dn) w.r.t. n, summed with cross terms
  T grad_n = grad_output * (gg_n * d2_nn + gg_k * d2_nk);

  // grad_k from backward_backward:
  T grad_k = grad_output * (gg_n * d2_nk + gg_k * d2_kk);

  return {grad_grad_output, grad_n, grad_k};
}

} // namespace torchscience::kernel::special_functions
