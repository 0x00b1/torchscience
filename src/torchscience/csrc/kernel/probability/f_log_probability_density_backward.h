#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Gradients of F-distribution log probability density function
// logpdf(x; dfn, dfd) = (dfn/2) * log(dfn/dfd) + (dfn/2 - 1) * log(x)
//                      - ((dfn + dfd)/2) * log(1 + dfn*x/dfd)
//                      - log_beta(dfn/2, dfd/2)
//
// d/dx = (dfn/2 - 1)/x - ((dfn + dfd)/2) * (dfn/dfd) / (1 + dfn*x/dfd)
// d/ddfn = 0.5 * (log(dfn/dfd) + 1/dfn + log(x) - log(1 + dfn*x/dfd) - x/(dfd + dfn*x))
//          - 0.5 * (digamma(dfn/2) - digamma((dfn+dfd)/2))
// d/ddfd = 0.5 * (-dfn/dfd + dfn*x/(dfd*(dfd + dfn*x)) - log(1 + dfn*x/dfd))
//          - 0.5 * (digamma(dfd/2) - digamma((dfn+dfd)/2))
template <typename T>
std::tuple<T, T, T> f_log_probability_density_backward(T gradient, T x, T dfn, T dfd) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  T half_dfn = dfn / T(2);
  T half_dfd = dfd / T(2);
  T half_sum = (dfn + dfd) / T(2);
  T ratio = dfn * x / dfd;
  T one_plus_ratio = T(1) + ratio;
  T log_one_plus_ratio = std::log(one_plus_ratio);
  T denom = dfd + dfn * x;

  // d/dx = (dfn/2 - 1)/x - ((dfn + dfd)/2) * (dfn/dfd) / (1 + dfn*x/dfd)
  T grad_x = gradient * ((half_dfn - T(1)) / x - half_sum * (dfn / dfd) / one_plus_ratio);

  // Precompute digamma values
  T psi_half_dfn = special_functions::digamma(half_dfn);
  T psi_half_dfd = special_functions::digamma(half_dfd);
  T psi_half_sum = special_functions::digamma(half_sum);

  // d/ddfn = 0.5 * (log(dfn/dfd) + 1/dfn + log(x) - log(1 + dfn*x/dfd) - x/(dfd + dfn*x))
  //          - 0.5 * (digamma(dfn/2) - digamma((dfn+dfd)/2))
  T grad_dfn = gradient * (T(0.5) * (std::log(dfn / dfd) + T(1) / dfn + std::log(x)
                                     - log_one_plus_ratio - x / denom)
                          - T(0.5) * (psi_half_dfn - psi_half_sum));

  // d/ddfd = 0.5 * (-dfn/dfd + dfn*x/(dfd*(dfd + dfn*x)) - log(1 + dfn*x/dfd))
  //          - 0.5 * (digamma(dfd/2) - digamma((dfn+dfd)/2))
  T grad_dfd = gradient * (T(0.5) * (-dfn / dfd + dfn * x / (dfd * denom) - log_one_plus_ratio)
                          - T(0.5) * (psi_half_dfd - psi_half_sum));

  return {grad_x, grad_dfn, grad_dfd};
}

}  // namespace torchscience::kernel::probability
