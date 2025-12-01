#pragma once

#include <cmath>
#include <tuple>
#include "chi2_cumulative_distribution.h"
#include "../special_functions/regularized_gamma_p_backward.h"

namespace torchscience::kernel::probability {

// Gradient of chi2_cumulative_distribution with respect to x:
// F(x; df) = P(df/2, x/2)
// dF/dx = dP/dx * (1/2) = (x/2)^(df/2-1) * exp(-x/2) / Gamma(df/2) * (1/2)
template <typename T>
T chi2_cumulative_distribution_grad_x(T x, T df) {
  if (x <= T(0)) return T(0);
  T a = df / T(2);
  T half_x = x / T(2);
  // dP/d(x/2) * d(x/2)/dx = dP/d(x/2) * 0.5
  return special_functions::regularized_gamma_p_grad_x(a, half_x) * T(0.5);
}

// Gradient of chi2_cumulative_distribution with respect to df:
// dF/ddf = dP/da * da/ddf = dP/da * (1/2)
template <typename T>
T chi2_cumulative_distribution_grad_df(T x, T df) {
  if (x <= T(0)) return T(0);
  T a = df / T(2);
  T half_x = x / T(2);
  return special_functions::regularized_gamma_p_grad_a(a, half_x) * T(0.5);
}

template <typename T>
std::tuple<T, T> chi2_cumulative_distribution_backward(T grad_output, T x, T df) {
  return {
    grad_output * chi2_cumulative_distribution_grad_x(x, df),
    grad_output * chi2_cumulative_distribution_grad_df(x, df)
  };
}

}  // namespace torchscience::kernel::probability
