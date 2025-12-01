#pragma once

#include <cmath>
#include <tuple>

#include "beta_cumulative_distribution_backward.h"
#include "beta_probability_density.h"
#include "beta_quantile.h"

namespace torchscience::kernel::probability {

// Beta PPF gradient via implicit differentiation
// If F(x; a, b) = p, then x = G(p; a, b) = PPF(p; a, b)
// By implicit differentiation:
//   dG/dp = 1 / (dF/dx) = 1 / pdf(x)
//   dG/da = -(dF/da) / (dF/dx) = -(dF/da) / pdf(x)
//   dG/db = -(dF/db) / (dF/dx) = -(dF/db) / pdf(x)
template <typename T>
std::tuple<T, T, T> beta_quantile_backward(T gradient, T p, T a, T b) {
  T x = beta_quantile(p, a, b);
  T pdf = beta_probability_density(x, a, b);

  if (pdf < T(1e-15)) {
    return {T(0), T(0), T(0)};
  }

  // Get CDF gradients w.r.t. a and b at the quantile x
  auto [grad_x_cumulative_distribution, grad_a_cumulative_distribution, grad_b_cumulative_distribution] = beta_cumulative_distribution_backward(T(1), x, a, b);

  // dPPF/dp = 1/pdf
  T dppf_dp = T(1) / pdf;

  // dPPF/da = -dCDF/da / pdf
  T dppf_da = -grad_a_cumulative_distribution / pdf;

  // dPPF/db = -dCDF/db / pdf
  T dppf_db = -grad_b_cumulative_distribution / pdf;

  return {
    gradient * dppf_dp,
    gradient * dppf_da,
    gradient * dppf_db
  };
}

}  // namespace torchscience::kernel::probability
