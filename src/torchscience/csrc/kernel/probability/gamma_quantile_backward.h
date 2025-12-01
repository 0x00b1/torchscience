#pragma once

#include <cmath>
#include <tuple>

#include "gamma_cumulative_distribution_backward.h"
#include "gamma_probability_density.h"
#include "gamma_quantile.h"

namespace torchscience::kernel::probability {

// Gamma PPF gradient via implicit differentiation
// If F(x; shape, scale) = p, then x = G(p; shape, scale) = PPF(p; shape, scale)
// By implicit differentiation:
//   dG/dp = 1 / pdf(x)
//   dG/dshape = -dF/dshape / pdf(x)
//   dG/dscale = -dF/dscale / pdf(x)
template <typename T>
std::tuple<T, T, T> gamma_quantile_backward(T gradient, T p, T shape, T scale) {
  T x = gamma_quantile(p, shape, scale);
  T pdf = gamma_probability_density(x, shape, scale);

  if (pdf < T(1e-15)) {
    return {T(0), T(0), T(0)};
  }

  // Get CDF gradients w.r.t. shape and scale at the quantile x
  auto [grad_x_cumulative_distribution, grad_shape_cumulative_distribution, grad_scale_cumulative_distribution] = gamma_cumulative_distribution_backward(T(1), x, shape, scale);

  // dPPF/dp = 1/pdf
  T dppf_dp = T(1) / pdf;

  // dPPF/dshape = -dCDF/dshape / pdf
  T dppf_dshape = -grad_shape_cumulative_distribution / pdf;

  // dPPF/dscale = -dCDF/dscale / pdf
  T dppf_dscale = -grad_scale_cumulative_distribution / pdf;

  return {
    gradient * dppf_dp,
    gradient * dppf_dshape,
    gradient * dppf_dscale
  };
}

}  // namespace torchscience::kernel::probability
