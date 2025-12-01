#pragma once

#include <cmath>
#include <tuple>

#include "gamma_probability_density.h"
#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Gamma PDF gradient
// f(x; shape, scale) = x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape))
//
// log f = (shape-1)*log(x) - x/scale - shape*log(scale) - lgamma(shape)
//
// df/dx = f * [(shape-1)/x - 1/scale]
// df/dshape = f * [log(x) - log(scale) - psi(shape)]
// df/dscale = f * [x/scale^2 - shape/scale]
template <typename T>
std::tuple<T, T, T> gamma_probability_density_backward(T gradient, T x, T shape, T scale) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  T pdf = gamma_probability_density(x, shape, scale);

  // df/dx
  T df_dx = pdf * ((shape - T(1)) / x - T(1) / scale);

  // df/dshape
  T psi_shape = special_functions::digamma(shape);
  T df_dshape = pdf * (std::log(x) - std::log(scale) - psi_shape);

  // df/dscale
  T df_dscale = pdf * (x / (scale * scale) - shape / scale);

  return {
    gradient * df_dx,
    gradient * df_dshape,
    gradient * df_dscale
  };
}

}  // namespace torchscience::kernel::probability
