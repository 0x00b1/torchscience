#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Gamma PDF: f(x; shape, scale) = x^(shape-1) * exp(-x/scale) / (scale^shape * Gamma(shape))
template <typename T>
T gamma_probability_density(T x, T shape, T scale) {
  if (x <= T(0)) return T(0);

  // Compute in log space for numerical stability
  T log_probability_density = (shape - T(1)) * std::log(x) - x / scale
            - shape * std::log(scale) - std::lgamma(shape);
  return std::exp(log_probability_density);
}

}  // namespace torchscience::kernel::probability
