#pragma once

#include <cmath>

#include "../special_functions/log_beta.h"

namespace torchscience::kernel::probability {

// Beta PDF: f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)
template <typename T>
T beta_probability_density(T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) return T(0);

  // Compute in log space for numerical stability
  T log_probability_density = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x)
            - special_functions::log_beta(a, b);
  return std::exp(log_probability_density);
}

}  // namespace torchscience::kernel::probability
