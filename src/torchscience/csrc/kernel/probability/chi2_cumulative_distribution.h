#pragma once

#include "../special_functions/regularized_gamma_p.h"

namespace torchscience::kernel::probability {

// Chi-squared CDF: P(k/2, x/2) where P is the regularized lower incomplete gamma
// F(x; k) = P(k/2, x/2) = gamma(k/2, x/2) / Gamma(k/2)
template <typename T>
T chi2_cumulative_distribution(T x, T df) {
  if (x <= T(0)) return T(0);
  return special_functions::regularized_gamma_p(df / T(2), x / T(2));
}

}  // namespace torchscience::kernel::probability
