#pragma once

#include "../special_functions/regularized_gamma_q.h"

namespace torchscience::kernel::probability {

// Chi-squared survival function (1 - CDF): Q(k/2, x/2)
// SF(x; k) = Q(k/2, x/2) = 1 - P(k/2, x/2)
template <typename T>
T chi2_survival(T x, T df) {
  if (x <= T(0)) return T(1);
  return special_functions::regularized_gamma_q(df / T(2), x / T(2));
}

}  // namespace torchscience::kernel::probability
