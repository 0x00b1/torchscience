#pragma once

#include "digamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T log_multivariate_gamma_backward(T gradient, T a, int64_t d) {
  // d/da log(Gamma_d(a)) = sum_{j=1}^{d} psi(a + (1-j)/2)
  T grad_a = T(0);
  for (int64_t j = 1; j <= d; ++j) {
    grad_a += digamma(a + (T(1) - static_cast<T>(j)) / T(2));
  }
  return gradient * grad_a;
}

} // namespace torchscience::kernel::special_functions
