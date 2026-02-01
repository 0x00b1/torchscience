#pragma once

#include <cmath>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

// Legendre polynomial P_n(z)
// P_n(z) = 2F1(-n, n+1; 1; (1-z)/2)
template <typename T>
T legendre_polynomial_p(T n, T z) {
  T a = -n;
  T b = n + T(1);
  T c = T(1);
  T w = (T(1) - z) / T(2);
  return hypergeometric_2_f_1(a, b, c, w);
}

} // namespace torchscience::kernel::special_functions
