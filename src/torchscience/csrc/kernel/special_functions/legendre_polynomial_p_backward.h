#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

// Backward for Legendre polynomial P_n(z)
//
// dP/dz: Using chain rule on P_n(z) = 2F1(-n, n+1; 1; w) where w = (1-z)/2
//   dP/dz = dP/dw * dw/dz = dP/dw * (-1/2)
//   d/dw 2F1(a,b;c;w) = (a*b/c) * 2F1(a+1,b+1;c+1;w)
//   So dP/dz = (-n)(n+1)/1 * 2F1(-n+1, n+2; 2; w) * (-1/2)
//            = n(n+1)/2 * 2F1(1-n, n+2; 2; w)
//
// dP/dn: Computed via finite differences since the analytical formula involves
//   derivatives of hypergeometric functions with respect to parameters
template <typename T>
std::tuple<T, T> legendre_polynomial_p_backward(T gradient, T n, T z) {
  T c = T(1);
  T w = (T(1) - z) / T(2);

  // dP/dz via hypergeometric derivative formula
  T a = -n;
  T b = n + T(1);
  T dF_dw_coeff = (a * b) / c;
  T F_deriv = hypergeometric_2_f_1(a + T(1), b + T(1), c + T(1), w);
  T dw_dz = T(-0.5);
  T gradient_z = gradient * dF_dw_coeff * F_deriv * dw_dz;

  // dP/dn via finite differences
  T eps = T(1e-7);
  T P_plus = hypergeometric_2_f_1(-(n + eps), (n + eps) + T(1), c, w);
  T P_minus = hypergeometric_2_f_1(-(n - eps), (n - eps) + T(1), c, w);
  T gradient_n = gradient * (P_plus - P_minus) / (T(2) * eps);

  return {gradient_n, gradient_z};
}

} // namespace torchscience::kernel::special_functions
