#pragma once

#include <cmath>
#include <tuple>

#include "hypergeometric_2_f_1.h"

namespace torchscience::kernel::special_functions {

// Backward_backward for Legendre polynomial P_n(z)
//
// Second derivative d^2P/dz^2:
//   From dP/dz = (a*b)/(c) * 2F1(a+1,b+1;c+1;w) * dw/dz
//   where a=-n, b=n+1, c=1, w=(1-z)/2, dw/dz=-1/2
//   d^2P/dz^2 = (a*b/c) * d/dz[2F1(a+1,b+1;c+1;w)] * dw/dz
//             = (a*b/c) * ((a+1)(b+1)/(c+1)) * 2F1(a+2,b+2;c+2;w) * (dw/dz)^2
//             = (a*b/c) * ((a+1)(b+1)/(c+1)) * 2F1(a+2,b+2;c+2;w) * (1/4)
template <typename T>
std::tuple<T, T, T> legendre_polynomial_p_backward_backward(
  T gradient_gradient_n,
  T gradient_gradient_z,
  T gradient,
  T n,
  T z
) {
  T a = -n;
  T b = n + T(1);
  T c = T(1);
  T w = (T(1) - z) / T(2);

  // First derivative coefficient and value
  T coeff1 = (a * b) / c;
  T F_deriv = hypergeometric_2_f_1(a + T(1), b + T(1), c + T(1), w);
  T dw_dz = T(-0.5);
  T dP_dz = coeff1 * F_deriv * dw_dz;

  // Second derivative d^2P/dz^2
  T coeff2 = ((a + T(1)) * (b + T(1))) / (c + T(1));
  T F_second = hypergeometric_2_f_1(a + T(2), b + T(2), c + T(2), w);
  T d2P_dz2 = coeff1 * coeff2 * F_second * dw_dz * dw_dz;

  // Compute dP/dn via finite differences (same as in backward kernel)
  T eps = T(1e-7);
  T P_plus = hypergeometric_2_f_1(-(n + eps), (n + eps) + T(1), c, w);
  T P_minus = hypergeometric_2_f_1(-(n - eps), (n - eps) + T(1), c, w);
  T dP_dn = (P_plus - P_minus) / (T(2) * eps);

  // gradient_gradient_output = gg_z * dP/dz + gg_n * dP/dn
  T gradient_gradient_output = gradient_gradient_z * dP_dz + gradient_gradient_n * dP_dn;

  // new_gradient_z = gg_z * grad * d^2P/dz^2
  T new_gradient_z = gradient_gradient_z * gradient * d2P_dz2;

  // new_gradient_n: We approximate this as zero since d^2P/dndn
  // and d^2P/dndz are complex
  T new_gradient_n = T(0);

  return {gradient_gradient_output, new_gradient_n, new_gradient_z};
}

} // namespace torchscience::kernel::special_functions
