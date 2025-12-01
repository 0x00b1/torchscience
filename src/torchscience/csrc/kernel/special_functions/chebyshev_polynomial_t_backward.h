#pragma once

#include <cmath>
#include <tuple>

#include "chebyshev_polynomial_u.h"

namespace torchscience::kernel::special_functions {

// Backward for Chebyshev polynomial of the first kind T_n(x)
//
// dT_n/dx = n * U_{n-1}(x) where U is Chebyshev of second kind
//
// dT_n/dn:
//   For |x| <= 1: dT/dn = -sin(n * acos(x)) * acos(x)
//   For x > 1:    dT/dn = sinh(n * acosh(x)) * acosh(x)
//   For x < -1:   dT/dn = -sin(n*pi)*pi * cosh(n * acosh(-x))
//                       + cos(n*pi) * sinh(n * acosh(-x)) * acosh(-x)
template <typename T>
std::tuple<T, T> chebyshev_polynomial_t_backward(T gradient, T x, T n) {
  const T pi = T(3.14159265358979323846);

  T gradient_x;
  T gradient_n;

  if (std::abs(x) <= T(1)) {
    // Standard domain
    if (n > T(0)) {
      gradient_x = gradient * n * chebyshev_polynomial_u(x, n - T(1));
    } else {
      gradient_x = T(0);
    }

    T theta = std::acos(x);
    gradient_n = gradient * (-std::sin(n * theta) * theta);
  } else if (x > T(1)) {
    // Hyperbolic domain x > 1
    T eta = std::acosh(x);
    T sinh_val = std::sinh(n * eta);
    T cosh_val = std::cosh(n * eta);

    // dT/dx = n * sinh(n * acosh(x)) / sqrt(x^2 - 1)
    T sqrt_x2_minus_1 = std::sqrt(x * x - T(1));
    gradient_x = gradient * n * sinh_val / sqrt_x2_minus_1;

    // dT/dn = sinh(n * acosh(x)) * acosh(x)
    gradient_n = gradient * sinh_val * eta;
  } else {
    // Hyperbolic domain x < -1
    T eta = std::acosh(-x);
    T sinh_val = std::sinh(n * eta);
    T cosh_val = std::cosh(n * eta);
    T cos_n_pi = std::cos(n * pi);
    T sin_n_pi = std::sin(n * pi);

    // dT/dx = -cos(n*pi) * n * sinh(n * acosh(-x)) / sqrt(x^2 - 1)
    T sqrt_x2_minus_1 = std::sqrt(x * x - T(1));
    gradient_x = gradient * (-cos_n_pi * n * sinh_val / sqrt_x2_minus_1);

    // dT/dn = -sin(n*pi)*pi * cosh(...) + cos(n*pi) * sinh(...) * acosh(-x)
    gradient_n = gradient * (-sin_n_pi * pi * cosh_val + cos_n_pi * sinh_val * eta);
  }

  return {gradient_x, gradient_n};
}

} // namespace torchscience::kernel::special_functions
