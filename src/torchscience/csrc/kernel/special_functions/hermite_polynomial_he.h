#pragma once

#include <cmath>

#include "hermite_polynomial_h.h"

namespace torchscience::kernel::special_functions {

// Probabilists' Hermite polynomial He_n(z)
// He_n(z) = 2^(-n/2) * H_n(z/sqrt(2))
//
// This is a scaling transformation of the Physicists' Hermite polynomial.
// The probabilists' polynomials satisfy the recurrence:
//   He_{n+1}(z) = z * He_n(z) - n * He_{n-1}(z)
// with He_0(z) = 1, He_1(z) = z
//
// Applications: probability theory, Gaussian integrals, Edgeworth expansion
template <typename T>
T hermite_polynomial_he(T n, T z) {
  const T sqrt_2 = std::sqrt(T(2));
  T scale = std::pow(T(2), -n / T(2));
  return scale * hermite_polynomial_h(n, z / sqrt_2);
}

} // namespace torchscience::kernel::special_functions
