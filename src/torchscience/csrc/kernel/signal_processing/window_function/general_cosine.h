#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// General cosine window: w[k] = sum_{j=0}^{M-1} (-1)^j * a_j * cos(2*pi*j*k / denom)
template<typename scalar_t>
inline scalar_t general_cosine(
  int64_t i,
  int64_t n,
  const scalar_t* coeffs,
  int64_t num_coeffs,
  bool periodic
) {
  // Convention: single-point windows have no windowing effect
  if (n == 1) {
    (void)i;
    (void)coeffs;
    (void)num_coeffs;
    (void)periodic;
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    scalar_t result = scalar_t(0);
    scalar_t sign = scalar_t(1);
    for (int64_t j = 0; j < num_coeffs; ++j) {
      result += sign * coeffs[j];
      sign = -sign;
    }
    return result;
  }

  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  scalar_t result = scalar_t(0);
  scalar_t sign = scalar_t(1);
  for (int64_t j = 0; j < num_coeffs; ++j) {
    result += sign * coeffs[j] * std::cos(scalar_t(j) * x);
    sign = -sign;
  }
  return result;
}

// Gradient w.r.t. coefficient j: d/d(a_j) = (-1)^j * cos(j * x)
template<typename scalar_t>
inline scalar_t general_cosine_backward_coeff(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  int64_t coeff_index,
  bool periodic
) {
  if (n == 1) {
    scalar_t sign = (coeff_index % 2 == 0) ? scalar_t(1) : scalar_t(-1);
    return grad_out * sign;
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    scalar_t sign = (coeff_index % 2 == 0) ? scalar_t(1) : scalar_t(-1);
    return grad_out * sign;
  }

  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  scalar_t sign = (coeff_index % 2 == 0) ? scalar_t(1) : scalar_t(-1);
  return grad_out * sign * std::cos(scalar_t(coeff_index) * x);
}

}  // namespace torchscience::kernel::window_function
