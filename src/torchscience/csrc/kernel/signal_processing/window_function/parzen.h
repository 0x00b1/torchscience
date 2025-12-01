#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Parzen window (also called de la Vallee Poussin)
// Piecewise cubic polynomial (4th-order B-spline window)
//
// The formula uses position n relative to center, normalized by half-length:
//   For |n| <= (L-1)/4:
//     w[k] = 1 - 6*(n/(L/2))^2 + 6*(|n|/(L/2))^3
//   For (L-1)/4 < |n| <= L/2:
//     w[k] = 2*(1 - |n|/(L/2))^3
//
// where L is the effective length (n for symmetric, n+1 for periodic).
template<typename scalar_t>
inline scalar_t parzen(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }

  // L is the effective length for the formula
  // For symmetric (periodic=false): L = n
  // For periodic (periodic=true): L = n + 1 (we take first n points)
  scalar_t L = periodic ? scalar_t(n + 1) : scalar_t(n);

  // Position n relative to center: ranges from -(L-1)/2 to (L-1)/2
  scalar_t center = (L - scalar_t(1)) / scalar_t(2);
  scalar_t pos = scalar_t(i) - center;
  scalar_t abs_pos = std::abs(pos);

  // Half-length used for normalization
  scalar_t half = L / scalar_t(2);

  // Quarter boundary for piecewise definition
  scalar_t quarter = (L - scalar_t(1)) / scalar_t(4);

  // Normalized position
  scalar_t x = pos / half;
  scalar_t abs_x = abs_pos / half;

  if (abs_pos <= quarter) {
    // Inner region: 1 - 6*x^2 + 6*|x|^3
    return scalar_t(1) - scalar_t(6) * x * x + scalar_t(6) * abs_x * abs_x * abs_x;
  } else {
    // Outer region: 2*(1 - |x|)^3
    scalar_t y = scalar_t(1) - abs_x;
    return scalar_t(2) * y * y * y;
  }
}

}  // namespace torchscience::kernel::window_function
