#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Triangular window (different from Bartlett - does not touch zero at endpoints)
//
// For symmetric (periodic=false):
//   n odd:  w[k] = 2*(k+1)/(n+1)         for k < n/2
//           w[k] = 2*(n-k)/(n+1)         for k >= n/2
//   n even: w[k] = (2*k+1)/n             for k < n/2
//           w[k] = (2*(n-k)-1)/n         for k >= n/2
//
// For periodic (periodic=true):
//   Same formula but using n+1 points and taking first n
//
// Unlike Bartlett which has zero endpoints, triangular window has non-zero endpoints.
template<typename scalar_t>
inline scalar_t triangular(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }

  // For periodic window, compute as if length n+1 symmetric window
  int64_t L = periodic ? n + 1 : n;

  if (L % 2 == 1) {
    // Odd effective length: endpoints are 2/(L+1)
    scalar_t half_L = scalar_t(L + 1) / scalar_t(2);
    if (i < L / 2 + 1) {
      // First half (including center for odd)
      return scalar_t(i + 1) / half_L;
    } else {
      // Second half
      return scalar_t(L - i) / half_L;
    }
  } else {
    // Even effective length: endpoints are 1/L
    scalar_t half = scalar_t(L) / scalar_t(2);
    if (i < L / 2) {
      // First half
      return (scalar_t(2 * i + 1)) / scalar_t(L);
    } else {
      // Second half
      return (scalar_t(2 * (L - i) - 1)) / scalar_t(L);
    }
  }
}

}  // namespace torchscience::kernel::window_function
