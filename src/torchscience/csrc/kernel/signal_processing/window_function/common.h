#pragma once

#include <cmath>
#include <c10/macros/Macros.h>

namespace torchscience::kernel::window_function {

// Computes the denominator used for window index normalization.
// periodic: denominator is n (for FFT/STFT compatibility)
// symmetric: denominator is n-1 (for filter design)
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t window_denominator(int64_t n, bool periodic) {
  return periodic ? static_cast<scalar_t>(n) : static_cast<scalar_t>(n - 1);
}

}  // namespace torchscience::kernel::window_function
