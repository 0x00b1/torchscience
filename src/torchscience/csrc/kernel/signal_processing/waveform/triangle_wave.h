#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience {
namespace kernel {

template <typename scalar_t>
inline scalar_t triangle_wave_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // Normalized phase in [0, 1) for each cycle
  scalar_t raw_phi = t * frequency + phase / two_pi;
  scalar_t phi = raw_phi - std::floor(raw_phi);
  // Triangle: rises from -1 to +1 in first half, falls from +1 to -1 in second half
  scalar_t tri = phi < scalar_t(0.5)
      ? scalar_t(4.0) * phi - scalar_t(1.0)
      : scalar_t(3.0) - scalar_t(4.0) * phi;
  return amplitude * tri;
}

}  // namespace kernel
}  // namespace torchscience
