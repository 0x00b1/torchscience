#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience {
namespace kernel {

template <typename scalar_t>
inline scalar_t sawtooth_wave_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // Normalized phase in [0, 1) for each cycle
  scalar_t raw_phi = t * frequency + phase / two_pi;
  scalar_t phi = raw_phi - std::floor(raw_phi);
  // Sawtooth: linear from -1 to +1 over one period
  return amplitude * (scalar_t(2.0) * phi - scalar_t(1.0));
}

}  // namespace kernel
}  // namespace torchscience
