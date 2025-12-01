#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

// Gradients for sine_wave:
// y = A * sin(2π * f * t + φ)
// dy/dt = A * 2π * f * cos(2π * f * t + φ)
// dy/df = A * 2π * t * cos(2π * f * t + φ)
// dy/dA = sin(2π * f * t + φ)
// dy/dφ = A * cos(2π * f * t + φ)

template <typename scalar_t>
inline void sine_wave_backward_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase,
    scalar_t grad_out,
    scalar_t& grad_t,
    scalar_t& grad_frequency,
    scalar_t& grad_amplitude,
    scalar_t& grad_phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  scalar_t arg = two_pi * frequency * t + phase;
  scalar_t cos_arg = std::cos(arg);
  scalar_t sin_arg = std::sin(arg);

  grad_t = grad_out * amplitude * two_pi * frequency * cos_arg;
  grad_frequency = grad_out * amplitude * two_pi * t * cos_arg;
  grad_amplitude = grad_out * sin_arg;
  grad_phase = grad_out * amplitude * cos_arg;
}

}  // namespace torchscience::kernel
