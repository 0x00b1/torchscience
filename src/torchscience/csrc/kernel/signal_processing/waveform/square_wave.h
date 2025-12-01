// src/torchscience/csrc/kernel/signal_processing/waveform/square_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience {
namespace kernel {

// Hardcoded sharpness for differentiable square wave.
// Value of 100.0 provides near-ideal edges while remaining differentiable.
constexpr double kSquareWaveSharpness = 100.0;

template <typename scalar_t>
inline scalar_t square_wave_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase,
    scalar_t duty) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  const scalar_t sharpness = static_cast<scalar_t>(kSquareWaveSharpness);

  // Normalized phase in [0, 1) for each cycle
  scalar_t raw_phi = t * frequency + phase / two_pi;
  scalar_t phi = raw_phi - std::floor(raw_phi);

  // Smooth square wave using difference of sigmoids
  scalar_t rising = scalar_t(1.0) / (scalar_t(1.0) + std::exp(-sharpness * phi));
  scalar_t falling = scalar_t(1.0) / (scalar_t(1.0) + std::exp(-sharpness * (phi - duty)));

  // Output: +amplitude when phi < duty, -amplitude otherwise
  return amplitude * (scalar_t(2.0) * (rising - falling) - scalar_t(1.0));
}

}  // namespace kernel
}  // namespace torchscience
