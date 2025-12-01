// src/torchscience/csrc/kernel/signal_processing/waveform/linear_chirp_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t linear_chirp_wave_kernel(
    scalar_t t,
    scalar_t f0,
    scalar_t f1,
    scalar_t t1,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // Linear chirp: f(t) = f0 + (f1-f0)*t/t1
  // phase(t) = 2π * (f0*t + (f1-f0)*t²/(2*t1))
  scalar_t k = (f1 - f0) / t1;
  scalar_t inst_phase = two_pi * (f0 * t + scalar_t(0.5) * k * t * t) + phase;
  return amplitude * std::cos(inst_phase);
}

}  // namespace torchscience::kernel
