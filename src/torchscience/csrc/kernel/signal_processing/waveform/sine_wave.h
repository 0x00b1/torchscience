// src/torchscience/csrc/kernel/signal_processing/waveform/sine_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience {
namespace kernel {

template <typename scalar_t>
inline scalar_t sine_wave_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  return amplitude * std::sin(two_pi * frequency * t + phase);
}

}  // namespace kernel
}  // namespace torchscience
