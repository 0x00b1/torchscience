// src/torchscience/csrc/kernel/signal_processing/waveform/frequency_modulated_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t frequency_modulated_wave_kernel(
    scalar_t t,
    scalar_t carrier_frequency,
    scalar_t modulator_frequency,
    scalar_t modulation_index,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // FM: y(t) = A * cos(2π*fc*t + β*sin(2π*fm*t) + φ)
  scalar_t modulator = std::sin(two_pi * modulator_frequency * t);
  scalar_t inst_phase = two_pi * carrier_frequency * t + modulation_index * modulator + phase;
  return amplitude * std::cos(inst_phase);
}

}  // namespace torchscience::kernel
