// src/torchscience/csrc/kernel/signal_processing/waveform/cosine_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include "sine_wave.h"

namespace torchscience {
namespace kernel {

template <typename scalar_t>
inline scalar_t cosine_wave_kernel(
    scalar_t t,
    scalar_t frequency,
    scalar_t amplitude,
    scalar_t phase) {
  // cos(x) = sin(x + pi/2)
  return sine_wave_kernel(t, frequency, amplitude, phase + c10::pi<scalar_t> / 2);
}

}  // namespace kernel
}  // namespace torchscience
