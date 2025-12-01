// src/torchscience/csrc/kernel/signal_processing/waveform/sinc_pulse_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t sinc_pulse_wave_kernel(
    int64_t i,
    scalar_t center,
    scalar_t bandwidth,
    scalar_t amplitude) {
  scalar_t x = bandwidth * (static_cast<scalar_t>(i) - center);
  if (std::abs(x) < scalar_t(1e-10)) {
    return amplitude;
  }
  scalar_t pi_x = c10::pi<scalar_t> * x;
  return amplitude * std::sin(pi_x) / pi_x;
}

}  // namespace torchscience::kernel
