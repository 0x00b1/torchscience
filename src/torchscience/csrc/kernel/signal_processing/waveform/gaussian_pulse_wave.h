// src/torchscience/csrc/kernel/signal_processing/waveform/gaussian_pulse_wave.h
#pragma once

#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t gaussian_pulse_wave_kernel(
    int64_t i,
    scalar_t center,
    scalar_t std,
    scalar_t amplitude) {
  scalar_t x = static_cast<scalar_t>(i) - center;
  scalar_t exponent = -scalar_t(0.5) * (x * x) / (std * std);
  return amplitude * std::exp(exponent);
}

}  // namespace torchscience::kernel
