// src/torchscience/csrc/kernel/signal_processing/waveform/logarithmic_chirp_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t logarithmic_chirp_wave_kernel(
    scalar_t t,
    scalar_t f0,
    scalar_t f1,
    scalar_t t1,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // Logarithmic chirp: f(t) = f0 * (f1/f0)^(t/t1)
  // phase(t) = 2π * f0 * t1 * (k^(t/t1) - 1) / ln(k) where k = f1/f0
  scalar_t k = f1 / f0;
  scalar_t log_k = std::log(k);
  scalar_t inst_phase = two_pi * f0 * t1 * (std::pow(k, t / t1) - scalar_t(1.0)) / log_k + phase;
  return amplitude * std::cos(inst_phase);
}

}  // namespace torchscience::kernel
