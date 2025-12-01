// src/torchscience/csrc/kernel/signal_processing/waveform/hyperbolic_chirp_wave.h
#pragma once

#include <c10/util/MathConstants.h>
#include <cmath>

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t hyperbolic_chirp_wave_kernel(
    scalar_t t,
    scalar_t f0,
    scalar_t f1,
    scalar_t t1,
    scalar_t amplitude,
    scalar_t phase) {
  constexpr scalar_t two_pi = 2 * c10::pi<scalar_t>;
  // Hyperbolic chirp (matches scipy.signal.chirp with method='hyperbolic'):
  // f(t) = alpha / (beta * t + gamma) where alpha = f0*f1*t1, beta = f0-f1, gamma = f1*t1
  // sing = -f1 * t1 / (f0 - f1) = f1 * t1 / (f1 - f0)
  // phase(t) = 2π * f0 * f1 * t1 / (f1 - f0) * ln(|1 - t * (f1 - f0) / (f1 * t1)|)
  scalar_t df = f1 - f0;
  scalar_t sing = f1 * t1 / df;  // singular point
  scalar_t inst_phase = two_pi * f0 * sing * std::log(std::abs(scalar_t(1.0) - t / sing)) + phase;
  return amplitude * std::cos(inst_phase);
}

}  // namespace torchscience::kernel
