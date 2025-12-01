// src/torchscience/csrc/kernel/signal_processing/waveform/impulse_wave.h
#pragma once

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t impulse_wave_kernel(int64_t i, int64_t position, scalar_t amplitude) {
  return (i == position) ? amplitude : scalar_t(0.0);
}

}  // namespace torchscience::kernel
