// src/torchscience/csrc/kernel/signal_processing/waveform/ramp_wave.h
#pragma once

namespace torchscience::kernel {

template <typename scalar_t>
inline scalar_t ramp_wave_kernel(int64_t i, int64_t position, scalar_t slope) {
  return (i >= position) ? slope * static_cast<scalar_t>(i - position) : scalar_t(0.0);
}

}  // namespace torchscience::kernel
