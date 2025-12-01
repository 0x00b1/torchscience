#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::signal_processing::waveform {

inline at::Tensor sine_wave(
    c10::optional<int64_t> n,
    c10::optional<at::Tensor> t,
    const at::Tensor& frequency,
    double sample_rate,
    const at::Tensor& amplitude,
    const at::Tensor& phase,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device) {

  TORCH_CHECK(
      n.has_value() != t.has_value(),
      "sine_wave: Exactly one of n or t must be provided");

  if (n.has_value()) {
    TORCH_CHECK(n.value() >= 0, "sine_wave: n must be non-negative, got ", n.value());
  }

  int64_t n_samples = t.has_value() ? t.value().numel() : n.value();

  // Broadcast parameter shapes
  auto broadcast_shape = at::infer_size(
      at::infer_size(frequency.sizes(), amplitude.sizes()),
      phase.sizes());

  // Output shape: (*broadcast_shape, n_samples)
  std::vector<int64_t> out_shape(broadcast_shape.begin(), broadcast_shape.end());
  out_shape.push_back(n_samples);

  auto out_dtype = dtype.value_or(t.has_value() ? t.value().scalar_type() : at::kFloat);
  return at::empty(out_shape, frequency.options().dtype(out_dtype).device(at::kMeta));
}

}  // namespace torchscience::meta::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("sine_wave", &torchscience::meta::signal_processing::waveform::sine_wave);
}
