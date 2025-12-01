#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "../../../kernel/signal_processing/waveform/triangle_wave.h"

namespace torchscience::cpu::signal_processing::waveform {

inline at::Tensor triangle_wave(
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
      "triangle_wave: Exactly one of n or t must be provided");

  if (n.has_value()) {
    TORCH_CHECK(n.value() >= 0, "triangle_wave: n must be non-negative");
  }

  if (n.has_value() && n.value() == 0) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    return at::empty({0}, options);
  }

  at::Tensor time;
  if (t.has_value()) {
    time = t.value();
  } else {
    TORCH_CHECK(sample_rate > 0, "triangle_wave: sample_rate must be positive");
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    time = at::arange(n.value(), options) / sample_rate;
  }

  int64_t n_samples = time.numel();

  auto broadcast_shape = at::infer_size(
      at::infer_size(frequency.sizes(), amplitude.sizes()),
      phase.sizes());

  auto freq_exp = frequency.expand(broadcast_shape).contiguous();
  auto amp_exp = amplitude.expand(broadcast_shape).contiguous();
  auto phase_exp = phase.expand(broadcast_shape).contiguous();

  std::vector<int64_t> out_shape(broadcast_shape.begin(), broadcast_shape.end());
  out_shape.push_back(n_samples);

  auto out_dtype = dtype.value_or(time.scalar_type());
  auto options = at::TensorOptions().dtype(out_dtype).device(time.device());
  auto output = at::empty(out_shape, options);

  int64_t batch_size = 1;
  for (auto s : broadcast_shape) batch_size *= s;

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "triangle_wave_cpu", [&] {
    auto time_data = time.data_ptr<scalar_t>();
    auto freq_data = freq_exp.data_ptr<scalar_t>();
    auto amp_data = amp_exp.data_ptr<scalar_t>();
    auto phase_data = phase_exp.data_ptr<scalar_t>();
    auto out_data = output.data_ptr<scalar_t>();

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t freq = freq_data[b];
        scalar_t amp = amp_data[b];
        scalar_t ph = phase_data[b];

        for (int64_t i = 0; i < n_samples; ++i) {
          out_data[b * n_samples + i] = kernel::triangle_wave_kernel(
              time_data[i], freq, amp, ph);
        }
      }
    });
  });

  return output;
}

}  // namespace torchscience::cpu::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("triangle_wave", &torchscience::cpu::signal_processing::waveform::triangle_wave);
}
