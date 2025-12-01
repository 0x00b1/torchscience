#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "../../../kernel/signal_processing/waveform/sinc_pulse_wave.h"

namespace torchscience::cpu::signal_processing::waveform {

inline at::Tensor sinc_pulse_wave(
    int64_t n,
    const at::Tensor& center,
    const at::Tensor& bandwidth,
    const at::Tensor& amplitude,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device) {

  TORCH_CHECK(n >= 0, "sinc_pulse_wave: n must be non-negative, got ", n);

  // Early return for n == 0
  if (n == 0) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    return at::empty({0}, options);
  }

  // Broadcast parameter shapes together
  auto broadcast_shape = at::infer_size(center.sizes(), bandwidth.sizes());
  broadcast_shape = at::infer_size(broadcast_shape, amplitude.sizes());

  // Expand parameters to broadcast shape
  auto center_exp = center.expand(broadcast_shape).contiguous();
  auto bandwidth_exp = bandwidth.expand(broadcast_shape).contiguous();
  auto amp_exp = amplitude.expand(broadcast_shape).contiguous();

  // Output shape: (*broadcast_shape, n)
  std::vector<int64_t> out_shape(broadcast_shape.begin(), broadcast_shape.end());
  out_shape.push_back(n);

  auto out_dtype = dtype.value_or(amp_exp.scalar_type());
  auto options = at::TensorOptions()
      .dtype(out_dtype)
      .device(device.value_or(at::kCPU));
  auto output = at::zeros(out_shape, options);

  // Flatten batch dimensions for iteration
  int64_t batch_size = 1;
  for (auto s : broadcast_shape) batch_size *= s;

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "sinc_pulse_wave_cpu", [&] {
    auto center_data = center_exp.to(out_dtype).data_ptr<scalar_t>();
    auto bandwidth_data = bandwidth_exp.to(out_dtype).data_ptr<scalar_t>();
    auto amp_data = amp_exp.to(out_dtype).data_ptr<scalar_t>();
    auto out_data = output.data_ptr<scalar_t>();

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t ctr = center_data[b];
        scalar_t bw = bandwidth_data[b];
        scalar_t amp = amp_data[b];

        for (int64_t i = 0; i < n; ++i) {
          out_data[b * n + i] = kernel::sinc_pulse_wave_kernel<scalar_t>(i, ctr, bw, amp);
        }
      }
    });
  });

  // If batch_size is 1 and broadcast_shape is empty, squeeze the output
  if (broadcast_shape.empty()) {
    output = output.squeeze(0);
  }

  return output;
}

}  // namespace torchscience::cpu::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("sinc_pulse_wave", &torchscience::cpu::signal_processing::waveform::sinc_pulse_wave);
}
