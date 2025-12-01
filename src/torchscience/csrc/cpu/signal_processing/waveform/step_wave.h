#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "../../../kernel/signal_processing/waveform/step_wave.h"

namespace torchscience::cpu::signal_processing::waveform {

inline at::Tensor step_wave(
    int64_t n,
    const at::Tensor& position,
    const at::Tensor& amplitude,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device) {

  TORCH_CHECK(n >= 0, "step_wave: n must be non-negative, got ", n);

  // Early return for n == 0
  if (n == 0) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    return at::empty({0}, options);
  }

  // Broadcast parameter shapes together
  auto broadcast_shape = at::infer_size(position.sizes(), amplitude.sizes());

  // Expand parameters to broadcast shape
  auto pos_exp = position.expand(broadcast_shape).contiguous();
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

  // Position tensor should be int64 for indexing
  auto pos_int = pos_exp.to(at::kLong);

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "step_wave_cpu", [&] {
    auto pos_data = pos_int.data_ptr<int64_t>();
    auto amp_data = amp_exp.to(out_dtype).data_ptr<scalar_t>();
    auto out_data = output.data_ptr<scalar_t>();

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        int64_t pos = pos_data[b];
        scalar_t amp = amp_data[b];

        for (int64_t i = 0; i < n; ++i) {
          out_data[b * n + i] = kernel::step_wave_kernel<scalar_t>(i, pos, amp);
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
  m.impl("step_wave", &torchscience::cpu::signal_processing::waveform::step_wave);
}
