#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "../../../kernel/signal_processing/waveform/sine_wave_backward.h"

namespace torchscience::cpu::signal_processing::waveform {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sine_wave_backward(
    const at::Tensor& grad_output,
    c10::optional<int64_t> n,
    c10::optional<at::Tensor> t,
    const at::Tensor& frequency,
    double sample_rate,
    const at::Tensor& amplitude,
    const at::Tensor& phase) {

  // Reconstruct time tensor
  at::Tensor time;
  if (t.has_value()) {
    time = t.value();
  } else {
    time = at::arange(n.value(), grad_output.options()) / sample_rate;
  }

  int64_t n_samples = time.numel();
  auto broadcast_shape = at::infer_size(
      at::infer_size(frequency.sizes(), amplitude.sizes()),
      phase.sizes());

  int64_t batch_size = 1;
  for (auto s : broadcast_shape) batch_size *= s;

  auto freq_exp = frequency.expand(broadcast_shape).contiguous();
  auto amp_exp = amplitude.expand(broadcast_shape).contiguous();
  auto phase_exp = phase.expand(broadcast_shape).contiguous();

  // Gradient accumulators
  auto grad_t = t.has_value() ? at::zeros_like(time) : at::Tensor();
  auto grad_freq = at::zeros_like(freq_exp);
  auto grad_amp = at::zeros_like(amp_exp);
  auto grad_phase = at::zeros_like(phase_exp);

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "sine_wave_backward_cpu", [&] {
    auto time_data = time.data_ptr<scalar_t>();
    auto freq_data = freq_exp.data_ptr<scalar_t>();
    auto amp_data = amp_exp.data_ptr<scalar_t>();
    auto phase_data = phase_exp.data_ptr<scalar_t>();
    auto grad_out_data = grad_output.data_ptr<scalar_t>();

    auto grad_freq_data = grad_freq.data_ptr<scalar_t>();
    auto grad_amp_data = grad_amp.data_ptr<scalar_t>();
    auto grad_phase_data = grad_phase.data_ptr<scalar_t>();

    // Parallelize batch parameter gradients (no race condition)
    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t freq = freq_data[b];
        scalar_t amp = amp_data[b];
        scalar_t ph = phase_data[b];

        scalar_t acc_freq = 0, acc_amp = 0, acc_phase = 0;

        for (int64_t i = 0; i < n_samples; ++i) {
          scalar_t g_t, g_f, g_a, g_p;
          kernel::sine_wave_backward_kernel(
              time_data[i], freq, amp, ph,
              grad_out_data[b * n_samples + i],
              g_t, g_f, g_a, g_p);

          acc_freq += g_f;
          acc_amp += g_a;
          acc_phase += g_p;
        }

        grad_freq_data[b] = acc_freq;
        grad_amp_data[b] = acc_amp;
        grad_phase_data[b] = acc_phase;
      }
    });

    // Compute grad_t separately (parallelize over samples to avoid race condition)
    if (grad_t.defined()) {
      auto grad_t_data = grad_t.data_ptr<scalar_t>();
      at::parallel_for(0, n_samples, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          scalar_t acc_t = 0;
          for (int64_t b = 0; b < batch_size; ++b) {
            scalar_t g_t, g_f, g_a, g_p;
            kernel::sine_wave_backward_kernel(
                time_data[i], freq_data[b], amp_data[b], phase_data[b],
                grad_out_data[b * n_samples + i],
                g_t, g_f, g_a, g_p);
            acc_t += g_t;
          }
          grad_t_data[i] = acc_t;
        }
      });
    }
  });

  // Reduce gradients back to original shapes
  auto grad_freq_reduced = grad_freq.sum_to_size(frequency.sizes());
  auto grad_amp_reduced = grad_amp.sum_to_size(amplitude.sizes());
  auto grad_phase_reduced = grad_phase.sum_to_size(phase.sizes());

  return std::make_tuple(grad_t, grad_freq_reduced, grad_amp_reduced, grad_phase_reduced);
}

}  // namespace torchscience::cpu::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("sine_wave_backward", &torchscience::cpu::signal_processing::waveform::sine_wave_backward);
}
