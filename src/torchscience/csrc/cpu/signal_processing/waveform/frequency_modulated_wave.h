#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include "../../../kernel/signal_processing/waveform/frequency_modulated_wave.h"

namespace torchscience::cpu::signal_processing::waveform {

// Sinusoidal FM: uses modulator_frequency to generate internal modulating signal
inline at::Tensor frequency_modulated_wave(
    c10::optional<int64_t> n,
    c10::optional<at::Tensor> t,
    const at::Tensor& carrier_frequency,
    const at::Tensor& modulator_frequency,
    const at::Tensor& modulation_index,
    double sample_rate,
    const at::Tensor& amplitude,
    const at::Tensor& phase,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device) {

  // Validate input
  TORCH_CHECK(
      n.has_value() != t.has_value(),
      "frequency_modulated_wave: Exactly one of n or t must be provided");

  if (n.has_value()) {
    TORCH_CHECK(n.value() >= 0, "frequency_modulated_wave: n must be non-negative, got ", n.value());
  }

  // Early return for n == 0
  if (n.has_value() && n.value() == 0) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    return at::empty({0}, options);
  }

  // Determine time tensor
  at::Tensor time;
  if (t.has_value()) {
    time = t.value();
  } else {
    TORCH_CHECK(sample_rate > 0, "frequency_modulated_wave: sample_rate must be positive");
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    time = at::arange(n.value(), options) / sample_rate;
  }

  int64_t n_samples = time.numel();

  // Broadcast parameter shapes together
  auto broadcast_shape = at::infer_size(
      at::infer_size(
          at::infer_size(
              at::infer_size(carrier_frequency.sizes(), modulator_frequency.sizes()),
              modulation_index.sizes()),
          amplitude.sizes()),
      phase.sizes());

  // Expand parameters to broadcast shape
  auto fc_exp = carrier_frequency.expand(broadcast_shape).contiguous();
  auto fm_exp = modulator_frequency.expand(broadcast_shape).contiguous();
  auto beta_exp = modulation_index.expand(broadcast_shape).contiguous();
  auto amp_exp = amplitude.expand(broadcast_shape).contiguous();
  auto phase_exp = phase.expand(broadcast_shape).contiguous();

  // Output shape: (*broadcast_shape, n_samples)
  std::vector<int64_t> out_shape(broadcast_shape.begin(), broadcast_shape.end());
  out_shape.push_back(n_samples);

  auto out_dtype = dtype.value_or(time.scalar_type());
  auto options = at::TensorOptions().dtype(out_dtype).device(time.device());
  auto output = at::empty(out_shape, options);

  // Flatten batch dimensions for iteration
  int64_t batch_size = 1;
  for (auto s : broadcast_shape) batch_size *= s;

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "frequency_modulated_wave_cpu", [&] {
    auto time_data = time.data_ptr<scalar_t>();
    auto fc_data = fc_exp.data_ptr<scalar_t>();
    auto fm_data = fm_exp.data_ptr<scalar_t>();
    auto beta_data = beta_exp.data_ptr<scalar_t>();
    auto amp_data = amp_exp.data_ptr<scalar_t>();
    auto phase_data = phase_exp.data_ptr<scalar_t>();
    auto out_data = output.data_ptr<scalar_t>();

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t fc = fc_data[b];
        scalar_t fm = fm_data[b];
        scalar_t beta = beta_data[b];
        scalar_t amp = amp_data[b];
        scalar_t ph = phase_data[b];

        for (int64_t i = 0; i < n_samples; ++i) {
          out_data[b * n_samples + i] = kernel::frequency_modulated_wave_kernel(
              time_data[i], fc, fm, beta, amp, ph);
        }
      }
    });
  });

  return output;
}

// Arbitrary FM: uses modulating_signal tensor (cumsum integrated) as modulator
inline at::Tensor frequency_modulated_wave_arbitrary(
    c10::optional<int64_t> n,
    c10::optional<at::Tensor> t,
    const at::Tensor& carrier_frequency,
    const at::Tensor& modulating_signal,
    const at::Tensor& modulation_index,
    double sample_rate,
    const at::Tensor& amplitude,
    const at::Tensor& phase,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device) {

  // For arbitrary modulation, we need to integrate the modulating signal
  // The formula is: y(t) = A * cos(2π*fc*t + β*integral(m(t)) + φ)
  // We approximate the integral using cumulative sum scaled by sample period

  // Validate input
  TORCH_CHECK(
      n.has_value() != t.has_value(),
      "frequency_modulated_wave_arbitrary: Exactly one of n or t must be provided");

  if (n.has_value()) {
    TORCH_CHECK(n.value() >= 0, "frequency_modulated_wave_arbitrary: n must be non-negative, got ", n.value());
  }

  // Early return for n == 0
  if (n.has_value() && n.value() == 0) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    return at::empty({0}, options);
  }

  // Determine time tensor
  at::Tensor time;
  double dt;
  if (t.has_value()) {
    time = t.value();
    // Estimate sample period from time tensor (assume uniform spacing)
    if (time.numel() > 1) {
      dt = (time[-1].item<double>() - time[0].item<double>()) / (time.numel() - 1);
    } else {
      dt = 1.0;  // Default for single sample
    }
  } else {
    TORCH_CHECK(sample_rate > 0, "frequency_modulated_wave_arbitrary: sample_rate must be positive");
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .device(device.value_or(at::kCPU));
    time = at::arange(n.value(), options) / sample_rate;
    dt = 1.0 / sample_rate;
  }

  int64_t n_samples = time.numel();
  TORCH_CHECK(modulating_signal.size(-1) == n_samples,
              "frequency_modulated_wave_arbitrary: modulating_signal last dimension must match time length, got ",
              modulating_signal.size(-1), " vs ", n_samples);

  // Broadcast parameter shapes together (excluding the time dimension of modulating_signal)
  // modulating_signal shape could be (..., n_samples)
  auto mod_batch_shape = modulating_signal.sizes().vec();
  mod_batch_shape.pop_back();  // Remove time dimension

  auto broadcast_shape = at::infer_size(
      at::infer_size(
          at::infer_size(
              at::infer_size(carrier_frequency.sizes(), at::IntArrayRef(mod_batch_shape)),
              modulation_index.sizes()),
          amplitude.sizes()),
      phase.sizes());

  // Expand parameters to broadcast shape
  auto fc_exp = carrier_frequency.expand(broadcast_shape).contiguous();
  auto beta_exp = modulation_index.expand(broadcast_shape).contiguous();
  auto amp_exp = amplitude.expand(broadcast_shape).contiguous();
  auto phase_exp = phase.expand(broadcast_shape).contiguous();

  // Expand modulating_signal to (*broadcast_shape, n_samples)
  std::vector<int64_t> mod_exp_shape(broadcast_shape.begin(), broadcast_shape.end());
  mod_exp_shape.push_back(n_samples);
  auto mod_exp = modulating_signal.expand(mod_exp_shape).contiguous();

  // Output shape: (*broadcast_shape, n_samples)
  std::vector<int64_t> out_shape(broadcast_shape.begin(), broadcast_shape.end());
  out_shape.push_back(n_samples);

  auto out_dtype = dtype.value_or(time.scalar_type());
  auto options = at::TensorOptions().dtype(out_dtype).device(time.device());
  auto output = at::empty(out_shape, options);

  // Flatten batch dimensions for iteration
  int64_t batch_size = 1;
  for (auto s : broadcast_shape) batch_size *= s;

  constexpr double two_pi = 2 * c10::pi<double>;

  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "frequency_modulated_wave_arbitrary_cpu", [&] {
    auto time_data = time.data_ptr<scalar_t>();
    auto fc_data = fc_exp.data_ptr<scalar_t>();
    auto beta_data = beta_exp.data_ptr<scalar_t>();
    auto amp_data = amp_exp.data_ptr<scalar_t>();
    auto phase_data = phase_exp.data_ptr<scalar_t>();
    auto mod_data = mod_exp.data_ptr<scalar_t>();
    auto out_data = output.data_ptr<scalar_t>();

    scalar_t dt_val = static_cast<scalar_t>(dt);
    scalar_t two_pi_val = static_cast<scalar_t>(two_pi);

    at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        scalar_t fc = fc_data[b];
        scalar_t beta = beta_data[b];
        scalar_t amp = amp_data[b];
        scalar_t ph = phase_data[b];

        // Integrate modulating signal using cumulative sum (trapezoidal-ish)
        scalar_t integral = 0;
        for (int64_t i = 0; i < n_samples; ++i) {
          scalar_t t_val = time_data[i];
          scalar_t m_val = mod_data[b * n_samples + i];

          // FM: y(t) = A * cos(2π*fc*t + β*integral(m(τ))dτ + φ)
          scalar_t inst_phase = two_pi_val * fc * t_val + beta * integral + ph;
          out_data[b * n_samples + i] = amp * std::cos(inst_phase);

          // Update integral for next sample
          integral += m_val * dt_val;
        }
      }
    });
  });

  return output;
}

}  // namespace torchscience::cpu::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("frequency_modulated_wave", &torchscience::cpu::signal_processing::waveform::frequency_modulated_wave);
  m.impl("frequency_modulated_wave_arbitrary", &torchscience::cpu::signal_processing::waveform::frequency_modulated_wave_arbitrary);
}
