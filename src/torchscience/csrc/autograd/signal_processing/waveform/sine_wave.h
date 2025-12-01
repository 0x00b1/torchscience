#pragma once

#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::signal_processing::waveform {

class SineWaveFunction : public torch::autograd::Function<SineWaveFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      c10::optional<int64_t> n,
      c10::optional<at::Tensor> t,
      const at::Tensor& frequency,
      double sample_rate,
      const at::Tensor& amplitude,
      const at::Tensor& phase,
      c10::optional<at::ScalarType> dtype,
      c10::optional<at::Layout> layout,
      c10::optional<at::Device> device) {

    at::AutoDispatchBelowAutograd guard;
    auto output = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::sine_wave", "")
        .typed<at::Tensor(c10::optional<int64_t>, c10::optional<at::Tensor>, const at::Tensor&, double, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>)>()
        .call(n, t, frequency, sample_rate, amplitude, phase, dtype, layout, device);

    // Save for backward
    ctx->saved_data["n"] = n;
    ctx->saved_data["sample_rate"] = sample_rate;
    ctx->save_for_backward({
        t.has_value() ? t.value() : at::Tensor(),
        frequency, amplitude, phase, output
    });

    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {

    auto saved = ctx->get_saved_variables();
    auto t_saved = saved[0];
    auto frequency = saved[1];
    auto amplitude = saved[2];
    auto phase = saved[3];
    auto output = saved[4];

    auto n = ctx->saved_data["n"].toOptional<int64_t>();
    auto sample_rate = ctx->saved_data["sample_rate"].toDouble();

    auto grad_output = grad_outputs[0];

    // Reconstruct time tensor if needed
    at::Tensor time;
    if (t_saved.defined()) {
      time = t_saved;
    } else {
      time = at::arange(n.value(), output.options()) / sample_rate;
    }

    // Call backward kernel
    auto grads = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::sine_wave_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&, c10::optional<int64_t>, c10::optional<at::Tensor>, const at::Tensor&, double, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, n, t_saved.defined() ? c10::optional<at::Tensor>(t_saved) : c10::nullopt,
              frequency, sample_rate, amplitude, phase);

    // Return: n, t, frequency, sample_rate, amplitude, phase, dtype, layout, device
    return {
        at::Tensor(),  // n (int, no grad)
        std::get<0>(grads),  // t
        std::get<1>(grads),  // frequency
        at::Tensor(),  // sample_rate (float, no grad)
        std::get<2>(grads),  // amplitude
        std::get<3>(grads),  // phase
        at::Tensor(),  // dtype
        at::Tensor(),  // layout
        at::Tensor()   // device
    };
  }
};

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
  return SineWaveFunction::apply(
      n, t, frequency, sample_rate, amplitude, phase, dtype, layout, device);
}

}  // namespace torchscience::autograd::signal_processing::waveform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("sine_wave", &torchscience::autograd::signal_processing::waveform::sine_wave);
}
