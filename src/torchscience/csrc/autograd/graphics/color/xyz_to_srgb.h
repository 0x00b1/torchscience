#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::color {

/**
 * Autograd function for XYZ to sRGB conversion.
 */
class XyzToSrgbFunction : public torch::autograd::Function<XyzToSrgbFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input
  ) {
    ctx->save_for_backward({input});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::xyz_to_srgb", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(input);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor input = saved[0];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    at::Tensor grad_input = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::xyz_to_srgb_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, input);

    return {grad_input};
  }
};

inline at::Tensor xyz_to_srgb(const at::Tensor& input) {
  return XyzToSrgbFunction::apply(input);
}

}  // namespace torchscience::autograd::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("xyz_to_srgb", &torchscience::autograd::graphics::color::xyz_to_srgb);
}
