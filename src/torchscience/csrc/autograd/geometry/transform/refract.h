#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for vector refraction using Snell's law.
 */
class RefractFunction : public torch::autograd::Function<RefractFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& direction,
      const at::Tensor& normal,
      const at::Tensor& eta
  ) {
    ctx->save_for_backward({direction, normal, eta});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::refract", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(direction, normal, eta);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor direction = saved[0];
    at::Tensor normal = saved[1];
    at::Tensor eta = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::refract_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, direction, normal, eta);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor refract(
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& eta
) {
  return RefractFunction::apply(direction, normal, eta);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("refract", &torchscience::autograd::geometry::transform::refract);
}
