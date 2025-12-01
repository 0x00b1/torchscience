#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for vector reflection.
 */
class ReflectFunction : public torch::autograd::Function<ReflectFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& direction,
      const at::Tensor& normal
  ) {
    ctx->save_for_backward({direction, normal});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::reflect", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(direction, normal);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor direction = saved[0];
    at::Tensor normal = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::reflect_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, direction, normal);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor reflect(const at::Tensor& direction, const at::Tensor& normal) {
  return ReflectFunction::apply(direction, normal);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("reflect", &torchscience::autograd::geometry::transform::reflect);
}
