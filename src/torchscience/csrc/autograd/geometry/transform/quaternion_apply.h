#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for quaternion_apply with gradient support.
 */
class QuaternionApplyFunction : public torch::autograd::Function<QuaternionApplyFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& q,
      const at::Tensor& point
  ) {
    ctx->save_for_backward({q, point});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_apply", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(q, point);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q = saved[0];
    at::Tensor point = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_apply_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, q, point);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor quaternion_apply(const at::Tensor& q, const at::Tensor& point) {
  return QuaternionApplyFunction::apply(q, point);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_apply", &torchscience::autograd::geometry::transform::quaternion_apply);
}
