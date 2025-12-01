#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for quaternion normalize with gradient support.
 */
class QuaternionNormalizeFunction : public torch::autograd::Function<QuaternionNormalizeFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& q
  ) {
    ctx->save_for_backward({q});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_normalize", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(q);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q = saved[0];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto grad_q = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_normalize_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, q);

    return {grad_q};
  }
};

inline at::Tensor quaternion_normalize(const at::Tensor& q) {
  return QuaternionNormalizeFunction::apply(q);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_normalize", &torchscience::autograd::geometry::transform::quaternion_normalize);
}
