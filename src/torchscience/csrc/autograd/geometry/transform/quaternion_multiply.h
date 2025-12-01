#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for quaternion multiplication with gradient support.
 */
class QuaternionMultiplyFunction : public torch::autograd::Function<QuaternionMultiplyFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& q1,
      const at::Tensor& q2
  ) {
    ctx->save_for_backward({q1, q2});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_multiply", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(q1, q2);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q1 = saved[0];
    at::Tensor q2 = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_multiply_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, q1, q2);

    return {std::get<0>(result), std::get<1>(result)};
  }
};

inline at::Tensor quaternion_multiply(const at::Tensor& q1, const at::Tensor& q2) {
  return QuaternionMultiplyFunction::apply(q1, q2);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_multiply", &torchscience::autograd::geometry::transform::quaternion_multiply);
}
