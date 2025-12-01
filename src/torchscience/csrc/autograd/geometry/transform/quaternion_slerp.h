#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for quaternion slerp with gradient support.
 */
class QuaternionSlerpFunction
    : public torch::autograd::Function<QuaternionSlerpFunction> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& q1,
                            const at::Tensor& q2,
                            const at::Tensor& t) {
    ctx->save_for_backward({q1, q2, t});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::quaternion_slerp", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                          const at::Tensor&)>()
        .call(q1, q2, t);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor q1 = saved[0];
    at::Tensor q2 = saved[1];
    at::Tensor t = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::quaternion_slerp_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&)>()
            .call(grad_output, q1, q2, t);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor quaternion_slerp(const at::Tensor& q1,
                                   const at::Tensor& q2,
                                   const at::Tensor& t) {
  return QuaternionSlerpFunction::apply(q1, q2, t);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("quaternion_slerp",
         &torchscience::autograd::geometry::transform::quaternion_slerp);
}
