#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::transform {

/**
 * Autograd function for matrix to quaternion with gradient support.
 */
class MatrixToQuaternionFunction : public torch::autograd::Function<MatrixToQuaternionFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& matrix
  ) {
    ctx->save_for_backward({matrix});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::matrix_to_quaternion", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(matrix);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor matrix = saved[0];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto grad_matrix = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::matrix_to_quaternion_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, matrix);

    return {grad_matrix};
  }
};

inline at::Tensor matrix_to_quaternion(const at::Tensor& matrix) {
  return MatrixToQuaternionFunction::apply(matrix);
}

}  // namespace torchscience::autograd::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("matrix_to_quaternion", &torchscience::autograd::geometry::transform::matrix_to_quaternion);
}
