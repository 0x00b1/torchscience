#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class PointwiseMutualInformationBackward
    : public torch::autograd::Function<PointwiseMutualInformationBackward> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& grad_output,
      const at::Tensor& joint,
      std::vector<int64_t> dims,
      const std::string& input_type,
      c10::optional<double> base,
      bool joint_requires_grad) {
    ctx->save_for_backward({grad_output, joint});
    ctx->saved_data["dims"] = dims;
    ctx->saved_data["input_type"] = input_type;
    ctx->saved_data["base"] = base;
    ctx->saved_data["joint_requires_grad"] = joint_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    at::Tensor grad_joint =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "torchscience::pointwise_mutual_information_backward", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, at::IntArrayRef,
                const std::string&, c10::optional<double>)>()
            .call(grad_output, joint, dims, input_type, base);

    return grad_joint;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor grad_output = saved[0];
    at::Tensor joint = saved[1];

    std::vector<int64_t> dims = ctx->saved_data["dims"].toIntVector();
    std::string input_type = ctx->saved_data["input_type"].toStringRef();
    c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
    bool joint_requires_grad =
        ctx->saved_data["joint_requires_grad"].toBool();

    at::Tensor gg_joint = grad_outputs[0];

    if (!gg_joint.defined() || !joint_requires_grad) {
      return {
          at::Tensor(),  // grad_grad_output
          at::Tensor(),  // grad_joint
          at::Tensor(),  // grad_dims
          at::Tensor(),  // grad_input_type
          at::Tensor(),  // grad_base
          at::Tensor()   // grad_joint_requires_grad
      };
    }

    at::AutoDispatchBelowAutograd guard;

    auto [grad_grad_output, grad_joint_out] =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "torchscience::pointwise_mutual_information_backward_backward",
                "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                at::IntArrayRef, const std::string&, c10::optional<double>)>()
            .call(gg_joint, grad_output, joint, dims, input_type, base);

    return {
        grad_grad_output,
        joint_requires_grad ? grad_joint_out : at::Tensor(),
        at::Tensor(),  // grad_dims
        at::Tensor(),  // grad_input_type
        at::Tensor(),  // grad_base
        at::Tensor()   // grad_joint_requires_grad
    };
  }
};

class PointwiseMutualInformationFunction
    : public torch::autograd::Function<PointwiseMutualInformationFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& joint,
      std::vector<int64_t> dims,
      const std::string& input_type,
      c10::optional<double> base) {
    ctx->saved_data["dims"] = dims;
    ctx->saved_data["input_type"] = input_type;
    ctx->saved_data["base"] = base;

    bool joint_requires_grad =
        joint.requires_grad() && at::isFloatingType(joint.scalar_type());
    ctx->saved_data["joint_requires_grad"] = joint_requires_grad;

    at::AutoDispatchBelowAutograd guard;

    at::Tensor output =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::pointwise_mutual_information", "")
            .typed<at::Tensor(
                const at::Tensor&, at::IntArrayRef, const std::string&,
                c10::optional<double>)>()
            .call(joint, dims, input_type, base);

    ctx->save_for_backward({joint});

    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor joint = saved[0];
    at::Tensor grad_output = grad_outputs[0];

    std::vector<int64_t> dims = ctx->saved_data["dims"].toIntVector();
    std::string input_type = ctx->saved_data["input_type"].toStringRef();
    c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
    bool joint_requires_grad =
        ctx->saved_data["joint_requires_grad"].toBool();

    if (!joint_requires_grad) {
      return {
          at::Tensor(),  // grad_joint
          at::Tensor(),  // grad_dims
          at::Tensor(),  // grad_input_type
          at::Tensor()   // grad_base
      };
    }

    at::Tensor grad_joint = PointwiseMutualInformationBackward::apply(
        grad_output, joint, dims, input_type, base, joint_requires_grad);

    return {
        grad_joint,
        at::Tensor(),  // grad_dims
        at::Tensor(),  // grad_input_type
        at::Tensor()   // grad_base
    };
  }
};

inline at::Tensor pointwise_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base) {
  std::vector<int64_t> dims_vec(dims.begin(), dims.end());
  return PointwiseMutualInformationFunction::apply(
      joint, dims_vec, input_type, base);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl(
      "pointwise_mutual_information",
      &torchscience::autograd::information_theory::pointwise_mutual_information);
}
