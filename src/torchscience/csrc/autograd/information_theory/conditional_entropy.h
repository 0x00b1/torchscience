#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class ConditionalEntropyBackward
    : public torch::autograd::Function<ConditionalEntropyBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& joint,
        int64_t condition_dim,
        int64_t target_dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base,
        bool joint_requires_grad
    ) {
        ctx->save_for_backward({grad_output, joint});
        ctx->saved_data["condition_dim"] = condition_dim;
        ctx->saved_data["target_dim"] = target_dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;
        ctx->saved_data["joint_requires_grad"] = joint_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_joint = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::conditional_entropy_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_output, joint, condition_dim, target_dim, input_type, reduction, base);

        return grad_joint;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor joint = saved[1];

        int64_t condition_dim = ctx->saved_data["condition_dim"].toInt();
        int64_t target_dim = ctx->saved_data["target_dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool joint_requires_grad = ctx->saved_data["joint_requires_grad"].toBool();

        at::Tensor gg_joint = grad_outputs[0];

        if (!gg_joint.defined() || !joint_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_joint
                at::Tensor(),  // grad_condition_dim
                at::Tensor(),  // grad_target_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor(),  // grad_base
                at::Tensor()   // grad_joint_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad_joint] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::conditional_entropy_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(gg_joint, grad_output, joint, condition_dim, target_dim, input_type, reduction, base);

        return {
            grad_grad_output,
            joint_requires_grad ? grad_joint : at::Tensor(),
            at::Tensor(),  // grad_condition_dim
            at::Tensor(),  // grad_target_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor(),  // grad_base
            at::Tensor()   // grad_joint_requires_grad
        };
    }
};

class ConditionalEntropyFunction
    : public torch::autograd::Function<ConditionalEntropyFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& joint,
        int64_t condition_dim,
        int64_t target_dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->saved_data["condition_dim"] = condition_dim;
        ctx->saved_data["target_dim"] = target_dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        bool joint_requires_grad = joint.requires_grad() && at::isFloatingType(joint.scalar_type());
        ctx->saved_data["joint_requires_grad"] = joint_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::conditional_entropy", "")
            .typed<at::Tensor(
                const at::Tensor&,
                int64_t,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(joint, condition_dim, target_dim, input_type, reduction, base);

        ctx->save_for_backward({joint});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor joint = saved[0];
        at::Tensor grad_output = grad_outputs[0];

        int64_t condition_dim = ctx->saved_data["condition_dim"].toInt();
        int64_t target_dim = ctx->saved_data["target_dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool joint_requires_grad = ctx->saved_data["joint_requires_grad"].toBool();

        if (!joint_requires_grad) {
            return {
                at::Tensor(),  // grad_joint
                at::Tensor(),  // grad_condition_dim
                at::Tensor(),  // grad_target_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor()   // grad_base
            };
        }

        at::Tensor grad_joint = ConditionalEntropyBackward::apply(
            grad_output,
            joint,
            condition_dim,
            target_dim,
            input_type,
            reduction,
            base,
            joint_requires_grad
        );

        return {
            grad_joint,
            at::Tensor(),  // grad_condition_dim
            at::Tensor(),  // grad_target_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor()   // grad_base
        };
    }
};

inline at::Tensor conditional_entropy(
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return ConditionalEntropyFunction::apply(joint, condition_dim, target_dim, input_type, reduction, base);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("conditional_entropy", &torchscience::autograd::information_theory::conditional_entropy);
}
