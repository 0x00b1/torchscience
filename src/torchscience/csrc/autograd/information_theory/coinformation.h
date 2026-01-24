#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class CoinformationFunction
    : public torch::autograd::Function<CoinformationFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& joint,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        bool requires_grad = joint.requires_grad() && at::isFloatingType(joint.scalar_type());
        ctx->saved_data["requires_grad"] = requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::coinformation", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(joint, input_type, reduction, base);

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

        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool requires_grad = ctx->saved_data["requires_grad"].toBool();

        if (!requires_grad) {
            return {
                at::Tensor(),   // grad_joint
                at::Tensor(),   // grad_input_type
                at::Tensor(),   // grad_reduction
                at::Tensor()    // grad_base
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_joint = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::coinformation_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_output, joint, input_type, reduction, base);

        return {
            grad_joint,
            at::Tensor(),   // grad_input_type
            at::Tensor(),   // grad_reduction
            at::Tensor()    // grad_base
        };
    }
};

inline at::Tensor coinformation(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return CoinformationFunction::apply(
        joint, input_type, reduction, base
    );
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("coinformation", &torchscience::autograd::information_theory::coinformation);
}
