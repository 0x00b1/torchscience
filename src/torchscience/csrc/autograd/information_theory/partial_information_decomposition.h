#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class PartialInformationDecompositionFunction
    : public torch::autograd::Function<PartialInformationDecompositionFunction> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& joint,
        const std::string& method,
        const std::string& input_type,
        c10::optional<double> base
    ) {
        ctx->saved_data["method"] = method;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["base"] = base;

        bool requires_grad = joint.requires_grad() && at::isFloatingType(joint.scalar_type());
        ctx->saved_data["requires_grad"] = requires_grad;

        at::AutoDispatchBelowAutograd guard;

        std::vector<at::Tensor> outputs = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::partial_information_decomposition", "")
            .typed<std::vector<at::Tensor>(
                const at::Tensor&,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(joint, method, input_type, base);

        ctx->save_for_backward({joint});

        return outputs;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor joint = saved[0];

        // grad_outputs contains gradients for [redundancy, unique_x, unique_y, synergy, mutual_info]
        at::Tensor grad_redundancy = grad_outputs[0];
        at::Tensor grad_unique_x = grad_outputs[1];
        at::Tensor grad_unique_y = grad_outputs[2];
        at::Tensor grad_synergy = grad_outputs[3];
        at::Tensor grad_mutual_info = grad_outputs[4];

        std::string method = ctx->saved_data["method"].toStringRef();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool requires_grad = ctx->saved_data["requires_grad"].toBool();

        if (!requires_grad) {
            return {
                at::Tensor(),   // grad_joint
                at::Tensor(),   // grad_method
                at::Tensor(),   // grad_input_type
                at::Tensor()    // grad_base
            };
        }

        at::AutoDispatchBelowAutograd guard;

        std::vector<at::Tensor> grads = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::partial_information_decomposition_backward", "")
            .typed<std::vector<at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_redundancy, grad_unique_x, grad_unique_y, grad_synergy, grad_mutual_info,
                  joint, method, input_type, base);

        return {
            grads[0],
            at::Tensor(),   // grad_method
            at::Tensor(),   // grad_input_type
            at::Tensor()    // grad_base
        };
    }
};

inline std::vector<at::Tensor> partial_information_decomposition(
    const at::Tensor& joint,
    const std::string& method,
    const std::string& input_type,
    c10::optional<double> base
) {
    return PartialInformationDecompositionFunction::apply(
        joint, method, input_type, base
    );
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("partial_information_decomposition", &torchscience::autograd::information_theory::partial_information_decomposition);
}
