#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class ConditionalMutualInformationFunction
    : public torch::autograd::Function<ConditionalMutualInformationFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& joint,
        std::vector<int64_t> dims_x,
        std::vector<int64_t> dims_y,
        std::vector<int64_t> dims_z,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->saved_data["dims_x"] = dims_x;
        ctx->saved_data["dims_y"] = dims_y;
        ctx->saved_data["dims_z"] = dims_z;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        bool requires_grad = joint.requires_grad() && at::isFloatingType(joint.scalar_type());
        ctx->saved_data["requires_grad"] = requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::conditional_mutual_information", "")
            .typed<at::Tensor(
                const at::Tensor&,
                at::IntArrayRef,
                at::IntArrayRef,
                at::IntArrayRef,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(joint, dims_x, dims_y, dims_z, input_type, reduction, base);

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

        auto dims_x = ctx->saved_data["dims_x"].toIntVector();
        auto dims_y = ctx->saved_data["dims_y"].toIntVector();
        auto dims_z = ctx->saved_data["dims_z"].toIntVector();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool requires_grad = ctx->saved_data["requires_grad"].toBool();

        if (!requires_grad) {
            return {
                at::Tensor(),   // grad_joint
                at::Tensor(),   // grad_dims_x
                at::Tensor(),   // grad_dims_y
                at::Tensor(),   // grad_dims_z
                at::Tensor(),   // grad_input_type
                at::Tensor(),   // grad_reduction
                at::Tensor()    // grad_base
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_joint = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::conditional_mutual_information_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                at::IntArrayRef,
                at::IntArrayRef,
                at::IntArrayRef,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_output, joint, dims_x, dims_y, dims_z, input_type, reduction, base);

        return {
            grad_joint,
            at::Tensor(),   // grad_dims_x
            at::Tensor(),   // grad_dims_y
            at::Tensor(),   // grad_dims_z
            at::Tensor(),   // grad_input_type
            at::Tensor(),   // grad_reduction
            at::Tensor()    // grad_base
        };
    }
};

inline at::Tensor conditional_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims_x,
    at::IntArrayRef dims_y,
    at::IntArrayRef dims_z,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    std::vector<int64_t> dims_x_vec(dims_x.begin(), dims_x.end());
    std::vector<int64_t> dims_y_vec(dims_y.begin(), dims_y.end());
    std::vector<int64_t> dims_z_vec(dims_z.begin(), dims_z.end());
    return ConditionalMutualInformationFunction::apply(
        joint, dims_x_vec, dims_y_vec, dims_z_vec, input_type, reduction, base
    );
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("conditional_mutual_information", &torchscience::autograd::information_theory::conditional_mutual_information);
}
