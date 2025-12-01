#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class ShannonEntropyBackward
    : public torch::autograd::Function<ShannonEntropyBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base,
        bool p_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;
        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_p = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::shannon_entropy_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_output, p, dim, input_type, reduction, base);

        return grad_p;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];

        if (!gg_p.defined() || !p_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor(),  // grad_base
                at::Tensor()   // grad_p_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad_p] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::shannon_entropy_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(gg_p, grad_output, p, dim, input_type, reduction, base);

        return {
            grad_grad_output,
            p_requires_grad ? grad_p : at::Tensor(),
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor(),  // grad_base
            at::Tensor()   // grad_p_requires_grad
        };
    }
};

class ShannonEntropyFunction
    : public torch::autograd::Function<ShannonEntropyFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::shannon_entropy", "")
            .typed<at::Tensor(
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(p, dim, input_type, reduction, base);

        ctx->save_for_backward({p});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        if (!p_requires_grad) {
            return {
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor()   // grad_base
            };
        }

        at::Tensor grad_p = ShannonEntropyBackward::apply(
            grad_output,
            p,
            dim,
            input_type,
            reduction,
            base,
            p_requires_grad
        );

        return {
            grad_p,
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor()   // grad_base
        };
    }
};

inline at::Tensor shannon_entropy(
    const at::Tensor& p,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return ShannonEntropyFunction::apply(p, dim, input_type, reduction, base);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("shannon_entropy", &torchscience::autograd::information_theory::shannon_entropy);
}
