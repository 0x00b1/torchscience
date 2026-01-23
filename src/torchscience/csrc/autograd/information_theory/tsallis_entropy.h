#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class TsallisEntropyBackward
    : public torch::autograd::Function<TsallisEntropyBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        double q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        bool p_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p});
        ctx->saved_data["q"] = q;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_p = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::tsallis_entropy_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                const std::string&,
                const std::string&
            )>()
            .call(grad_output, p, q, dim, input_type, reduction);

        return grad_p;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];

        double q = ctx->saved_data["q"].toDouble();
        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];

        if (!gg_p.defined() || !p_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor()   // grad_p_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad_p] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::tsallis_entropy_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                const std::string&,
                const std::string&
            )>()
            .call(gg_p, grad_output, p, q, dim, input_type, reduction);

        return {
            grad_grad_output,
            grad_p,
            at::Tensor(),  // grad_q
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor()   // grad_p_requires_grad
        };
    }
};

class TsallisEntropyFunction
    : public torch::autograd::Function<TsallisEntropyFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        double q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction
    ) {
        ctx->saved_data["q"] = q;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::tsallis_entropy", "")
            .typed<at::Tensor(
                const at::Tensor&,
                double,
                int64_t,
                const std::string&,
                const std::string&
            )>()
            .call(p, q, dim, input_type, reduction);

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

        double q = ctx->saved_data["q"].toDouble();
        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        if (!p_requires_grad) {
            return {
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor()   // grad_reduction
            };
        }

        at::Tensor grad_p = TsallisEntropyBackward::apply(
            grad_output,
            p,
            q,
            dim,
            input_type,
            reduction,
            p_requires_grad
        );

        return {
            grad_p,
            at::Tensor(),  // grad_q
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_input_type
            at::Tensor()   // grad_reduction
        };
    }
};

inline at::Tensor tsallis_entropy(
    const at::Tensor& p,
    double q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction
) {
    return TsallisEntropyFunction::apply(p, q, dim, input_type, reduction);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("tsallis_entropy", &torchscience::autograd::information_theory::tsallis_entropy);
}
