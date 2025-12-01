#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::information_theory {

class CrossEntropyBackward
    : public torch::autograd::Function<CrossEntropyBackward> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base,
        bool p_requires_grad,
        bool q_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(grad_output, p, q, dim, input_type, reduction, base);

        return {std::get<0>(result), std::get<1>(result)};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];
        at::Tensor q = saved[2];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];
        at::Tensor gg_q = grad_outputs[1];

        if ((!gg_p.defined() || !p_requires_grad) &&
            (!gg_q.defined() || !q_requires_grad)) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor(),  // grad_base
                at::Tensor(),  // grad_p_requires_grad
                at::Tensor()   // grad_q_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(gg_p, gg_q, grad_output, p, q, dim, input_type, reduction, base);

        return {
            std::get<0>(result),  // grad_grad_output
            std::get<1>(result),  // grad_p
            std::get<2>(result),  // grad_q
            at::Tensor(),         // grad_dim
            at::Tensor(),         // grad_input_type
            at::Tensor(),         // grad_reduction
            at::Tensor(),         // grad_base
            at::Tensor(),         // grad_p_requires_grad
            at::Tensor()          // grad_q_requires_grad
        };
    }
};

class CrossEntropyFunction
    : public torch::autograd::Function<CrossEntropyFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base
    ) {
        ctx->save_for_backward({p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["base"] = base;
        ctx->saved_data["p_requires_grad"] = p.requires_grad();
        ctx->saved_data["q_requires_grad"] = q.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cross_entropy", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>
            )>()
            .call(p, q, dim, input_type, reduction, base);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor q = saved[1];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        c10::optional<double> base = ctx->saved_data["base"].toOptional<double>();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        at::Tensor grad_output = grad_outputs[0];

        if (!grad_output.defined()) {
            return {
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor()   // grad_base
            };
        }

        auto grads = CrossEntropyBackward::apply(
            grad_output, p, q, dim, input_type, reduction, base,
            p_requires_grad, q_requires_grad
        );

        return {
            p_requires_grad ? grads[0] : at::Tensor(),
            q_requires_grad ? grads[1] : at::Tensor(),
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_input_type
            at::Tensor(),  // grad_reduction
            at::Tensor()   // grad_base
        };
    }
};

inline at::Tensor cross_entropy(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return CrossEntropyFunction::apply(p, q, dim, input_type, reduction, base);
}

inline std::tuple<at::Tensor, at::Tensor> cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    auto result = CrossEntropyBackward::apply(
        grad_output, p, q, dim, input_type, reduction, base,
        p.requires_grad(), q.requires_grad()
    );
    return std::make_tuple(result[0], result[1]);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("cross_entropy", &torchscience::autograd::information_theory::cross_entropy);
    m.impl("cross_entropy_backward", &torchscience::autograd::information_theory::cross_entropy_backward);
}
