// src/torchscience/csrc/autograd/information_theory/kullback_leibler_divergence.h
#pragma once

#include <string>
#include <tuple>

#include <torch/extension.h>

namespace torchscience::autograd::information_theory {

/**
 * Backward function class for KL divergence double-backward support.
 */
class KullbackLeiblerDivergenceBackward
    : public torch::autograd::Function<KullbackLeiblerDivergenceBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        bool pairwise,
        bool p_requires_grad,
        bool q_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["pairwise"] = pairwise;
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kullback_leibler_divergence_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                bool
            )>()
            .call(grad_output, p, q, dim, input_type, reduction, pairwise);

        return {grad_p, grad_q};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];
        at::Tensor q = saved[2];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool pairwise = ctx->saved_data["pairwise"].toBool();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        bool gg_p_defined = grad_outputs[0].defined();
        bool gg_q_defined = grad_outputs[1].defined();

        if (!(gg_p_defined && p_requires_grad) &&
            !(gg_q_defined && q_requires_grad)) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor(),  // grad_pairwise
                at::Tensor(),  // grad_p_requires_grad
                at::Tensor()   // grad_q_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor gg_p = gg_p_defined && p_requires_grad ? grad_outputs[0] : at::Tensor();
        at::Tensor gg_q = gg_q_defined && q_requires_grad ? grad_outputs[1] : at::Tensor();

        auto [grad_grad_output, grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kullback_leibler_divergence_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                bool
            )>()
            .call(gg_p, gg_q, grad_output, p, q, dim, input_type, reduction, pairwise);

        return {
            grad_grad_output,
            p_requires_grad ? grad_p : at::Tensor(),
            q_requires_grad ? grad_q : at::Tensor(),
            at::Tensor(),  // grad_dim (not differentiable)
            at::Tensor(),  // grad_input_type (not differentiable)
            at::Tensor(),  // grad_reduction (not differentiable)
            at::Tensor(),  // grad_pairwise (not differentiable)
            at::Tensor(),  // grad_p_requires_grad (not differentiable)
            at::Tensor()   // grad_q_requires_grad (not differentiable)
        };
    }
};

/**
 * Autograd Function for KL divergence.
 *
 * D_KL(P || Q) = sum_i p_i * log(p_i / q_i)
 */
class KullbackLeiblerDivergence
    : public torch::autograd::Function<KullbackLeiblerDivergence> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        bool pairwise
    ) {
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["pairwise"] = pairwise;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::kullback_leibler_divergence", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                bool
            )>()
            .call(p, q, dim, input_type, reduction, pairwise);

        ctx->save_for_backward({p, q, output});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor q = saved[1];
        // at::Tensor output = saved[2];  // Available if needed

        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool pairwise = ctx->saved_data["pairwise"].toBool();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        if (!p_requires_grad && !q_requires_grad) {
            return {
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor()   // grad_pairwise
            };
        }

        std::vector<at::Tensor> gradients = KullbackLeiblerDivergenceBackward::apply(
            grad_output,
            p,
            q,
            dim,
            input_type,
            reduction,
            pairwise,
            p_requires_grad,
            q_requires_grad
        );

        return {
            p_requires_grad ? gradients[0] : at::Tensor(),
            q_requires_grad ? gradients[1] : at::Tensor(),
            at::Tensor(),  // grad_dim (not differentiable)
            at::Tensor(),  // grad_input_type (not differentiable)
            at::Tensor(),  // grad_reduction (not differentiable)
            at::Tensor()   // grad_pairwise (not differentiable)
        };
    }
};

/**
 * Wrapper function for KL divergence with autograd support.
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which to compute divergence
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", "batchmean", or "sum"
 * @param pairwise If true, compute all-pairs divergence matrix
 * @return KL divergence
 */
inline at::Tensor kullback_leibler_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    return KullbackLeiblerDivergence::apply(p, q, dim, input_type, reduction, pairwise);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("kullback_leibler_divergence", &torchscience::autograd::information_theory::kullback_leibler_divergence);
}
