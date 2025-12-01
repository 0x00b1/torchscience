// src/torchscience/csrc/autograd/information_theory/jensen_shannon_divergence.h
#pragma once

#include <string>
#include <tuple>

#include <c10/util/Optional.h>
#include <torch/extension.h>

namespace torchscience::autograd::information_theory {

/**
 * Backward function class for JS divergence double-backward support.
 */
class JensenShannonDivergenceBackward
    : public torch::autograd::Function<JensenShannonDivergenceBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        const c10::optional<double>& base,
        bool pairwise,
        bool p_requires_grad,
        bool q_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p, q});
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["has_base"] = base.has_value();
        if (base.has_value()) {
            ctx->saved_data["base"] = base.value();
        }
        ctx->saved_data["pairwise"] = pairwise;
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jensen_shannon_divergence_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                const c10::optional<double>&,
                bool
            )>()
            .call(grad_output, p, q, dim, input_type, reduction, base, pairwise);

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
        bool has_base = ctx->saved_data["has_base"].toBool();
        c10::optional<double> base;
        if (has_base) {
            base = ctx->saved_data["base"].toDouble();
        }
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
                at::Tensor(),  // grad_base
                at::Tensor(),  // grad_pairwise
                at::Tensor(),  // grad_p_requires_grad
                at::Tensor()   // grad_q_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor gg_p = gg_p_defined && p_requires_grad ? grad_outputs[0] : at::Tensor();
        at::Tensor gg_q = gg_q_defined && q_requires_grad ? grad_outputs[1] : at::Tensor();

        auto [grad_grad_output, grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jensen_shannon_divergence_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                const c10::optional<double>&,
                bool
            )>()
            .call(gg_p, gg_q, grad_output, p, q, dim, input_type, reduction, base, pairwise);

        return {
            grad_grad_output,
            p_requires_grad ? grad_p : at::Tensor(),
            q_requires_grad ? grad_q : at::Tensor(),
            at::Tensor(),  // grad_dim (not differentiable)
            at::Tensor(),  // grad_input_type (not differentiable)
            at::Tensor(),  // grad_reduction (not differentiable)
            at::Tensor(),  // grad_base (not differentiable)
            at::Tensor(),  // grad_pairwise (not differentiable)
            at::Tensor(),  // grad_p_requires_grad (not differentiable)
            at::Tensor()   // grad_q_requires_grad (not differentiable)
        };
    }
};

/**
 * Autograd Function for JS divergence.
 *
 * D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 * where M = 0.5 * (P + Q)
 */
class JensenShannonDivergence
    : public torch::autograd::Function<JensenShannonDivergence> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        const c10::optional<double>& base,
        bool pairwise
    ) {
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["has_base"] = base.has_value();
        if (base.has_value()) {
            ctx->saved_data["base"] = base.value();
        }
        ctx->saved_data["pairwise"] = pairwise;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jensen_shannon_divergence", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                const c10::optional<double>&,
                bool
            )>()
            .call(p, q, dim, input_type, reduction, base, pairwise);

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
        bool has_base = ctx->saved_data["has_base"].toBool();
        c10::optional<double> base;
        if (has_base) {
            base = ctx->saved_data["base"].toDouble();
        }
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
                at::Tensor(),  // grad_base
                at::Tensor()   // grad_pairwise
            };
        }

        std::vector<at::Tensor> gradients = JensenShannonDivergenceBackward::apply(
            grad_output,
            p,
            q,
            dim,
            input_type,
            reduction,
            base,
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
            at::Tensor(),  // grad_base (not differentiable)
            at::Tensor()   // grad_pairwise (not differentiable)
        };
    }
};

/**
 * Wrapper function for JS divergence with autograd support.
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which to compute divergence
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", "batchmean", or "sum"
 * @param base Optional log base (default: natural log)
 * @param pairwise If true, compute all-pairs divergence matrix
 * @return JS divergence
 */
inline at::Tensor jensen_shannon_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    const c10::optional<double>& base,
    bool pairwise
) {
    return JensenShannonDivergence::apply(p, q, dim, input_type, reduction, base, pairwise);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("jensen_shannon_divergence", &torchscience::autograd::information_theory::jensen_shannon_divergence);
}
