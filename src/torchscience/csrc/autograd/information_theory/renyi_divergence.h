// src/torchscience/csrc/autograd/information_theory/renyi_divergence.h
#pragma once

#include <string>
#include <tuple>

#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace torchscience::autograd::information_theory {

/**
 * Autograd Function for Renyi divergence.
 *
 * D_alpha(P || Q) = 1/(alpha-1) * log(sum_i p_i^alpha * q_i^(1-alpha))
 */
class RenyiDivergence
    : public torch::autograd::Function<RenyiDivergence> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        double alpha,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        c10::optional<double> base,
        bool pairwise
    ) {
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["has_base"] = base.has_value();
        ctx->saved_data["base"] = base.has_value() ? base.value() : 0.0;
        ctx->saved_data["pairwise"] = pairwise;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::renyi_divergence", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>,
                bool
            )>()
            .call(p, q, alpha, dim, input_type, reduction, base, pairwise);

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
        // at::Tensor output = saved[2];

        at::Tensor grad_output = grad_outputs[0];

        double alpha = ctx->saved_data["alpha"].toDouble();
        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool has_base = ctx->saved_data["has_base"].toBool();
        double base_val = ctx->saved_data["base"].toDouble();
        c10::optional<double> base = has_base ? c10::optional<double>(base_val) : c10::nullopt;
        bool pairwise = ctx->saved_data["pairwise"].toBool();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        if (!p_requires_grad && !q_requires_grad) {
            return {
                at::Tensor(),  // grad_p
                at::Tensor(),  // grad_q
                at::Tensor(),  // grad_alpha
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_input_type
                at::Tensor(),  // grad_reduction
                at::Tensor(),  // grad_base
                at::Tensor()   // grad_pairwise
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::renyi_divergence_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                const std::string&,
                const std::string&,
                c10::optional<double>,
                bool
            )>()
            .call(grad_output, p, q, alpha, dim, input_type, reduction, base, pairwise);

        return {
            p_requires_grad ? grad_p : at::Tensor(),
            q_requires_grad ? grad_q : at::Tensor(),
            at::Tensor(),  // grad_alpha (not differentiable)
            at::Tensor(),  // grad_dim (not differentiable)
            at::Tensor(),  // grad_input_type (not differentiable)
            at::Tensor(),  // grad_reduction (not differentiable)
            at::Tensor(),  // grad_base (not differentiable)
            at::Tensor()   // grad_pairwise (not differentiable)
        };
    }
};

/**
 * Wrapper function for Renyi divergence with autograd support.
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param alpha Order of Renyi divergence
 * @param dim Dimension along which to compute divergence
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", "batchmean", or "sum"
 * @param base Logarithm base (optional, defaults to e)
 * @param pairwise If true, compute all-pairs divergence matrix
 * @return Renyi divergence
 */
inline at::Tensor renyi_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    double alpha,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base,
    bool pairwise
) {
    return RenyiDivergence::apply(p, q, alpha, dim, input_type, reduction, base, pairwise);
}

}  // namespace torchscience::autograd::information_theory

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("renyi_divergence", &torchscience::autograd::information_theory::renyi_divergence);
}
