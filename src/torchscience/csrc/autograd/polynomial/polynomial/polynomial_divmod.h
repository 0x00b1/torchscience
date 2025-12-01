#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class PolynomialDivmodBackward
    : public torch::autograd::Function<PolynomialDivmodBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_Q,
        const at::Tensor& grad_R,
        const at::Tensor& Q,
        const at::Tensor& p,
        const at::Tensor& q,
        bool p_requires_grad,
        bool q_requires_grad
    ) {
        ctx->save_for_backward({grad_Q, grad_R, Q, p, q});
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_divmod_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_Q, grad_R, Q, p, q);

        return {grad_p, grad_q};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_Q = saved[0];
        at::Tensor grad_R = saved[1];
        at::Tensor Q = saved[2];
        at::Tensor p = saved[3];
        at::Tensor q = saved[4];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];  // gradient w.r.t. grad_p
        at::Tensor gg_q = grad_outputs[1];  // gradient w.r.t. grad_q

        if (!p_requires_grad && !q_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_Q, grad_grad_R, grad_Q_out, grad_p_out, grad_q_out] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_divmod_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_p.defined() ? gg_p : at::zeros_like(p),
                gg_q.defined() ? gg_q : at::zeros_like(q),
                grad_Q,
                grad_R,
                Q,
                p,
                q
            );

        return {
            grad_grad_Q,                                     // grad w.r.t. grad_Q
            grad_grad_R,                                     // grad w.r.t. grad_R
            grad_Q_out,                                      // grad w.r.t. Q
            p_requires_grad ? grad_p_out : at::Tensor(),     // grad w.r.t. p
            q_requires_grad ? grad_q_out : at::Tensor(),     // grad w.r.t. q
            at::Tensor(),                                    // p_requires_grad
            at::Tensor()                                     // q_requires_grad
        };
    }
};

// Forward function with first-order gradients
class PolynomialDivmod
    : public torch::autograd::Function<PolynomialDivmod> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q
    ) {
        at::AutoDispatchBelowAutograd guard;

        auto [quotient, remainder] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_divmod", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(p, q);

        ctx->save_for_backward({quotient, p, q});

        bool p_requires_grad = p.requires_grad() &&
            (at::isFloatingType(p.scalar_type()) || at::isComplexType(p.scalar_type()));
        bool q_requires_grad = q.requires_grad() &&
            (at::isFloatingType(q.scalar_type()) || at::isComplexType(q.scalar_type()));

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        return {quotient, remainder};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor Q = saved[0];  // quotient
        at::Tensor p = saved[1];
        at::Tensor q = saved[2];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        if (!p_requires_grad && !q_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_Q = grad_outputs[0];  // gradient w.r.t. quotient
        at::Tensor grad_R = grad_outputs[1];  // gradient w.r.t. remainder

        // Handle undefined gradients
        if (!grad_Q.defined()) {
            const int64_t N = p.size(-1);
            const int64_t M = q.size(-1);
            const int64_t B = p.numel() / N;
            const int64_t quot_len = N - M + 1;
            grad_Q = at::zeros({B, quot_len}, p.options());
        }
        if (!grad_R.defined()) {
            const int64_t N = p.size(-1);
            const int64_t M = q.size(-1);
            const int64_t B = p.numel() / N;
            const int64_t rem_len = (M > 1) ? (M - 1) : 1;
            grad_R = at::zeros({B, rem_len}, p.options());
        }

        auto grads = PolynomialDivmodBackward::apply(
            grad_Q,
            grad_R,
            Q,
            p,
            q,
            p_requires_grad,
            q_requires_grad
        );

        return {
            p_requires_grad ? grads[0] : at::Tensor(),
            q_requires_grad ? grads[1] : at::Tensor()
        };
    }
};

inline std::tuple<at::Tensor, at::Tensor> polynomial_divmod(
    const at::Tensor& p,
    const at::Tensor& q
) {
    auto results = PolynomialDivmod::apply(p, q);
    return {results[0], results[1]};
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polynomial_divmod", &torchscience::autograd::polynomial::polynomial_divmod);
}
