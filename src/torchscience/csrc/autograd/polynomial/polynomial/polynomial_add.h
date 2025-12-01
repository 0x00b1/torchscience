#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class PolynomialAddBackward
    : public torch::autograd::Function<PolynomialAddBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& q,
        bool p_requires_grad,
        bool q_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p, q});
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_add_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, p, q);

        return {grad_p, grad_q};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];
        at::Tensor q = saved[2];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];  // gradient w.r.t. grad_p
        at::Tensor gg_q = grad_outputs[1];  // gradient w.r.t. grad_q

        if (!p_requires_grad && !q_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, g_p, g_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_add_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_p.defined() ? gg_p : at::zeros_like(p),
                gg_q.defined() ? gg_q : at::zeros_like(q),
                grad_output,
                p,
                q
            );

        return {
            grad_grad_output,
            p_requires_grad ? g_p : at::Tensor(),
            q_requires_grad ? g_q : at::Tensor(),
            at::Tensor(),  // p_requires_grad (not differentiable)
            at::Tensor()   // q_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class PolynomialAdd
    : public torch::autograd::Function<PolynomialAdd> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q
    ) {
        ctx->save_for_backward({p, q});

        bool p_requires_grad = p.requires_grad() &&
            (at::isFloatingType(p.scalar_type()) || at::isComplexType(p.scalar_type()));
        bool q_requires_grad = q.requires_grad() &&
            (at::isFloatingType(q.scalar_type()) || at::isComplexType(q.scalar_type()));

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_add", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(p, q);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor q = saved[1];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        if (!p_requires_grad && !q_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        auto grads = PolynomialAddBackward::apply(
            grad_outputs[0],
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

inline at::Tensor polynomial_add(
    const at::Tensor& p,
    const at::Tensor& q
) {
    return PolynomialAdd::apply(p, q);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polynomial_add", &torchscience::autograd::polynomial::polynomial_add);
}
