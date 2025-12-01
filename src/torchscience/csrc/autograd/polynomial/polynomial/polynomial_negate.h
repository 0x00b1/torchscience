#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class PolynomialNegateBackward
    : public torch::autograd::Function<PolynomialNegateBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        bool p_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p});
        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto grad_p = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_negate_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, p);

        return {grad_p};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];  // gradient w.r.t. grad_p

        if (!p_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, g_p] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_negate_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_p.defined() ? gg_p : at::zeros_like(p),
                grad_output,
                p
            );

        return {
            grad_grad_output,
            p_requires_grad ? g_p : at::Tensor(),
            at::Tensor()  // p_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class PolynomialNegate
    : public torch::autograd::Function<PolynomialNegate> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p
    ) {
        ctx->save_for_backward({p});

        bool p_requires_grad = p.requires_grad() &&
            (at::isFloatingType(p.scalar_type()) || at::isComplexType(p.scalar_type()));

        ctx->saved_data["p_requires_grad"] = p_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_negate", "")
            .typed<at::Tensor(
                const at::Tensor&
            )>()
            .call(p);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();

        if (!p_requires_grad) {
            return {at::Tensor()};
        }

        auto grads = PolynomialNegateBackward::apply(
            grad_outputs[0],
            p,
            p_requires_grad
        );

        return {
            p_requires_grad ? grads[0] : at::Tensor()
        };
    }
};

inline at::Tensor polynomial_negate(
    const at::Tensor& p
) {
    return PolynomialNegate::apply(p);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polynomial_negate", &torchscience::autograd::polynomial::polynomial_negate);
}
