#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class PolynomialScaleBackward
    : public torch::autograd::Function<PolynomialScaleBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& p,
        const at::Tensor& c,
        bool p_requires_grad,
        bool c_requires_grad
    ) {
        ctx->save_for_backward({grad_output, p, c});
        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["c_requires_grad"] = c_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_c] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_scale_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, p, c);

        return {grad_p, grad_c};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor p = saved[1];
        at::Tensor c = saved[2];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool c_requires_grad = ctx->saved_data["c_requires_grad"].toBool();

        at::Tensor gg_p = grad_outputs[0];  // gradient w.r.t. grad_p
        at::Tensor gg_c = grad_outputs[1];  // gradient w.r.t. grad_c

        if (!p_requires_grad && !c_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, g_p, g_c] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_scale_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_p.defined() ? gg_p : at::zeros_like(p),
                gg_c.defined() ? gg_c : at::zeros_like(c),
                grad_output,
                p,
                c
            );

        return {
            grad_grad_output,
            p_requires_grad ? g_p : at::Tensor(),
            c_requires_grad ? g_c : at::Tensor(),
            at::Tensor(),  // p_requires_grad (not differentiable)
            at::Tensor()   // c_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class PolynomialScale
    : public torch::autograd::Function<PolynomialScale> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& c
    ) {
        ctx->save_for_backward({p, c});

        bool p_requires_grad = p.requires_grad() &&
            (at::isFloatingType(p.scalar_type()) || at::isComplexType(p.scalar_type()));
        bool c_requires_grad = c.requires_grad() &&
            (at::isFloatingType(c.scalar_type()) || at::isComplexType(c.scalar_type()));

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["c_requires_grad"] = c_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_scale", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(p, c);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor c = saved[1];

        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool c_requires_grad = ctx->saved_data["c_requires_grad"].toBool();

        if (!p_requires_grad && !c_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        auto grads = PolynomialScaleBackward::apply(
            grad_outputs[0],
            p,
            c,
            p_requires_grad,
            c_requires_grad
        );

        return {
            p_requires_grad ? grads[0] : at::Tensor(),
            c_requires_grad ? grads[1] : at::Tensor()
        };
    }
};

inline at::Tensor polynomial_scale(
    const at::Tensor& p,
    const at::Tensor& c
) {
    return PolynomialScale::apply(p, c);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polynomial_scale", &torchscience::autograd::polynomial::polynomial_scale);
}
