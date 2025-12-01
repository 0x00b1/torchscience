#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

class LegendrePolynomialPEvaluateBackward
    : public torch::autograd::Function<LegendrePolynomialPEvaluateBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& x,
        bool coeffs_requires_grad,
        bool x_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs, x});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["x_requires_grad"] = x_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_coeffs, grad_x] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::legendre_polynomial_p_evaluate_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, coeffs, x);

        return {grad_coeffs, grad_x};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];
        at::Tensor x = saved[2];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();

        at::Tensor gg_coeffs = grad_outputs[0];
        at::Tensor gg_x = grad_outputs[1];

        if (!coeffs_requires_grad && !x_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, g_coeffs, g_x] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::legendre_polynomial_p_evaluate_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_coeffs.defined() ? gg_coeffs : at::zeros_like(coeffs),
                gg_x.defined() ? gg_x : at::zeros_like(x),
                grad_output,
                coeffs,
                x
            );

        return {
            grad_grad_output,
            coeffs_requires_grad ? g_coeffs : at::Tensor(),
            x_requires_grad ? g_x : at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

class LegendrePolynomialPEvaluate
    : public torch::autograd::Function<LegendrePolynomialPEvaluate> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs,
        const at::Tensor& x
    ) {
        ctx->save_for_backward({coeffs, x});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));
        bool x_requires_grad = x.requires_grad() &&
            (at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["x_requires_grad"] = x_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::legendre_polynomial_p_evaluate", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(coeffs, x);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor x = saved[1];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();

        if (!coeffs_requires_grad && !x_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        auto grads = LegendrePolynomialPEvaluateBackward::apply(
            grad_outputs[0],
            coeffs,
            x,
            coeffs_requires_grad,
            x_requires_grad
        );

        return {
            coeffs_requires_grad ? grads[0] : at::Tensor(),
            x_requires_grad ? grads[1] : at::Tensor()
        };
    }
};

inline at::Tensor legendre_polynomial_p_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x
) {
    return LegendrePolynomialPEvaluate::apply(coeffs, x);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("legendre_polynomial_p_evaluate", &torchscience::autograd::polynomial::legendre_polynomial_p_evaluate);
}
