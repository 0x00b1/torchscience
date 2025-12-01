#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

class GegenbauerPolynomialCEvaluateBackward
    : public torch::autograd::Function<GegenbauerPolynomialCEvaluateBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& x,
        const at::Tensor& alpha,
        bool coeffs_requires_grad,
        bool x_requires_grad,
        bool alpha_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs, x, alpha});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["x_requires_grad"] = x_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_coeffs, grad_x, grad_alpha] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_evaluate_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, coeffs, x, alpha);

        return {grad_coeffs, grad_x, grad_alpha};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];
        at::Tensor x = saved[2];
        at::Tensor alpha = saved[3];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();
        bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

        at::Tensor gg_coeffs = grad_outputs[0];
        at::Tensor gg_x = grad_outputs[1];
        at::Tensor gg_alpha = grad_outputs[2];

        if (!coeffs_requires_grad && !x_requires_grad && !alpha_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, g_coeffs, g_x, g_alpha] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_evaluate_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_coeffs.defined() ? gg_coeffs : at::zeros_like(coeffs),
                gg_x.defined() ? gg_x : at::zeros_like(x),
                gg_alpha.defined() ? gg_alpha : at::zeros_like(alpha),
                grad_output,
                coeffs,
                x,
                alpha
            );

        return {
            grad_grad_output,
            coeffs_requires_grad ? g_coeffs : at::Tensor(),
            x_requires_grad ? g_x : at::Tensor(),
            alpha_requires_grad ? g_alpha : at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

class GegenbauerPolynomialCEvaluate
    : public torch::autograd::Function<GegenbauerPolynomialCEvaluate> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs,
        const at::Tensor& x,
        const at::Tensor& alpha
    ) {
        ctx->save_for_backward({coeffs, x, alpha});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));
        bool x_requires_grad = x.requires_grad() &&
            (at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type()));
        bool alpha_requires_grad = alpha.requires_grad() &&
            (at::isFloatingType(alpha.scalar_type()) || at::isComplexType(alpha.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["x_requires_grad"] = x_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_evaluate", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(coeffs, x, alpha);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor x = saved[1];
        at::Tensor alpha = saved[2];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();
        bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

        if (!coeffs_requires_grad && !x_requires_grad && !alpha_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        auto grads = GegenbauerPolynomialCEvaluateBackward::apply(
            grad_outputs[0],
            coeffs,
            x,
            alpha,
            coeffs_requires_grad,
            x_requires_grad,
            alpha_requires_grad
        );

        return {
            coeffs_requires_grad ? grads[0] : at::Tensor(),
            x_requires_grad ? grads[1] : at::Tensor(),
            alpha_requires_grad ? grads[2] : at::Tensor()
        };
    }
};

inline at::Tensor gegenbauer_polynomial_c_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha
) {
    return GegenbauerPolynomialCEvaluate::apply(coeffs, x, alpha);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gegenbauer_polynomial_c_evaluate", &torchscience::autograd::polynomial::gegenbauer_polynomial_c_evaluate);
}
