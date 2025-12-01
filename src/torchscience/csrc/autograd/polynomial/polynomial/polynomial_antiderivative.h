#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class PolynomialAntiderivativeBackward
    : public torch::autograd::Function<PolynomialAntiderivativeBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& constant,
        bool coeffs_requires_grad,
        bool constant_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs, constant});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["constant_requires_grad"] = constant_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_coeffs, grad_constant] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_antiderivative_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, coeffs, constant);

        return {grad_coeffs, grad_constant};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];
        at::Tensor constant = saved[2];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool constant_requires_grad = ctx->saved_data["constant_requires_grad"].toBool();

        at::Tensor gg_coeffs = grad_outputs[0];    // gradient w.r.t. grad_coeffs
        at::Tensor gg_constant = grad_outputs[1];  // gradient w.r.t. grad_constant

        if (!coeffs_requires_grad && !constant_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_antiderivative_backward_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_coeffs.defined() ? gg_coeffs : at::zeros_like(coeffs),
                gg_constant.defined() ? gg_constant : at::zeros_like(constant),
                coeffs
            );

        // For the antiderivative, the operator is linear, so:
        // - There is no gradient flowing back to coeffs from the backward pass
        // - There is no gradient flowing back to constant from the backward pass
        return {
            grad_grad_output,
            at::Tensor(),  // coeffs (not needed in backward of backward)
            at::Tensor(),  // constant (not needed in backward of backward)
            at::Tensor(),  // coeffs_requires_grad (not differentiable)
            at::Tensor()   // constant_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class PolynomialAntiderivative
    : public torch::autograd::Function<PolynomialAntiderivative> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs,
        const at::Tensor& constant
    ) {
        ctx->save_for_backward({coeffs, constant});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));
        bool constant_requires_grad = constant.requires_grad() &&
            (at::isFloatingType(constant.scalar_type()) || at::isComplexType(constant.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["constant_requires_grad"] = constant_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::polynomial_antiderivative", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(coeffs, constant);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor constant = saved[1];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool constant_requires_grad = ctx->saved_data["constant_requires_grad"].toBool();

        if (!coeffs_requires_grad && !constant_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        auto grads = PolynomialAntiderivativeBackward::apply(
            grad_outputs[0],
            coeffs,
            constant,
            coeffs_requires_grad,
            constant_requires_grad
        );

        return {
            coeffs_requires_grad ? grads[0] : at::Tensor(),
            constant_requires_grad ? grads[1] : at::Tensor()
        };
    }
};

inline at::Tensor polynomial_antiderivative(
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    return PolynomialAntiderivative::apply(coeffs, constant);
}

}  // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("polynomial_antiderivative", &torchscience::autograd::polynomial::polynomial_antiderivative);
}
