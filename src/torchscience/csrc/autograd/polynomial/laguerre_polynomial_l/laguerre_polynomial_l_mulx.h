#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

class LaguerrePolynomialLMulxBackward
    : public torch::autograd::Function<LaguerrePolynomialLMulxBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        bool coeffs_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laguerre_polynomial_l_mulx_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_output, coeffs);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

        if (!coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor gg_coeffs = grad_outputs[0];

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laguerre_polynomial_l_mulx_backward_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(gg_coeffs, coeffs);

        return {
            grad_grad_output,
            at::Tensor(),
            at::Tensor()
        };
    }
};

class LaguerrePolynomialLMulx
    : public torch::autograd::Function<LaguerrePolynomialLMulx> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs
    ) {
        ctx->save_for_backward({coeffs});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laguerre_polynomial_l_mulx", "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(coeffs);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

        if (!coeffs_requires_grad) {
            return {at::Tensor()};
        }

        at::Tensor grad_coeffs = LaguerrePolynomialLMulxBackward::apply(
            grad_outputs[0],
            coeffs,
            coeffs_requires_grad
        );

        return {grad_coeffs};
    }
};

inline at::Tensor laguerre_polynomial_l_mulx(const at::Tensor& coeffs) {
    return LaguerrePolynomialLMulx::apply(coeffs);
}

} // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("laguerre_polynomial_l_mulx", &torchscience::autograd::polynomial::laguerre_polynomial_l_mulx);
}
