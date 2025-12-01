#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

class GegenbauerPolynomialCMulxBackward
    : public torch::autograd::Function<GegenbauerPolynomialCMulxBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& alpha,
        bool coeffs_requires_grad,
        bool alpha_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs, alpha});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_mulx_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(grad_output, coeffs, alpha);

        return std::get<0>(result);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];
        at::Tensor alpha = saved[2];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

        if (!coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor gg_coeffs = grad_outputs[0];

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_mulx_backward_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(gg_coeffs, coeffs, alpha);

        return {
            grad_grad_output,
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

class GegenbauerPolynomialCMulx
    : public torch::autograd::Function<GegenbauerPolynomialCMulx> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs,
        const at::Tensor& alpha
    ) {
        ctx->save_for_backward({coeffs, alpha});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));
        bool alpha_requires_grad = alpha.requires_grad() &&
            (at::isFloatingType(alpha.scalar_type()) || at::isComplexType(alpha.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_mulx", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(coeffs, alpha);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor alpha = saved[1];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();

        if (!coeffs_requires_grad && !alpha_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_mulx_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(grad_outputs[0], coeffs, alpha);

        at::Tensor grad_coeffs = coeffs_requires_grad ? std::get<0>(result) : at::Tensor();
        at::Tensor grad_alpha = alpha_requires_grad ? std::get<1>(result) : at::Tensor();

        return {grad_coeffs, grad_alpha};
    }
};

inline at::Tensor gegenbauer_polynomial_c_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha
) {
    return GegenbauerPolynomialCMulx::apply(coeffs, alpha);
}

} // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gegenbauer_polynomial_c_mulx", &torchscience::autograd::polynomial::gegenbauer_polynomial_c_mulx);
}
