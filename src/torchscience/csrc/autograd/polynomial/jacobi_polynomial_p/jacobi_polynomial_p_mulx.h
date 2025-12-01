#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

class JacobiPolynomialPMulxBackward
    : public torch::autograd::Function<JacobiPolynomialPMulxBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& alpha,
        const at::Tensor& beta,
        bool coeffs_requires_grad,
        bool alpha_requires_grad,
        bool beta_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs, alpha, beta});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;
        ctx->saved_data["beta_requires_grad"] = beta_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jacobi_polynomial_p_mulx_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(grad_output, coeffs, alpha, beta);

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
        at::Tensor beta = saved[3];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

        if (!coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor gg_coeffs = grad_outputs[0];

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jacobi_polynomial_p_mulx_backward_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(gg_coeffs, coeffs, alpha, beta);

        return {
            grad_grad_output,
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

class JacobiPolynomialPMulx
    : public torch::autograd::Function<JacobiPolynomialPMulx> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& coeffs,
        const at::Tensor& alpha,
        const at::Tensor& beta
    ) {
        ctx->save_for_backward({coeffs, alpha, beta});

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            (at::isFloatingType(coeffs.scalar_type()) || at::isComplexType(coeffs.scalar_type()));
        bool alpha_requires_grad = alpha.requires_grad() &&
            (at::isFloatingType(alpha.scalar_type()) || at::isComplexType(alpha.scalar_type()));
        bool beta_requires_grad = beta.requires_grad() &&
            (at::isFloatingType(beta.scalar_type()) || at::isComplexType(beta.scalar_type()));

        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;
        ctx->saved_data["alpha_requires_grad"] = alpha_requires_grad;
        ctx->saved_data["beta_requires_grad"] = beta_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jacobi_polynomial_p_mulx", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(coeffs, alpha, beta);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor alpha = saved[1];
        at::Tensor beta = saved[2];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();
        bool alpha_requires_grad = ctx->saved_data["alpha_requires_grad"].toBool();
        bool beta_requires_grad = ctx->saved_data["beta_requires_grad"].toBool();

        if (!coeffs_requires_grad && !alpha_requires_grad && !beta_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto result = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jacobi_polynomial_p_mulx_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(grad_outputs[0], coeffs, alpha, beta);

        at::Tensor grad_coeffs = coeffs_requires_grad ? std::get<0>(result) : at::Tensor();
        at::Tensor grad_alpha = alpha_requires_grad ? std::get<1>(result) : at::Tensor();
        at::Tensor grad_beta = beta_requires_grad ? std::get<2>(result) : at::Tensor();

        return {grad_coeffs, grad_alpha, grad_beta};
    }
};

inline at::Tensor jacobi_polynomial_p_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    return JacobiPolynomialPMulx::apply(coeffs, alpha, beta);
}

} // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("jacobi_polynomial_p_mulx", &torchscience::autograd::polynomial::jacobi_polynomial_p_mulx);
}
