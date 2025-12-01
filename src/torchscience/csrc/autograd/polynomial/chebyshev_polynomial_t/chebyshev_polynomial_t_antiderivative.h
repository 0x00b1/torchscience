#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class ChebyshevPolynomialTAntiderivativeBackward
    : public torch::autograd::Function<ChebyshevPolynomialTAntiderivativeBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        bool coeffs_requires_grad
    ) {
        ctx->save_for_backward({grad_output, coeffs});
        ctx->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto grad_coeffs = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_antiderivative_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, coeffs);

        return {grad_coeffs};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];

        bool coeffs_requires_grad = ctx->saved_data["coeffs_requires_grad"].toBool();

        at::Tensor gg_coeffs = grad_outputs[0];

        if (!coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto grad_grad_output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_antiderivative_backward_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(
                gg_coeffs.defined() ? gg_coeffs : at::zeros_like(coeffs),
                grad_output,
                coeffs
            );

        return {
            grad_grad_output,
            at::Tensor(),  // No gradient for coeffs (linear operation, already handled)
            at::Tensor()   // coeffs_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class ChebyshevPolynomialTAntiderivative
    : public torch::autograd::Function<ChebyshevPolynomialTAntiderivative> {
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
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_antiderivative", "")
            .typed<at::Tensor(
                const at::Tensor&
            )>()
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

        auto grads = ChebyshevPolynomialTAntiderivativeBackward::apply(
            grad_outputs[0],
            coeffs,
            coeffs_requires_grad
        );

        return {
            coeffs_requires_grad ? grads[0] : at::Tensor()
        };
    }
};

inline at::Tensor chebyshev_polynomial_t_antiderivative(
    const at::Tensor& coeffs
) {
    return ChebyshevPolynomialTAntiderivative::apply(coeffs);
}

} // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("chebyshev_polynomial_t_antiderivative", &torchscience::autograd::polynomial::chebyshev_polynomial_t_antiderivative);
}
