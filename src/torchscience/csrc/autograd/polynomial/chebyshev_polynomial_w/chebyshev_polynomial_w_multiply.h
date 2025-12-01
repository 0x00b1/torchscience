#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::polynomial {

// Backward function for second-order gradients
class ChebyshevPolynomialWMultiplyBackward
    : public torch::autograd::Function<ChebyshevPolynomialWMultiplyBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& a,
        const at::Tensor& b,
        bool a_requires_grad,
        bool b_requires_grad
    ) {
        ctx->save_for_backward({grad_output, a, b});
        ctx->saved_data["a_requires_grad"] = a_requires_grad;
        ctx->saved_data["b_requires_grad"] = b_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_a, grad_b] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_w_multiply_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, a, b);

        return {grad_a, grad_b};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor a = saved[1];
        at::Tensor b = saved[2];

        bool a_requires_grad = ctx->saved_data["a_requires_grad"].toBool();
        bool b_requires_grad = ctx->saved_data["b_requires_grad"].toBool();

        at::Tensor gg_a = grad_outputs[0];  // gradient w.r.t. grad_a
        at::Tensor gg_b = grad_outputs[1];  // gradient w.r.t. grad_b

        if (!a_requires_grad && !b_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        // Handle undefined gradients
        if (!gg_a.defined()) {
            gg_a = at::zeros_like(a);
        }
        if (!gg_b.defined()) {
            gg_b = at::zeros_like(b);
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad_a_from_gg, grad_b_from_gg] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_w_multiply_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(gg_a, gg_b, grad_output, a, b);

        return {
            grad_grad_output,
            a_requires_grad ? grad_a_from_gg : at::Tensor(),
            b_requires_grad ? grad_b_from_gg : at::Tensor(),
            at::Tensor(),  // a_requires_grad (not differentiable)
            at::Tensor()   // b_requires_grad (not differentiable)
        };
    }
};

// Forward function with first-order gradients
class ChebyshevPolynomialWMultiply
    : public torch::autograd::Function<ChebyshevPolynomialWMultiply> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        ctx->save_for_backward({a, b});

        bool a_requires_grad = a.requires_grad() &&
            (at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()));
        bool b_requires_grad = b.requires_grad() &&
            (at::isFloatingType(b.scalar_type()) || at::isComplexType(b.scalar_type()));

        ctx->saved_data["a_requires_grad"] = a_requires_grad;
        ctx->saved_data["b_requires_grad"] = b_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chebyshev_polynomial_w_multiply", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(a, b);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor a = saved[0];
        at::Tensor b = saved[1];

        bool a_requires_grad = ctx->saved_data["a_requires_grad"].toBool();
        bool b_requires_grad = ctx->saved_data["b_requires_grad"].toBool();

        if (!a_requires_grad && !b_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        auto grads = ChebyshevPolynomialWMultiplyBackward::apply(
            grad_outputs[0],
            a,
            b,
            a_requires_grad,
            b_requires_grad
        );

        return {
            a_requires_grad ? grads[0] : at::Tensor(),
            b_requires_grad ? grads[1] : at::Tensor()
        };
    }
};

inline at::Tensor chebyshev_polynomial_w_multiply(
    const at::Tensor& a,
    const at::Tensor& b
) {
    return ChebyshevPolynomialWMultiply::apply(a, b);
}

} // namespace torchscience::autograd::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("chebyshev_polynomial_w_multiply", &torchscience::autograd::polynomial::chebyshev_polynomial_w_multiply);
}
