#pragma once

#include <tuple>

#include <torch/extension.h>

namespace torchscience::autograd::test_functions {

/**
 * Backward function class for double-backward support.
 */
class RosenbrockBackward
    : public torch::autograd::Function<RosenbrockBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& x,
        const at::Tensor& a,
        const at::Tensor& b,
        const bool x_requires_grad,
        const bool a_requires_grad,
        const bool b_requires_grad
    ) {
        context->save_for_backward({grad_output, x, a, b});
        context->saved_data["x_requires_grad"] = x_requires_grad;
        context->saved_data["a_requires_grad"] = a_requires_grad;
        context->saved_data["b_requires_grad"] = b_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_x, grad_a, grad_b] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::rosenbrock_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_output, x, a, b);

        return {grad_x, grad_a, grad_b};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        const bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
        const bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();
        const bool b_requires_grad = context->saved_data["b_requires_grad"].toBool();

        const bool grad_grad_x_defined = gradient_outputs[0].defined();
        const bool grad_grad_a_defined = gradient_outputs[1].defined();
        const bool grad_grad_b_defined = gradient_outputs[2].defined();

        if (!(grad_grad_x_defined && x_requires_grad) &&
            !(grad_grad_a_defined && a_requires_grad) &&
            !(grad_grad_b_defined && b_requires_grad)) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_x
                at::Tensor(),  // grad_a
                at::Tensor(),  // grad_b
                at::Tensor(),  // grad_x_requires_grad
                at::Tensor(),  // grad_a_requires_grad
                at::Tensor()   // grad_b_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_grad_x_input;
        at::Tensor grad_grad_a_input;
        at::Tensor grad_grad_b_input;

        if (grad_grad_x_defined && x_requires_grad) {
            grad_grad_x_input = gradient_outputs[0];
        }
        if (grad_grad_a_defined && a_requires_grad) {
            grad_grad_a_input = gradient_outputs[1];
        }
        if (grad_grad_b_defined && b_requires_grad) {
            grad_grad_b_input = gradient_outputs[2];
        }

        auto [grad_grad_output, grad_x, grad_a, grad_b] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::rosenbrock_backward_backward", "")
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
                    grad_grad_x_input,
                    grad_grad_a_input,
                    grad_grad_b_input,
                    saved[0],  // grad_output
                    saved[1],  // x
                    saved[2],  // a
                    saved[3]   // b
                );

        return {
            grad_grad_output,
            grad_x,
            grad_a,
            grad_b,
            at::Tensor(),  // grad_x_requires_grad
            at::Tensor(),  // grad_a_requires_grad
            at::Tensor()   // grad_b_requires_grad
        };
    }
};

/**
 * Forward function class with autograd support.
 */
class Rosenbrock
    : public torch::autograd::Function<Rosenbrock> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& x,
        const at::Tensor& a,
        const at::Tensor& b
    ) {
        context->save_for_backward({x, a, b});

        const bool is_differentiable = at::isFloatingType(x.scalar_type()) || at::isComplexType(x.scalar_type());

        context->saved_data["x_requires_grad"] = x.requires_grad() && is_differentiable;
        context->saved_data["a_requires_grad"] = a.requires_grad() && is_differentiable;
        context->saved_data["b_requires_grad"] = b.requires_grad() && is_differentiable;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::rosenbrock", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(x, a, b);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();

        at::Tensor x = saved[0];
        at::Tensor a = saved[1];
        at::Tensor b = saved[2];
        at::Tensor grad_output = gradient_outputs[0];

        bool x_requires_grad = context->saved_data["x_requires_grad"].toBool();
        bool a_requires_grad = context->saved_data["a_requires_grad"].toBool();
        bool b_requires_grad = context->saved_data["b_requires_grad"].toBool();

        std::vector<at::Tensor> gradients = RosenbrockBackward::apply(
            grad_output,
            x,
            a,
            b,
            x_requires_grad,
            a_requires_grad,
            b_requires_grad
        );

        at::Tensor grad_x;
        at::Tensor grad_a;
        at::Tensor grad_b;

        if (x_requires_grad) {
            grad_x = gradients[0];
        }
        if (a_requires_grad) {
            grad_a = gradients[1];
        }
        if (b_requires_grad) {
            grad_b = gradients[2];
        }

        return {grad_x, grad_a, grad_b};
    }
};

// Wrapper function
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    return Rosenbrock::apply(x, a, b);
}

}  // namespace torchscience::autograd::test_functions

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("rosenbrock", &torchscience::autograd::test_functions::rosenbrock);
}
