#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::optimization::combinatorial {

class Sinkhorn : public torch::autograd::Function<Sinkhorn> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& C,
        const at::Tensor& a,
        const at::Tensor& b,
        double epsilon,
        int64_t maxiter,
        double tol
    ) {
        at::AutoDispatchBelowAutograd guard;

        at::Tensor P = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::sinkhorn", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                int64_t,
                double
            )>()
            .call(C, a, b, epsilon, maxiter, tol);

        context->save_for_backward({P, C});
        context->saved_data["epsilon"] = epsilon;

        return P;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        double epsilon = context->saved_data["epsilon"].toDouble();

        at::Tensor P = saved[0];
        at::Tensor C = saved[1];
        at::Tensor grad_output = gradient_outputs[0];

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_C = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::sinkhorn_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double
            )>()
            .call(grad_output, P, C, epsilon);

        // Return gradients for: C, a, b, epsilon, maxiter, tol
        // Only C gets a gradient; others are not differentiable
        return {grad_C, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    return Sinkhorn::apply(C, a, b, epsilon, maxiter, tol);
}

}  // namespace torchscience::autograd::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "sinkhorn",
        &torchscience::autograd::optimization::combinatorial::sinkhorn
    );
}
