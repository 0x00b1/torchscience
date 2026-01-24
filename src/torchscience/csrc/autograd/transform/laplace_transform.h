#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class LaplaceTransformBackward
    : public torch::autograd::Function<LaplaceTransformBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& s,
        const at::Tensor& t,
        int64_t dim,
        int64_t integration_method,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input, s, t});
        context->saved_data["dim"] = dim;
        context->saved_data["integration_method"] = integration_method;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laplace_transform_backward", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, int64_t, int64_t
            )>()
            .call(grad_output, input, s, t, dim, integration_method);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];
        at::Tensor s = saved[2];
        at::Tensor t = saved[3];

        int64_t dim = context->saved_data["dim"].toInt();
        int64_t integration_method = context->saved_data["integration_method"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] =
            c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laplace_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, int64_t, int64_t
            )>()
            .call(grad_grad_input, grad_output, input, s, t, dim, integration_method);

        return {
            grad_grad_output, new_grad_input, at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()
        };
    }
};

class LaplaceTransform
    : public torch::autograd::Function<LaplaceTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        const at::Tensor& s,
        const at::Tensor& t,
        int64_t dim,
        int64_t integration_method
    ) {
        context->save_for_backward({input, s, t});
        context->saved_data["dim"] = dim;
        context->saved_data["integration_method"] = integration_method;

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laplace_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                const at::Tensor&, int64_t, int64_t)>()
            .call(input, s, t, dim, integration_method);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor s = saved[1];
        at::Tensor t = saved[2];
        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = context->saved_data["dim"].toInt();
        int64_t integration_method = context->saved_data["integration_method"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_input = LaplaceTransformBackward::apply(
            grad_output, input, s, t, dim, integration_method, input_requires_grad
        );

        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor laplace_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    return LaplaceTransform::apply(input, s, t, dim, integration_method);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "laplace_transform",
        &torchscience::autograd::transform::laplace_transform
    );
}
