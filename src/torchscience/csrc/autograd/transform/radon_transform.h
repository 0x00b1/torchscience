#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class RadonTransformBackward
    : public torch::autograd::Function<RadonTransformBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& angles,
        bool circle,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input, angles});
        context->saved_data["circle"] = circle;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::radon_transform_backward", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, bool
            )>()
            .call(grad_output, input, angles, circle);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];
        at::Tensor angles = saved[2];

        bool circle = context->saved_data["circle"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] =
            c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::radon_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, bool
            )>()
            .call(grad_grad_input, grad_output, input, angles, circle);

        return {grad_grad_output, new_grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class RadonTransform
    : public torch::autograd::Function<RadonTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        const at::Tensor& angles,
        bool circle
    ) {
        context->save_for_backward({input, angles});
        context->saved_data["circle"] = circle;

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::radon_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, bool)>()
            .call(input, angles, circle);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor angles = saved[1];
        at::Tensor grad_output = grad_outputs[0];

        bool circle = context->saved_data["circle"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_input = RadonTransformBackward::apply(
            grad_output, input, angles, circle, input_requires_grad
        );

        return {grad_input, at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor radon_transform(
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    return RadonTransform::apply(input, angles, circle);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "radon_transform",
        &torchscience::autograd::transform::radon_transform
    );
}
