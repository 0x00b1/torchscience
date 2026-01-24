#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class InverseHankelTransformBackward
    : public torch::autograd::Function<InverseHankelTransformBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& r_out,
        const at::Tensor& k_in,
        int64_t dim,
        double order,
        int64_t integration_method,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input, r_out, k_in});
        context->saved_data["dim"] = dim;
        context->saved_data["order"] = order;
        context->saved_data["integration_method"] = integration_method;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hankel_transform_backward", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, int64_t, double, int64_t
            )>()
            .call(grad_output, input, r_out, k_in, dim, order, integration_method);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];
        at::Tensor r_out = saved[2];
        at::Tensor k_in = saved[3];

        int64_t dim = context->saved_data["dim"].toInt();
        double order = context->saved_data["order"].toDouble();
        int64_t integration_method = context->saved_data["integration_method"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] =
            c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hankel_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, int64_t, double, int64_t
            )>()
            .call(grad_grad_input, grad_output, input, r_out, k_in, dim, order, integration_method);

        return {
            grad_grad_output, new_grad_input, at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
        };
    }
};

class InverseHankelTransform
    : public torch::autograd::Function<InverseHankelTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        const at::Tensor& r_out,
        const at::Tensor& k_in,
        int64_t dim,
        double order,
        int64_t integration_method
    ) {
        context->save_for_backward({input, r_out, k_in});
        context->saved_data["dim"] = dim;
        context->saved_data["order"] = order;
        context->saved_data["integration_method"] = integration_method;

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_hankel_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                const at::Tensor&, int64_t, double, int64_t)>()
            .call(input, r_out, k_in, dim, order, integration_method);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor r_out = saved[1];
        at::Tensor k_in = saved[2];
        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = context->saved_data["dim"].toInt();
        double order = context->saved_data["order"].toDouble();
        int64_t integration_method = context->saved_data["integration_method"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_input = InverseHankelTransformBackward::apply(
            grad_output, input, r_out, k_in, dim, order, integration_method, input_requires_grad
        );

        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor inverse_hankel_transform(
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& k_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    return InverseHankelTransform::apply(input, r_out, k_in, dim, order, integration_method);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "inverse_hankel_transform",
        &torchscience::autograd::transform::inverse_hankel_transform
    );
}
