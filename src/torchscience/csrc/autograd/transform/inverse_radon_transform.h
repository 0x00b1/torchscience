#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class InverseRadonTransformBackward
    : public torch::autograd::Function<InverseRadonTransformBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& sinogram,
        const at::Tensor& angles,
        bool circle,
        int64_t output_size,
        int64_t filter_type,
        bool sinogram_requires_grad
    ) {
        context->save_for_backward({grad_output, sinogram, angles});
        context->saved_data["circle"] = circle;
        context->saved_data["output_size"] = output_size;
        context->saved_data["filter_type"] = filter_type;
        context->saved_data["sinogram_requires_grad"] = sinogram_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_radon_transform_backward", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                bool, int64_t, int64_t
            )>()
            .call(grad_output, sinogram, angles, circle, output_size, filter_type);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor sinogram = saved[1];
        at::Tensor angles = saved[2];

        bool circle = context->saved_data["circle"].toBool();
        int64_t output_size = context->saved_data["output_size"].toInt();
        int64_t filter_type = context->saved_data["filter_type"].toInt();
        bool sinogram_requires_grad = context->saved_data["sinogram_requires_grad"].toBool();

        at::Tensor grad_grad_sinogram = grad_outputs[0];

        if (!grad_grad_sinogram.defined() || !sinogram_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_sinogram] =
            c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_radon_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, bool, int64_t, int64_t
            )>()
            .call(grad_grad_sinogram, grad_output, sinogram, angles, circle, output_size, filter_type);

        return {grad_grad_output, new_grad_sinogram, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class InverseRadonTransform
    : public torch::autograd::Function<InverseRadonTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& sinogram,
        const at::Tensor& angles,
        bool circle,
        int64_t output_size,
        int64_t filter_type
    ) {
        context->save_for_backward({sinogram, angles});
        context->saved_data["circle"] = circle;
        context->saved_data["output_size"] = output_size;
        context->saved_data["filter_type"] = filter_type;

        bool sinogram_requires_grad = sinogram.requires_grad() &&
            at::isFloatingType(sinogram.scalar_type());
        context->saved_data["sinogram_requires_grad"] = sinogram_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_radon_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, bool, int64_t, int64_t)>()
            .call(sinogram, angles, circle, output_size, filter_type);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor sinogram = saved[0];
        at::Tensor angles = saved[1];
        at::Tensor grad_output = grad_outputs[0];

        bool circle = context->saved_data["circle"].toBool();
        int64_t output_size = context->saved_data["output_size"].toInt();
        int64_t filter_type = context->saved_data["filter_type"].toInt();
        bool sinogram_requires_grad = context->saved_data["sinogram_requires_grad"].toBool();

        if (!sinogram_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_sinogram = InverseRadonTransformBackward::apply(
            grad_output, sinogram, angles, circle, output_size, filter_type, sinogram_requires_grad
        );

        return {grad_sinogram, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor inverse_radon_transform(
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    return InverseRadonTransform::apply(sinogram, angles, circle, output_size, filter_type);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "inverse_radon_transform",
        &torchscience::autograd::transform::inverse_radon_transform
    );
}
