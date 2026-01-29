#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class DiscreteWaveletTransformBackward
    : public torch::autograd::Function<DiscreteWaveletTransformBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& filter_lo,
        const at::Tensor& filter_hi,
        int64_t levels,
        int64_t mode,
        int64_t input_length,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input, filter_lo, filter_hi});
        context->saved_data["levels"] = levels;
        context->saved_data["mode"] = mode;
        context->saved_data["input_length"] = input_length;
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::discrete_wavelet_transform_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
            .call(grad_output, input, filter_lo, filter_hi, levels, mode, input_length);

        return {grad_input};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];
        at::Tensor filter_lo = saved[2];
        at::Tensor filter_hi = saved[3];

        int64_t levels = context->saved_data["levels"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        int64_t input_length = context->saved_data["input_length"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::discrete_wavelet_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
            .call(grad_grad_input, grad_output, input, filter_lo, filter_hi, levels, mode, input_length);

        return {grad_grad_output, new_grad_input, at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class DiscreteWaveletTransform
    : public torch::autograd::Function<DiscreteWaveletTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        const at::Tensor& filter_lo,
        const at::Tensor& filter_hi,
        int64_t levels,
        int64_t mode
    ) {
        context->save_for_backward({input, filter_lo, filter_hi});
        context->saved_data["levels"] = levels;
        context->saved_data["mode"] = mode;
        context->saved_data["input_length"] = input.size(-1);

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::discrete_wavelet_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
            .call(input, filter_lo, filter_hi, levels, mode);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor filter_lo = saved[1];
        at::Tensor filter_hi = saved[2];
        at::Tensor grad_output = grad_outputs[0];

        int64_t levels = context->saved_data["levels"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        int64_t input_length = context->saved_data["input_length"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        std::vector<at::Tensor> gradients = DiscreteWaveletTransformBackward::apply(
            grad_output, input, filter_lo, filter_hi, levels, mode, input_length, input_requires_grad
        );

        return {gradients[0], at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode
) {
    return DiscreteWaveletTransform::apply(input, filter_lo, filter_hi, levels, mode);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::autograd::transform::discrete_wavelet_transform
    );
}
