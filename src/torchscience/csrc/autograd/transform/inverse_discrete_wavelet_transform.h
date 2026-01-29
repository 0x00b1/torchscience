#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class InverseDiscreteWaveletTransformBackward
    : public torch::autograd::Function<InverseDiscreteWaveletTransformBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& coeffs,
        const at::Tensor& filter_lo,
        const at::Tensor& filter_hi,
        int64_t levels,
        int64_t mode,
        int64_t output_length,
        bool coeffs_requires_grad
    ) {
        context->save_for_backward({grad_output, coeffs, filter_lo, filter_hi});
        context->saved_data["levels"] = levels;
        context->saved_data["mode"] = mode;
        context->saved_data["output_length"] = output_length;
        context->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto grad_coeffs = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_discrete_wavelet_transform_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
            .call(grad_output, coeffs, filter_lo, filter_hi, levels, mode, output_length);

        return {grad_coeffs};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor coeffs = saved[1];
        at::Tensor filter_lo = saved[2];
        at::Tensor filter_hi = saved[3];

        int64_t levels = context->saved_data["levels"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        int64_t output_length = context->saved_data["output_length"].toInt();
        bool coeffs_requires_grad = context->saved_data["coeffs_requires_grad"].toBool();

        at::Tensor grad_grad_coeffs = grad_outputs[0];

        if (!grad_grad_coeffs.defined() || !coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_coeffs] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_discrete_wavelet_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
            .call(grad_grad_coeffs, grad_output, coeffs, filter_lo, filter_hi, levels, mode, output_length);

        return {grad_grad_output, new_grad_coeffs, at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

class InverseDiscreteWaveletTransform
    : public torch::autograd::Function<InverseDiscreteWaveletTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& coeffs,
        const at::Tensor& filter_lo,
        const at::Tensor& filter_hi,
        int64_t levels,
        int64_t mode,
        int64_t output_length
    ) {
        context->save_for_backward({coeffs, filter_lo, filter_hi});
        context->saved_data["levels"] = levels;
        context->saved_data["mode"] = mode;
        context->saved_data["output_length"] = output_length;

        bool coeffs_requires_grad = coeffs.requires_grad() &&
            at::isFloatingType(coeffs.scalar_type());
        context->saved_data["coeffs_requires_grad"] = coeffs_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_discrete_wavelet_transform", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
            .call(coeffs, filter_lo, filter_hi, levels, mode, output_length);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor coeffs = saved[0];
        at::Tensor filter_lo = saved[1];
        at::Tensor filter_hi = saved[2];
        at::Tensor grad_output = grad_outputs[0];

        int64_t levels = context->saved_data["levels"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        int64_t output_length = context->saved_data["output_length"].toInt();
        bool coeffs_requires_grad = context->saved_data["coeffs_requires_grad"].toBool();

        if (!coeffs_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        std::vector<at::Tensor> gradients = InverseDiscreteWaveletTransformBackward::apply(
            grad_output, coeffs, filter_lo, filter_hi, levels, mode, output_length, coeffs_requires_grad
        );

        return {gradients[0], at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor inverse_discrete_wavelet_transform(
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    return InverseDiscreteWaveletTransform::apply(coeffs, filter_lo, filter_hi, levels, mode, output_length);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "inverse_discrete_wavelet_transform",
        &torchscience::autograd::transform::inverse_discrete_wavelet_transform
    );
}
