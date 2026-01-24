#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

/**
 * Backward function class for inverse Fourier transform double-backward support.
 */
class InverseFourierTransformBackward
    : public torch::autograd::Function<InverseFourierTransformBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        int64_t n,
        int64_t dim,
        int64_t padding_mode,
        double padding_value,
        const c10::optional<at::Tensor>& window,
        int64_t norm,
        bool input_requires_grad
    ) {
        context->save_for_backward({grad_output, input});
        if (window.has_value()) {
            context->saved_data["window"] = window.value();
        }
        context->saved_data["n"] = n;
        context->saved_data["dim"] = dim;
        context->saved_data["padding_mode"] = padding_mode;
        context->saved_data["padding_value"] = padding_value;
        context->saved_data["norm"] = norm;
        context->saved_data["has_window"] = window.has_value();
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_fourier_transform_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&, int64_t)>()
            .call(grad_output, input, n, dim, padding_mode, padding_value, window, norm);

        return {grad_input};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];

        int64_t n = context->saved_data["n"].toInt();
        int64_t dim = context->saved_data["dim"].toInt();
        int64_t padding_mode = context->saved_data["padding_mode"].toInt();
        double padding_value = context->saved_data["padding_value"].toDouble();
        int64_t norm = context->saved_data["norm"].toInt();
        bool has_window = context->saved_data["has_window"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        c10::optional<at::Tensor> window = c10::nullopt;
        if (has_window) {
            window = context->saved_data["window"].toTensor();
        }

        at::Tensor grad_grad_input = grad_outputs[0];

        if (!grad_grad_input.defined() || !input_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_input
                at::Tensor(),  // grad_n
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_padding_mode
                at::Tensor(),  // grad_padding_value
                at::Tensor(),  // grad_window
                at::Tensor(),  // grad_norm
                at::Tensor()   // grad_input_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_fourier_transform_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&, int64_t
            )>()
            .call(grad_grad_input, grad_output, input, n, dim, padding_mode, padding_value, window, norm);

        return {
            grad_grad_output,
            new_grad_input,
            at::Tensor(),  // grad_n
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_padding_mode
            at::Tensor(),  // grad_padding_value
            at::Tensor(),  // grad_window
            at::Tensor(),  // grad_norm
            at::Tensor()   // grad_input_requires_grad
        };
    }
};

/**
 * Forward function class for inverse Fourier transform with autograd support.
 */
class InverseFourierTransform
    : public torch::autograd::Function<InverseFourierTransform> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        int64_t n,
        int64_t dim,
        int64_t padding_mode,
        double padding_value,
        const c10::optional<at::Tensor>& window,
        int64_t norm
    ) {
        context->save_for_backward({input});
        if (window.has_value()) {
            context->saved_data["window"] = window.value();
        }
        context->saved_data["n"] = n;
        context->saved_data["dim"] = dim;
        context->saved_data["padding_mode"] = padding_mode;
        context->saved_data["padding_value"] = padding_value;
        context->saved_data["norm"] = norm;
        context->saved_data["has_window"] = window.has_value();

        bool input_requires_grad = input.requires_grad() &&
            (at::isFloatingType(input.scalar_type()) || at::isComplexType(input.scalar_type()));
        context->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::inverse_fourier_transform", "")
            .typed<at::Tensor(const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&, int64_t)>()
            .call(input, n, dim, padding_mode, padding_value, window, norm);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor grad_output = grad_outputs[0];

        int64_t n = context->saved_data["n"].toInt();
        int64_t dim = context->saved_data["dim"].toInt();
        int64_t padding_mode = context->saved_data["padding_mode"].toInt();
        double padding_value = context->saved_data["padding_value"].toDouble();
        int64_t norm = context->saved_data["norm"].toInt();
        bool has_window = context->saved_data["has_window"].toBool();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();

        c10::optional<at::Tensor> window = c10::nullopt;
        if (has_window) {
            window = context->saved_data["window"].toTensor();
        }

        if (!input_requires_grad) {
            return {
                at::Tensor(),  // grad_input
                at::Tensor(),  // grad_n
                at::Tensor(),  // grad_dim
                at::Tensor(),  // grad_padding_mode
                at::Tensor(),  // grad_padding_value
                at::Tensor(),  // grad_window
                at::Tensor()   // grad_norm
            };
        }

        std::vector<at::Tensor> gradients = InverseFourierTransformBackward::apply(
            grad_output,
            input,
            n,
            dim,
            padding_mode,
            padding_value,
            window,
            norm,
            input_requires_grad
        );

        return {
            gradients[0],  // grad_input
            at::Tensor(),  // grad_n
            at::Tensor(),  // grad_dim
            at::Tensor(),  // grad_padding_mode
            at::Tensor(),  // grad_padding_value
            at::Tensor(),  // grad_window
            at::Tensor()   // grad_norm
        };
    }
};

inline at::Tensor inverse_fourier_transform(
    const at::Tensor& input,
    int64_t n,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window,
    int64_t norm
) {
    return InverseFourierTransform::apply(input, n, dim, padding_mode, padding_value, window, norm);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "inverse_fourier_transform",
        &torchscience::autograd::transform::inverse_fourier_transform
    );
}
