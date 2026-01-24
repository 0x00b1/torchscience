#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::transform {

class ConvolutionBackward
    : public torch::autograd::Function<ConvolutionBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& kernel,
        int64_t dim,
        int64_t mode,
        bool input_requires_grad,
        bool kernel_requires_grad
    ) {
        context->save_for_backward({grad_output, input, kernel});
        context->saved_data["dim"] = dim;
        context->saved_data["mode"] = mode;
        context->saved_data["input_requires_grad"] = input_requires_grad;
        context->saved_data["kernel_requires_grad"] = kernel_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [grad_input, grad_kernel] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::convolution_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t
            )>()
            .call(grad_output, input, kernel, dim, mode);

        return {grad_input, grad_kernel};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor input = saved[1];
        at::Tensor kernel = saved[2];

        int64_t dim = context->saved_data["dim"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();
        bool kernel_requires_grad = context->saved_data["kernel_requires_grad"].toBool();

        at::Tensor grad_grad_input = grad_outputs[0];
        at::Tensor grad_grad_kernel = grad_outputs[1];

        if ((!grad_grad_input.defined() || !input_requires_grad) &&
            (!grad_grad_kernel.defined() || !kernel_requires_grad)) {
            return {
                at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, new_grad_input, new_grad_kernel] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::convolution_backward_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, int64_t, int64_t
            )>()
            .call(grad_grad_input, grad_grad_kernel, grad_output, input, kernel, dim, mode);

        return {
            grad_grad_output, new_grad_input, new_grad_kernel,
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()
        };
    }
};

class Convolution
    : public torch::autograd::Function<Convolution> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& input,
        const at::Tensor& kernel,
        int64_t dim,
        int64_t mode
    ) {
        context->save_for_backward({input, kernel});
        context->saved_data["dim"] = dim;
        context->saved_data["mode"] = mode;

        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        bool kernel_requires_grad = kernel.requires_grad() &&
            at::isFloatingType(kernel.scalar_type());
        context->saved_data["input_requires_grad"] = input_requires_grad;
        context->saved_data["kernel_requires_grad"] = kernel_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::convolution", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
            .call(input, kernel, dim, mode);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor kernel = saved[1];
        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = context->saved_data["dim"].toInt();
        int64_t mode = context->saved_data["mode"].toInt();
        bool input_requires_grad = context->saved_data["input_requires_grad"].toBool();
        bool kernel_requires_grad = context->saved_data["kernel_requires_grad"].toBool();

        if (!input_requires_grad && !kernel_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        std::vector<at::Tensor> gradients = ConvolutionBackward::apply(
            grad_output, input, kernel, dim, mode, input_requires_grad, kernel_requires_grad
        );

        at::Tensor grad_input = input_requires_grad ? gradients[0] : at::Tensor();
        at::Tensor grad_kernel = kernel_requires_grad ? gradients[1] : at::Tensor();

        return {grad_input, grad_kernel, at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    return Convolution::apply(input, kernel, dim, mode);
}

}  // namespace torchscience::autograd::transform

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "convolution",
        &torchscience::autograd::transform::convolution
    );
}
