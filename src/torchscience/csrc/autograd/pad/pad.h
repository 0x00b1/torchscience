#pragma once

#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::pad {

// Second-order backward function (backward of backward)
class PadBackwardFunction : public torch::autograd::Function<PadBackwardFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        std::vector<int64_t> input_shape,
        std::vector<int64_t> padding,
        std::string mode,
        c10::optional<std::vector<int64_t>> dim,
        int64_t order,
        bool grad_output_requires_grad
    ) {
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["mode"] = mode;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["order"] = order;
        ctx->saved_data["grad_output_requires_grad"] = grad_output_requires_grad;

        at::AutoDispatchBelowAutograd guard;
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::pad_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                std::vector<int64_t>,
                std::vector<int64_t>,
                std::string,
                c10::optional<std::vector<int64_t>>,
                int64_t
            )>()
            .call(grad_output, input_shape, padding, mode, dim, order);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_grad_inputs
    ) {
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto mode = ctx->saved_data["mode"].toStringRef();
        auto dim = ctx->saved_data["dim"].toOptional<std::vector<int64_t>>();
        auto order = ctx->saved_data["order"].toInt();
        bool grad_output_requires_grad = ctx->saved_data["grad_output_requires_grad"].toBool();

        at::Tensor grad_grad_output;

        if (grad_output_requires_grad && grad_grad_inputs[0].defined()) {
            at::AutoDispatchBelowAutograd guard;
            // pad_backward_backward is just the forward pad operation on the grad_grad_input
            grad_grad_output = c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::pad_backward_backward", "")
                .typed<at::Tensor(
                    const at::Tensor&,
                    std::vector<int64_t>,
                    std::string,
                    c10::optional<std::vector<int64_t>>,
                    int64_t
                )>()
                .call(grad_grad_inputs[0], padding, mode, dim, order);
        }

        // Return gradients for: grad_output, input_shape, padding, mode, dim, order, grad_output_requires_grad
        return {
            grad_grad_output,      // grad w.r.t. grad_output
            at::Tensor(),          // input_shape has no gradient
            at::Tensor(),          // padding has no gradient
            at::Tensor(),          // mode has no gradient
            at::Tensor(),          // dim has no gradient
            at::Tensor(),          // order has no gradient
            at::Tensor()           // grad_output_requires_grad has no gradient
        };
    }
};

// Main forward function
class PadFunction : public torch::autograd::Function<PadFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        std::vector<int64_t> padding,
        std::string mode,
        double value,
        c10::optional<std::vector<int64_t>> dim,
        int64_t order,
        c10::optional<at::Tensor> out
    ) {
        ctx->saved_data["input_shape"] = input.sizes().vec();
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["mode"] = mode;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["order"] = order;
        ctx->saved_data["input_requires_grad"] = input.requires_grad();

        at::AutoDispatchBelowAutograd guard;
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::pad", "")
            .typed<at::Tensor(
                const at::Tensor&,
                std::vector<int64_t>,
                std::string,
                double,
                c10::optional<std::vector<int64_t>>,
                int64_t,
                c10::optional<at::Tensor>
            )>()
            .call(input, padding, mode, value, dim, order, out);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto input_shape = ctx->saved_data["input_shape"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto mode = ctx->saved_data["mode"].toStringRef();
        auto dim = ctx->saved_data["dim"].toOptional<std::vector<int64_t>>();
        auto order = ctx->saved_data["order"].toInt();
        bool input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();

        at::Tensor grad_input;

        if (input_requires_grad && grad_outputs[0].defined()) {
            // Use the differentiable backward function for second-order gradients
            grad_input = PadBackwardFunction::apply(
                grad_outputs[0],
                input_shape,
                padding,
                mode,
                dim,
                order,
                grad_outputs[0].requires_grad()
            );
        }

        // Return gradients for: input, padding, mode, value, dim, order, out
        return {
            grad_input,            // grad w.r.t. input
            at::Tensor(),          // padding has no gradient
            at::Tensor(),          // mode has no gradient
            at::Tensor(),          // value has no gradient
            at::Tensor(),          // dim has no gradient
            at::Tensor(),          // order has no gradient
            at::Tensor()           // out has no gradient
        };
    }
};

inline at::Tensor pad(
    const at::Tensor& input,
    std::vector<int64_t> padding,
    std::string mode,
    double value,
    c10::optional<std::vector<int64_t>> dim,
    int64_t order,
    c10::optional<at::Tensor> out
) {
    return PadFunction::apply(input, padding, mode, value, dim, order, out);
}

}  // namespace torchscience::autograd::pad

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("pad", &torchscience::autograd::pad::pad);
}
