#pragma once

#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::pad {

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

        at::Tensor grad_input;
        {
            at::AutoDispatchBelowAutograd guard;
            grad_input = c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::pad_backward", "")
                .typed<at::Tensor(
                    const at::Tensor&,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    std::string,
                    c10::optional<std::vector<int64_t>>,
                    int64_t
                )>()
                .call(grad_outputs[0], input_shape, padding, mode, dim, order);
        }

        // Return gradients for: input, padding, mode, value, dim, order, out
        // Only input has gradient
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
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
