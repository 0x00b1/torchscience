#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::morphology {

/**
 * Autograd function for dilation.
 */
class DilationFunction : public torch::autograd::Function<DilationFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const at::Tensor& structuring_element,
        c10::optional<at::IntArrayRef> origin,
        int64_t padding_mode
    ) {
        ctx->save_for_backward({input, structuring_element});

        // Save origin as a tensor if provided
        if (origin.has_value()) {
            auto origin_tensor = at::tensor(
                std::vector<int64_t>(origin->begin(), origin->end()),
                at::TensorOptions().dtype(at::kLong)
            );
            ctx->save_for_backward({input, structuring_element, origin_tensor});
        } else {
            ctx->save_for_backward({input, structuring_element});
        }
        ctx->saved_data["has_origin"] = origin.has_value();
        ctx->saved_data["padding_mode"] = padding_mode;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::dilation", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, c10::optional<at::IntArrayRef>, int64_t)>()
            .call(input, structuring_element, origin, padding_mode);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];
        at::Tensor structuring_element = saved[1];

        bool has_origin = ctx->saved_data["has_origin"].toBool();
        int64_t padding_mode = ctx->saved_data["padding_mode"].toInt();

        c10::optional<at::IntArrayRef> origin = c10::nullopt;
        std::vector<int64_t> origin_vec;
        if (has_origin) {
            at::Tensor origin_tensor = saved[2];
            origin_vec.resize(origin_tensor.numel());
            auto origin_accessor = origin_tensor.accessor<int64_t, 1>();
            for (int64_t i = 0; i < origin_tensor.numel(); ++i) {
                origin_vec[i] = origin_accessor[i];
            }
            origin = at::IntArrayRef(origin_vec);
        }

        at::Tensor grad_output = grad_outputs[0];

        if (!grad_output.defined()) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::dilation_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::optional<at::IntArrayRef>, int64_t)>()
            .call(grad_output, input, structuring_element, origin, padding_mode);

        // No gradient for structuring_element, origin, or padding_mode
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

inline at::Tensor dilation(
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    return DilationFunction::apply(input, structuring_element, origin, padding_mode);
}

}  // namespace torchscience::autograd::morphology

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("dilation", &torchscience::autograd::morphology::dilation);
}
