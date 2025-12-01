#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autograd::graphics::tone_mapping {

class Reinhard : public torch::autograd::Function<Reinhard> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input,
        const std::optional<at::Tensor>& white_point
    ) {
        ctx->saved_data["input_requires_grad"] = input.requires_grad();
        ctx->saved_data["has_white_point"] = white_point.has_value();
        if (white_point.has_value()) {
            ctx->saved_data["white_point_requires_grad"] = white_point.value().requires_grad();
        } else {
            ctx->saved_data["white_point_requires_grad"] = false;
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::reinhard", "")
            .typed<at::Tensor(const at::Tensor&, const std::optional<at::Tensor>&)>()
            .call(input, white_point);

        if (white_point.has_value()) {
            ctx->save_for_backward({input, white_point.value()});
        } else {
            ctx->save_for_backward({input});
        }

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];

        at::Tensor grad_output = grad_outputs[0];

        bool input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();
        bool has_white_point = ctx->saved_data["has_white_point"].toBool();
        bool white_point_requires_grad = ctx->saved_data["white_point_requires_grad"].toBool();

        if (!input_requires_grad && !white_point_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        std::optional<at::Tensor> white_point;
        if (has_white_point) {
            white_point = saved[1];
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_input, grad_white_point] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::reinhard_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&
            )>()
            .call(grad_output, input, white_point);

        return {
            input_requires_grad ? grad_input : at::Tensor(),
            white_point_requires_grad ? grad_white_point : at::Tensor()
        };
    }
};

inline at::Tensor reinhard(
    const at::Tensor& input,
    const std::optional<at::Tensor>& white_point
) {
    return Reinhard::apply(input, white_point);
}

}  // namespace torchscience::autograd::graphics::tone_mapping

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("reinhard", &torchscience::autograd::graphics::tone_mapping::reinhard);
}
