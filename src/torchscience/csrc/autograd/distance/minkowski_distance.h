#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::distance {

/**
 * Autograd Function for Minkowski distance.
 */
class MinkowskiDistance
    : public torch::autograd::Function<MinkowskiDistance> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x,
        const at::Tensor& y,
        double p,
        const c10::optional<at::Tensor>& weight
    ) {
        bool has_weight = weight.has_value() && weight->defined();
        if (has_weight) {
            ctx->saved_data["weight"] = weight.value();
            ctx->saved_data["has_weight"] = true;
            ctx->saved_data["w_requires_grad"] = weight->requires_grad() && at::isFloatingType(weight->scalar_type());
        } else {
            ctx->saved_data["has_weight"] = false;
            ctx->saved_data["w_requires_grad"] = false;
        }
        ctx->saved_data["p"] = p;

        bool x_requires_grad = x.requires_grad() && at::isFloatingType(x.scalar_type());
        bool y_requires_grad = y.requires_grad() && at::isFloatingType(y.scalar_type());
        ctx->saved_data["x_requires_grad"] = x_requires_grad;
        ctx->saved_data["y_requires_grad"] = y_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::minkowski_distance", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                double,
                const c10::optional<at::Tensor>&
            )>()
            .call(x, y, p, weight);

        // Save tensors for backward (needed for gradient computation)
        ctx->save_for_backward({x, y, output});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor x = saved[0];
        at::Tensor y = saved[1];
        at::Tensor dist_output = saved[2];

        at::Tensor grad_output = grad_outputs[0];

        double p = ctx->saved_data["p"].toDouble();
        bool has_weight = ctx->saved_data["has_weight"].toBool();
        bool x_requires_grad = ctx->saved_data["x_requires_grad"].toBool();
        bool y_requires_grad = ctx->saved_data["y_requires_grad"].toBool();
        bool w_requires_grad = ctx->saved_data["w_requires_grad"].toBool();

        c10::optional<at::Tensor> weight;
        if (has_weight) {
            weight = ctx->saved_data["weight"].toTensor();
        }

        if (!x_requires_grad && !y_requires_grad && !w_requires_grad) {
            return {
                at::Tensor(),  // grad_x
                at::Tensor(),  // grad_y
                at::Tensor(),  // grad_p
                at::Tensor()   // grad_weight
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_x, grad_y, grad_w] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::minkowski_distance_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                double,
                const c10::optional<at::Tensor>&,
                const at::Tensor&
            )>()
            .call(grad_output, x, y, p, weight, dist_output);

        return {
            x_requires_grad ? grad_x : at::Tensor(),
            y_requires_grad ? grad_y : at::Tensor(),
            at::Tensor(),  // grad_p (not differentiable)
            w_requires_grad ? grad_w : at::Tensor()
        };
    }
};

inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    return MinkowskiDistance::apply(x, y, p, weight);
}

}  // namespace torchscience::autograd::distance

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("minkowski_distance", &torchscience::autograd::distance::minkowski_distance);
}
