#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

namespace {

/**
 * Reduce gradient to target size by summing over broadcasted dimensions.
 * This handles the case where inputs were broadcasted during forward pass
 * and gradients need to be reduced back to original shapes.
 */
inline at::Tensor reduce_grad_to_size(const at::Tensor& grad, c10::IntArrayRef target_size) {
    if (grad.sizes() == target_size) {
        return grad;
    }

    // Handle empty target (scalar)
    if (target_size.empty()) {
        return grad.sum();
    }

    // Compute which dimensions to sum over
    std::vector<int64_t> dims_to_reduce;
    int64_t grad_dim = grad.dim();
    int64_t target_dim = static_cast<int64_t>(target_size.size());

    // Sum over leading dimensions that don't exist in target
    for (int64_t i = 0; i < grad_dim - target_dim; ++i) {
        dims_to_reduce.push_back(i);
    }

    // Sum over dimensions that were size 1 in target but expanded in grad
    for (int64_t i = 0; i < target_dim; ++i) {
        int64_t grad_idx = grad_dim - target_dim + i;
        if (target_size[i] == 1 && grad.size(grad_idx) != 1) {
            dims_to_reduce.push_back(grad_idx);
        }
    }

    if (dims_to_reduce.empty()) {
        // Just reshape if dimensions are compatible
        return grad.view(target_size);
    }

    // Sum over the dimensions, keeping dims to preserve structure
    at::Tensor reduced = grad.sum(dims_to_reduce, /*keepdim=*/true);

    // Reshape to target size
    return reduced.view(target_size);
}

}  // anonymous namespace

/**
 * Nested Backward class for second-order gradients.
 */
class CookTorranceBackward
    : public torch::autograd::Function<CookTorranceBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_output,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& roughness,
        const at::Tensor& f0,
        bool normal_requires_grad,
        bool view_requires_grad,
        bool light_requires_grad,
        bool roughness_requires_grad,
        bool f0_requires_grad
    ) {
        ctx->save_for_backward({grad_output, normal, view, light, roughness, f0});
        ctx->saved_data["normal_requires_grad"] = normal_requires_grad;
        ctx->saved_data["view_requires_grad"] = view_requires_grad;
        ctx->saved_data["light_requires_grad"] = light_requires_grad;
        ctx->saved_data["roughness_requires_grad"] = roughness_requires_grad;
        ctx->saved_data["f0_requires_grad"] = f0_requires_grad;

        // Save original sizes for gradient reduction in backward_backward pass
        ctx->saved_data["grad_output_sizes"] = grad_output.sizes().vec();
        ctx->saved_data["normal_sizes"] = normal.sizes().vec();
        ctx->saved_data["view_sizes"] = view.sizes().vec();
        ctx->saved_data["light_sizes"] = light.sizes().vec();
        ctx->saved_data["roughness_sizes"] = roughness.sizes().vec();
        ctx->saved_data["f0_sizes"] = f0.sizes().vec();

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_view, grad_light, grad_roughness, grad_f0] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::cook_torrance_backward", "")
                .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&
                )>()
                .call(grad_output, normal, view, light, roughness, f0);

        return {grad_normal, grad_view, grad_light, grad_roughness, grad_f0};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor grad_output = saved[0];
        at::Tensor normal = saved[1];
        at::Tensor view = saved[2];
        at::Tensor light = saved[3];
        at::Tensor roughness = saved[4];
        at::Tensor f0 = saved[5];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool roughness_requires_grad = ctx->saved_data["roughness_requires_grad"].toBool();
        bool f0_requires_grad = ctx->saved_data["f0_requires_grad"].toBool();

        // Retrieve original sizes for gradient reduction
        auto grad_output_sizes = ctx->saved_data["grad_output_sizes"].toIntVector();
        auto normal_sizes = ctx->saved_data["normal_sizes"].toIntVector();
        auto view_sizes = ctx->saved_data["view_sizes"].toIntVector();
        auto light_sizes = ctx->saved_data["light_sizes"].toIntVector();
        auto roughness_sizes = ctx->saved_data["roughness_sizes"].toIntVector();
        auto f0_sizes = ctx->saved_data["f0_sizes"].toIntVector();

        // grad_outputs[0] = gg_normal, [1] = gg_view, [2] = gg_light, [3] = gg_roughness, [4] = gg_f0
        at::Tensor gg_normal = grad_outputs[0].defined() ? grad_outputs[0] : at::zeros_like(normal);
        at::Tensor gg_view = grad_outputs[1].defined() ? grad_outputs[1] : at::zeros_like(view);
        at::Tensor gg_light = grad_outputs[2].defined() ? grad_outputs[2] : at::zeros_like(light);
        at::Tensor gg_roughness = grad_outputs[3].defined() ? grad_outputs[3] : at::zeros_like(roughness);
        at::Tensor gg_f0 = grad_outputs[4].defined() ? grad_outputs[4] : at::zeros_like(f0);

        bool any_requires_grad = normal_requires_grad || view_requires_grad ||
                                  light_requires_grad || roughness_requires_grad || f0_requires_grad;

        if (!any_requires_grad) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad2_normal
                at::Tensor(),  // grad2_view
                at::Tensor(),  // grad2_light
                at::Tensor(),  // grad2_roughness
                at::Tensor(),  // grad2_f0
                at::Tensor(),  // normal_requires_grad (not differentiable)
                at::Tensor(),  // view_requires_grad
                at::Tensor(),  // light_requires_grad
                at::Tensor(),  // roughness_requires_grad
                at::Tensor()   // f0_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_grad_output, grad2_normal, grad2_view, grad2_light, grad2_roughness, grad2_f0] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::cook_torrance_backward_backward", "")
                .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&
                )>()
                .call(gg_normal, gg_view, gg_light, gg_roughness, gg_f0,
                      grad_output, normal, view, light, roughness, f0);

        // Reduce gradients to original input sizes (handles broadcasting)
        return {
            reduce_grad_to_size(grad_grad_output, grad_output_sizes),
            normal_requires_grad ? reduce_grad_to_size(grad2_normal, normal_sizes) : at::Tensor(),
            view_requires_grad ? reduce_grad_to_size(grad2_view, view_sizes) : at::Tensor(),
            light_requires_grad ? reduce_grad_to_size(grad2_light, light_sizes) : at::Tensor(),
            roughness_requires_grad ? reduce_grad_to_size(grad2_roughness, roughness_sizes) : at::Tensor(),
            f0_requires_grad ? reduce_grad_to_size(grad2_f0, f0_sizes) : at::Tensor(),
            at::Tensor(),  // normal_requires_grad (not differentiable)
            at::Tensor(),  // view_requires_grad
            at::Tensor(),  // light_requires_grad
            at::Tensor(),  // roughness_requires_grad
            at::Tensor()   // f0_requires_grad
        };
    }
};

/**
 * Autograd Function for Cook-Torrance BRDF.
 */
class CookTorrance
    : public torch::autograd::Function<CookTorrance> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& roughness,
        const at::Tensor& f0
    ) {
        bool normal_requires_grad = normal.requires_grad() && at::isFloatingType(normal.scalar_type());
        bool view_requires_grad = view.requires_grad() && at::isFloatingType(view.scalar_type());
        bool light_requires_grad = light.requires_grad() && at::isFloatingType(light.scalar_type());
        bool roughness_requires_grad = roughness.requires_grad() && at::isFloatingType(roughness.scalar_type());
        bool f0_requires_grad = f0.requires_grad() && at::isFloatingType(f0.scalar_type());

        ctx->saved_data["normal_requires_grad"] = normal_requires_grad;
        ctx->saved_data["view_requires_grad"] = view_requires_grad;
        ctx->saved_data["light_requires_grad"] = light_requires_grad;
        ctx->saved_data["roughness_requires_grad"] = roughness_requires_grad;
        ctx->saved_data["f0_requires_grad"] = f0_requires_grad;

        // Save original sizes for gradient reduction in backward pass
        ctx->saved_data["normal_sizes"] = normal.sizes().vec();
        ctx->saved_data["view_sizes"] = view.sizes().vec();
        ctx->saved_data["light_sizes"] = light.sizes().vec();
        ctx->saved_data["roughness_sizes"] = roughness.sizes().vec();
        ctx->saved_data["f0_sizes"] = f0.sizes().vec();

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::cook_torrance", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(normal, view, light, roughness, f0);

        // Save tensors for backward
        ctx->save_for_backward({normal, view, light, roughness, f0});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor normal = saved[0];
        at::Tensor view = saved[1];
        at::Tensor light = saved[2];
        at::Tensor roughness = saved[3];
        at::Tensor f0 = saved[4];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool roughness_requires_grad = ctx->saved_data["roughness_requires_grad"].toBool();
        bool f0_requires_grad = ctx->saved_data["f0_requires_grad"].toBool();

        // Retrieve original sizes for gradient reduction
        auto normal_sizes = ctx->saved_data["normal_sizes"].toIntVector();
        auto view_sizes = ctx->saved_data["view_sizes"].toIntVector();
        auto light_sizes = ctx->saved_data["light_sizes"].toIntVector();
        auto roughness_sizes = ctx->saved_data["roughness_sizes"].toIntVector();
        auto f0_sizes = ctx->saved_data["f0_sizes"].toIntVector();

        if (!normal_requires_grad && !view_requires_grad && !light_requires_grad &&
            !roughness_requires_grad && !f0_requires_grad) {
            return {
                at::Tensor(),  // grad_normal
                at::Tensor(),  // grad_view
                at::Tensor(),  // grad_light
                at::Tensor(),  // grad_roughness
                at::Tensor()   // grad_f0
            };
        }

        // Use nested backward function for double backward support
        std::vector<at::Tensor> grads = CookTorranceBackward::apply(
            grad_output, normal, view, light, roughness, f0,
            normal_requires_grad, view_requires_grad, light_requires_grad,
            roughness_requires_grad, f0_requires_grad
        );

        // Reduce gradients to original input sizes (handles broadcasting)
        return {
            normal_requires_grad ? reduce_grad_to_size(grads[0], normal_sizes) : at::Tensor(),
            view_requires_grad ? reduce_grad_to_size(grads[1], view_sizes) : at::Tensor(),
            light_requires_grad ? reduce_grad_to_size(grads[2], light_sizes) : at::Tensor(),
            roughness_requires_grad ? reduce_grad_to_size(grads[3], roughness_sizes) : at::Tensor(),
            f0_requires_grad ? reduce_grad_to_size(grads[4], f0_sizes) : at::Tensor()
        };
    }
};

inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    return CookTorrance::apply(normal, view, light, roughness, f0);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("cook_torrance", &torchscience::autograd::graphics::shading::cook_torrance);
}
