#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autograd::graphics::projection {

class PerspectiveProjection : public torch::autograd::Function<PerspectiveProjection> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& fov,
        const at::Tensor& aspect,
        const at::Tensor& near,
        const at::Tensor& far
    ) {
        ctx->saved_data["fov_requires_grad"] = fov.requires_grad();
        ctx->saved_data["aspect_requires_grad"] = aspect.requires_grad();
        ctx->saved_data["near_requires_grad"] = near.requires_grad();
        ctx->saved_data["far_requires_grad"] = far.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::perspective_projection", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(fov, aspect, near, far);

        ctx->save_for_backward({fov, aspect, near, far});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor fov = saved[0];
        at::Tensor aspect = saved[1];
        at::Tensor near = saved[2];
        at::Tensor far = saved[3];

        at::Tensor grad_output = grad_outputs[0];

        bool fov_requires_grad = ctx->saved_data["fov_requires_grad"].toBool();
        bool aspect_requires_grad = ctx->saved_data["aspect_requires_grad"].toBool();
        bool near_requires_grad = ctx->saved_data["near_requires_grad"].toBool();
        bool far_requires_grad = ctx->saved_data["far_requires_grad"].toBool();

        if (!fov_requires_grad && !aspect_requires_grad &&
            !near_requires_grad && !far_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_fov, grad_aspect, grad_near, grad_far] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::perspective_projection_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, fov, aspect, near, far);

        return {
            fov_requires_grad ? grad_fov : at::Tensor(),
            aspect_requires_grad ? grad_aspect : at::Tensor(),
            near_requires_grad ? grad_near : at::Tensor(),
            far_requires_grad ? grad_far : at::Tensor()
        };
    }
};

inline at::Tensor perspective_projection(
    const at::Tensor& fov,
    const at::Tensor& aspect,
    const at::Tensor& near,
    const at::Tensor& far
) {
    return PerspectiveProjection::apply(fov, aspect, near, far);
}

}  // namespace torchscience::autograd::graphics::projection

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("perspective_projection", &torchscience::autograd::graphics::projection::perspective_projection);
}
