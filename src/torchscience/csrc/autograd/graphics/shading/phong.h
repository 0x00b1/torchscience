#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autograd::graphics::shading {

class Phong : public torch::autograd::Function<Phong> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& normal,
        const at::Tensor& view,
        const at::Tensor& light,
        const at::Tensor& shininess
    ) {
        ctx->saved_data["normal_requires_grad"] = normal.requires_grad();
        ctx->saved_data["view_requires_grad"] = view.requires_grad();
        ctx->saved_data["light_requires_grad"] = light.requires_grad();
        ctx->saved_data["shininess_requires_grad"] = shininess.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::phong", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
            .call(normal, view, light, shininess);

        ctx->save_for_backward({normal, view, light, shininess});

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
        at::Tensor shininess = saved[3];

        at::Tensor grad_output = grad_outputs[0];

        bool normal_requires_grad = ctx->saved_data["normal_requires_grad"].toBool();
        bool view_requires_grad = ctx->saved_data["view_requires_grad"].toBool();
        bool light_requires_grad = ctx->saved_data["light_requires_grad"].toBool();
        bool shininess_requires_grad = ctx->saved_data["shininess_requires_grad"].toBool();

        if (!normal_requires_grad && !view_requires_grad && !light_requires_grad && !shininess_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_normal, grad_view, grad_light, grad_shininess] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::phong_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_output, normal, view, light, shininess);

        return {
            normal_requires_grad ? grad_normal : at::Tensor(),
            view_requires_grad ? grad_view : at::Tensor(),
            light_requires_grad ? grad_light : at::Tensor(),
            shininess_requires_grad ? grad_shininess : at::Tensor()
        };
    }
};

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return Phong::apply(normal, view, light, shininess);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("phong", &torchscience::autograd::graphics::shading::phong);
}
