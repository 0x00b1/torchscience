#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autograd::graphics::lighting {

class Spotlight : public torch::autograd::Function<Spotlight> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& light_pos,
        const at::Tensor& surface_pos,
        const at::Tensor& spot_direction,
        const at::Tensor& intensity,
        const at::Tensor& inner_angle,
        const at::Tensor& outer_angle
    ) {
        ctx->saved_data["light_pos_requires_grad"] = light_pos.requires_grad();
        ctx->saved_data["surface_pos_requires_grad"] = surface_pos.requires_grad();
        ctx->saved_data["spot_direction_requires_grad"] = spot_direction.requires_grad();
        ctx->saved_data["intensity_requires_grad"] = intensity.requires_grad();
        ctx->saved_data["inner_angle_requires_grad"] = inner_angle.requires_grad();
        ctx->saved_data["outer_angle_requires_grad"] = outer_angle.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [irradiance, light_dir] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::spotlight", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(light_pos, surface_pos, spot_direction, intensity, inner_angle, outer_angle);

        ctx->save_for_backward({light_pos, surface_pos, spot_direction, intensity, inner_angle, outer_angle});

        return {irradiance, light_dir};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor light_pos = saved[0];
        at::Tensor surface_pos = saved[1];
        at::Tensor spot_direction = saved[2];
        at::Tensor intensity = saved[3];
        at::Tensor inner_angle = saved[4];
        at::Tensor outer_angle = saved[5];

        at::Tensor grad_irradiance = grad_outputs[0];
        // grad_outputs[1] is gradient w.r.t. light_dir, which we ignore for now

        bool light_pos_requires_grad = ctx->saved_data["light_pos_requires_grad"].toBool();
        bool surface_pos_requires_grad = ctx->saved_data["surface_pos_requires_grad"].toBool();
        bool spot_direction_requires_grad = ctx->saved_data["spot_direction_requires_grad"].toBool();
        bool intensity_requires_grad = ctx->saved_data["intensity_requires_grad"].toBool();
        bool inner_angle_requires_grad = ctx->saved_data["inner_angle_requires_grad"].toBool();
        bool outer_angle_requires_grad = ctx->saved_data["outer_angle_requires_grad"].toBool();

        if (!light_pos_requires_grad && !surface_pos_requires_grad &&
            !spot_direction_requires_grad && !intensity_requires_grad &&
            !inner_angle_requires_grad && !outer_angle_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_light_pos, grad_surface_pos, grad_spot_direction,
              grad_intensity, grad_inner_angle, grad_outer_angle] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::spotlight_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&
            )>()
            .call(grad_irradiance, light_pos, surface_pos, spot_direction, intensity, inner_angle, outer_angle);

        return {
            light_pos_requires_grad ? grad_light_pos : at::Tensor(),
            surface_pos_requires_grad ? grad_surface_pos : at::Tensor(),
            spot_direction_requires_grad ? grad_spot_direction : at::Tensor(),
            intensity_requires_grad ? grad_intensity : at::Tensor(),
            inner_angle_requires_grad ? grad_inner_angle : at::Tensor(),
            outer_angle_requires_grad ? grad_outer_angle : at::Tensor()
        };
    }
};

inline std::tuple<at::Tensor, at::Tensor> spotlight(
    const at::Tensor& light_pos,
    const at::Tensor& surface_pos,
    const at::Tensor& spot_direction,
    const at::Tensor& intensity,
    const at::Tensor& inner_angle,
    const at::Tensor& outer_angle
) {
    auto results = Spotlight::apply(light_pos, surface_pos, spot_direction, intensity, inner_angle, outer_angle);
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::graphics::lighting

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("spotlight", &torchscience::autograd::graphics::lighting::spotlight);
}
