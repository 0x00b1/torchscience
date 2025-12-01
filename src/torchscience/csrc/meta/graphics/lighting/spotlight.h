#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::lighting {

inline std::tuple<at::Tensor, at::Tensor> spotlight(
    const at::Tensor& light_pos,
    const at::Tensor& surface_pos,
    const at::Tensor& spot_direction,
    const at::Tensor& intensity,
    const at::Tensor& inner_angle,
    const at::Tensor& outer_angle
) {
    TORCH_CHECK(light_pos.size(-1) == 3, "spotlight: light_pos must have last dimension 3");
    TORCH_CHECK(surface_pos.size(-1) == 3, "spotlight: surface_pos must have last dimension 3");
    TORCH_CHECK(spot_direction.size(-1) == 3, "spotlight: spot_direction must have last dimension 3");

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < light_pos.dim() - 1; ++i) {
        batch_shape.push_back(light_pos.size(i));
    }

    return std::make_tuple(
        at::empty(batch_shape, light_pos.options()),
        at::empty_like(light_pos)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> spotlight_backward(
    const at::Tensor& grad_irradiance,
    const at::Tensor& light_pos,
    const at::Tensor& surface_pos,
    const at::Tensor& spot_direction,
    const at::Tensor& intensity,
    const at::Tensor& inner_angle,
    const at::Tensor& outer_angle
) {
    return std::make_tuple(
        at::empty_like(light_pos),
        at::empty_like(surface_pos),
        at::empty_like(spot_direction),
        at::empty_like(intensity),
        at::empty_like(inner_angle),
        at::empty_like(outer_angle)
    );
}

}  // namespace torchscience::meta::graphics::lighting

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("spotlight", &torchscience::meta::graphics::lighting::spotlight);
    m.impl("spotlight_backward", &torchscience::meta::graphics::lighting::spotlight_backward);
}
