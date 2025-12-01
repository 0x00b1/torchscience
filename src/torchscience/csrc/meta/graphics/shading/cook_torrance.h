// src/torchscience/csrc/meta/graphics/shading/cook_torrance.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

/**
 * Meta implementation for Cook-Torrance BRDF shape inference.
 */
inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    TORCH_CHECK(normal.size(-1) == 3, "cook_torrance: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "cook_torrance: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "cook_torrance: light must have last dimension 3");

    // Check if f0 is RGB (last dim = 3)
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions (excluding the last dim of 3 for vectors)
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Compute broadcast batch shape
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    if (f0_is_rgb) {
        batch_shape.push_back(3);
    }

    return at::empty(batch_shape, normal.options());
}

/**
 * Meta implementation for Cook-Torrance backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    return std::make_tuple(
        at::empty_like(normal),
        at::empty_like(view),
        at::empty_like(light),
        at::empty_like(roughness),
        at::empty_like(f0)
    );
}

/**
 * Meta implementation for Cook-Torrance backward_backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward_backward(
    const at::Tensor& gg_normal,
    const at::Tensor& gg_view,
    const at::Tensor& gg_light,
    const at::Tensor& gg_roughness,
    const at::Tensor& gg_f0,
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    return std::make_tuple(
        at::empty_like(grad_output),
        at::empty_like(normal),
        at::empty_like(view),
        at::empty_like(light),
        at::empty_like(roughness),
        at::empty_like(f0)
    );
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("cook_torrance", &torchscience::meta::graphics::shading::cook_torrance);
    m.impl("cook_torrance_backward", &torchscience::meta::graphics::shading::cook_torrance_backward);
    m.impl("cook_torrance_backward_backward", &torchscience::meta::graphics::shading::cook_torrance_backward_backward);
}
