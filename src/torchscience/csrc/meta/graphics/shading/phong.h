#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

inline at::Tensor phong(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    TORCH_CHECK(normal.size(-1) == 3, "phong: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "phong: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "phong: light must have last dimension 3");

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < normal.dim() - 1; ++i) {
        output_shape.push_back(normal.size(i));
    }

    return at::empty(output_shape, normal.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> phong_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& shininess
) {
    return std::make_tuple(
        at::empty_like(normal),
        at::empty_like(view),
        at::empty_like(light),
        at::empty_like(shininess)
    );
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("phong", &torchscience::meta::graphics::shading::phong);
    m.impl("phong_backward", &torchscience::meta::graphics::shading::phong_backward);
}
