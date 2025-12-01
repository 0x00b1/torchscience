#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::texture_mapping {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> cube_mapping(
    const at::Tensor& direction
) {
    TORCH_CHECK(direction.size(-1) == 3, "cube_mapping: direction must have last dimension 3");

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < direction.dim() - 1; ++i) {
        batch_shape.push_back(direction.size(i));
    }

    return std::make_tuple(
        at::empty(batch_shape, direction.options().dtype(at::kLong)),
        at::empty(batch_shape, direction.options()),
        at::empty(batch_shape, direction.options())
    );
}

}  // namespace torchscience::meta::graphics::texture_mapping

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("cube_mapping", &torchscience::meta::graphics::texture_mapping::cube_mapping);
}
