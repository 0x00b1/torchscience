#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::projection {

inline at::Tensor perspective_projection(
    const at::Tensor& fov,
    const at::Tensor& aspect,
    const at::Tensor& near,
    const at::Tensor& far
) {
    // Determine batch shape (broadcast all inputs)
    auto batch_sizes = at::infer_size(fov.sizes(), aspect.sizes());
    batch_sizes = at::infer_size(batch_sizes, near.sizes());
    batch_sizes = at::infer_size(batch_sizes, far.sizes());

    // Output shape: batch_shape + (4, 4)
    std::vector<int64_t> output_shape(batch_sizes.begin(), batch_sizes.end());
    output_shape.push_back(4);
    output_shape.push_back(4);

    return at::empty(output_shape, fov.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
perspective_projection_backward(
    const at::Tensor& grad_output,
    const at::Tensor& fov,
    const at::Tensor& aspect,
    const at::Tensor& near,
    const at::Tensor& far
) {
    // Batch shape is grad_output shape minus last two dims
    auto grad_sizes = grad_output.sizes();
    std::vector<int64_t> batch_shape(grad_sizes.begin(), grad_sizes.end() - 2);

    auto options = grad_output.options();
    return std::make_tuple(
        at::empty(batch_shape, options),
        at::empty(batch_shape, options),
        at::empty(batch_shape, options),
        at::empty(batch_shape, options)
    );
}

}  // namespace torchscience::meta::graphics::projection

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("perspective_projection", &torchscience::meta::graphics::projection::perspective_projection);
    m.impl("perspective_projection_backward", &torchscience::meta::graphics::projection::perspective_projection_backward);
}
