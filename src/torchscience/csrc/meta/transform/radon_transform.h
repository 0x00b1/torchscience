#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of Radon transform for shape inference.
 */
inline at::Tensor radon_transform(
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    TORCH_CHECK(input.dim() >= 2, "radon_transform: input must be at least 2D");
    TORCH_CHECK(angles.dim() == 1, "radon_transform: angles must be 1D");

    int64_t ndim = input.dim();
    int64_t H = input.size(-2);
    int64_t W = input.size(-1);
    int64_t num_angles = angles.size(0);

    // Number of detector bins
    int64_t num_bins = static_cast<int64_t>(std::ceil(std::sqrt(H * H + W * W)));
    if (num_bins % 2 == 0) {
        num_bins += 1;
    }

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 2; i++) {
        output_shape.push_back(input.size(i));
    }
    output_shape.push_back(num_angles);
    output_shape.push_back(num_bins);

    return at::empty(output_shape, input.options());
}

inline at::Tensor radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor>
radon_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_input.defined()) {
        grad_grad_output = radon_transform(grad_grad_input, angles, circle);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "radon_transform",
        &torchscience::meta::transform::radon_transform
    );

    module.impl(
        "radon_transform_backward",
        &torchscience::meta::transform::radon_transform_backward
    );

    module.impl(
        "radon_transform_backward_backward",
        &torchscience::meta::transform::radon_transform_backward_backward
    );
}
