#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of inverse Radon transform for shape inference.
 */
inline at::Tensor inverse_radon_transform(
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    TORCH_CHECK(sinogram.dim() >= 2, "inverse_radon_transform: sinogram must be at least 2D");
    TORCH_CHECK(angles.dim() == 1, "inverse_radon_transform: angles must be 1D");

    int64_t ndim = sinogram.dim();
    int64_t num_angles = sinogram.size(-2);
    int64_t num_bins = sinogram.size(-1);

    TORCH_CHECK(
        angles.size(0) == num_angles,
        "inverse_radon_transform: angles size (", angles.size(0),
        ") must match sinogram angles dimension (", num_angles, ")"
    );

    // Determine output image size
    int64_t img_size = output_size;
    if (img_size <= 0) {
        img_size = static_cast<int64_t>(std::floor(num_bins / std::sqrt(2.0)));
        if (img_size < 1) img_size = 1;
    }
    int64_t H = img_size;
    int64_t W = img_size;

    // Build output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 2; i++) {
        output_shape.push_back(sinogram.size(i));
    }
    output_shape.push_back(H);
    output_shape.push_back(W);

    return at::empty(output_shape, sinogram.options());
}

inline at::Tensor inverse_radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    return at::empty_like(sinogram);
}

inline std::tuple<at::Tensor, at::Tensor>
inverse_radon_transform_backward_backward(
    const at::Tensor& grad_grad_sinogram,
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_sinogram.defined()) {
        grad_grad_output = inverse_radon_transform(
            grad_grad_sinogram, angles, circle, output_size, filter_type
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "inverse_radon_transform",
        &torchscience::meta::transform::inverse_radon_transform
    );

    module.impl(
        "inverse_radon_transform_backward",
        &torchscience::meta::transform::inverse_radon_transform_backward
    );

    module.impl(
        "inverse_radon_transform_backward_backward",
        &torchscience::meta::transform::inverse_radon_transform_backward_backward
    );
}
