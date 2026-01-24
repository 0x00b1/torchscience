#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of Z-transform for shape inference.
 */
inline at::Tensor z_transform(
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Output shape: replace dimension dim with z_out's shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            for (int64_t j = 0; j < z_out.dim(); j++) {
                output_shape.push_back(z_out.size(j));
            }
        } else {
            output_shape.push_back(input.size(i));
        }
    }

    // Handle case where z_out is 0-dimensional (scalar)
    if (z_out.dim() == 0) {
        output_shape.clear();
        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                output_shape.push_back(input.size(i));
            }
        }
    }

    // Output dtype matches z_out if complex, else promote
    auto options = z_out.is_complex() ? z_out.options() : input.options();

    return at::empty(output_shape, options);
}

inline at::Tensor z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor>
z_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_input.defined()) {
        grad_grad_output = z_transform(grad_grad_input, z_out, dim);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "z_transform",
        &torchscience::meta::transform::z_transform
    );

    module.impl(
        "z_transform_backward",
        &torchscience::meta::transform::z_transform_backward
    );

    module.impl(
        "z_transform_backward_backward",
        &torchscience::meta::transform::z_transform_backward_backward
    );
}
