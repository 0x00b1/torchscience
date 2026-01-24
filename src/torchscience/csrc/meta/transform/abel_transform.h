#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of Abel transform for shape inference.
 */
inline at::Tensor abel_transform(
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Output shape: replace dimension dim with y_out's shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            for (int64_t j = 0; j < y_out.dim(); j++) {
                output_shape.push_back(y_out.size(j));
            }
        } else {
            output_shape.push_back(input.size(i));
        }
    }

    // Handle case where y_out is 0-dimensional (scalar)
    if (y_out.dim() == 0) {
        output_shape.clear();
        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                output_shape.push_back(input.size(i));
            }
        }
    }

    return at::empty(output_shape, input.options());
}

inline at::Tensor abel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor>
abel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_input.defined()) {
        grad_grad_output = abel_transform(
            grad_grad_input, y_out, r_in, dim, integration_method
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "abel_transform",
        &torchscience::meta::transform::abel_transform
    );

    module.impl(
        "abel_transform_backward",
        &torchscience::meta::transform::abel_transform_backward
    );

    module.impl(
        "abel_transform_backward_backward",
        &torchscience::meta::transform::abel_transform_backward_backward
    );
}
