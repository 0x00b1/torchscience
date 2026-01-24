#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of Hankel transform for shape inference.
 */
inline at::Tensor hankel_transform(
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Output shape: replace dimension dim with k_out's shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            for (int64_t j = 0; j < k_out.dim(); j++) {
                output_shape.push_back(k_out.size(j));
            }
        } else {
            output_shape.push_back(input.size(i));
        }
    }

    // Handle case where k_out is 0-dimensional (scalar)
    if (k_out.dim() == 0) {
        output_shape.clear();
        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                output_shape.push_back(input.size(i));
            }
        }
    }

    return at::empty(output_shape, input.options());
}

inline at::Tensor hankel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor>
hankel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_input.defined()) {
        grad_grad_output = hankel_transform(grad_grad_input, k_out, r_in, dim, order, integration_method);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "hankel_transform",
        &torchscience::meta::transform::hankel_transform
    );

    module.impl(
        "hankel_transform_backward",
        &torchscience::meta::transform::hankel_transform_backward
    );

    module.impl(
        "hankel_transform_backward_backward",
        &torchscience::meta::transform::hankel_transform_backward_backward
    );
}
