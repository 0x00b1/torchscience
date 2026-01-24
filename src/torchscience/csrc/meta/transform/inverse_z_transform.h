#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Meta implementation of inverse Z-transform for shape inference.
 */
inline at::Tensor inverse_z_transform(
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    // Output shape: replace dimension dim with n_out's shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            for (int64_t j = 0; j < n_out.dim(); j++) {
                output_shape.push_back(n_out.size(j));
            }
        } else {
            output_shape.push_back(input.size(i));
        }
    }

    // Handle case where n_out is 0-dimensional (scalar)
    if (n_out.dim() == 0) {
        output_shape.clear();
        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                output_shape.push_back(input.size(i));
            }
        }
    }

    // Output is real
    auto options = input.options();
    if (input.is_complex()) {
        if (input.scalar_type() == at::kComplexFloat) {
            options = options.dtype(at::kFloat);
        } else {
            options = options.dtype(at::kDouble);
        }
    }

    return at::empty(output_shape, options);
}

inline at::Tensor inverse_z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor>
inverse_z_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    at::Tensor grad_grad_output = at::Tensor();
    if (grad_grad_input.defined()) {
        grad_grad_output = inverse_z_transform(grad_grad_input, n_out, z_in, dim);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "inverse_z_transform",
        &torchscience::meta::transform::inverse_z_transform
    );

    module.impl(
        "inverse_z_transform_backward",
        &torchscience::meta::transform::inverse_z_transform_backward
    );

    module.impl(
        "inverse_z_transform_backward_backward",
        &torchscience::meta::transform::inverse_z_transform_backward_backward
    );
}
