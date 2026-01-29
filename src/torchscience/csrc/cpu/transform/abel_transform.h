#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/sqrt.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of numerical Abel transform.
 *
 * Computes F(y) = 2 * integral_y^inf f(r) * r / sqrt(r^2 - y^2) dr
 * using numerical quadrature with singularity handling.
 *
 * @param input Input tensor f(r) sampled at points r_in
 * @param y_out Impact parameter values where to evaluate the transform
 * @param r_in Radial points where input is sampled (positive, sorted)
 * @param dim Dimension along which to integrate
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Abel transform evaluated at y_out
 */
inline at::Tensor abel_transform(
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "abel_transform: input tensor must be non-empty");
    TORCH_CHECK(r_in.dim() == 1, "abel_transform: r_in must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "abel_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == r_in.size(0),
        "abel_transform: input size along dim must match r_in length");

    int64_t n_r = r_in.size(0);
    TORCH_CHECK(n_r >= 2, "abel_transform: need at least 2 radial points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor y_c = y_out.contiguous();
    at::Tensor r_c = r_in.contiguous();

    // Flatten y_out for computation
    at::Tensor y_flat = y_c.flatten();
    int64_t n_y = y_flat.size(0);

    // Move input's dim to the last position
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_r]

    // Get batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < input_moved.dim() - 1; i++) {
        batch_shape.push_back(input_moved.size(i));
    }

    // Flatten batch dimensions (handle 1D input specially)
    at::Tensor input_flat;
    if (input_moved.dim() == 1) {
        // 1D input: add batch dimension
        input_flat = input_moved.unsqueeze(0);  // [1, n_r]
    } else {
        input_flat = input_moved.flatten(0, -2);  // [batch, n_r]
    }
    int64_t batch_size = input_flat.size(0);

    // Compute dr weights using tensor operations (dtype-agnostic)
    at::Tensor dr = at::zeros({n_r}, r_c.options());
    at::Tensor r_diff = r_c.slice(0, 1, n_r) - r_c.slice(0, 0, n_r - 1);
    dr[0] = r_diff[0] / 2.0;
    if (n_r > 2) {
        dr.slice(0, 1, n_r - 1) = (r_c.slice(0, 2, n_r) - r_c.slice(0, 0, n_r - 2)) / 2.0;
    }
    dr[n_r - 1] = r_diff[n_r - 2] / 2.0;

    // Allocate output
    at::Tensor result = at::zeros({batch_size, n_y}, input_c.options());

    // Small epsilon to avoid division by zero at singularity
    double eps = 1e-10;

    // Compute Abel transform for each y value
    // F(y) = 2 * sum_i f(r_i) * r_i / sqrt(r_i^2 - y^2) * dr_i  (for r_i > y)
    AT_DISPATCH_FLOATING_TYPES(input_c.scalar_type(), "abel_transform", [&] {
        auto result_accessor = result.accessor<scalar_t, 2>();
        auto input_accessor = input_flat.accessor<scalar_t, 2>();
        auto y_accessor = y_flat.accessor<scalar_t, 1>();
        auto r_accessor = r_c.accessor<scalar_t, 1>();
        auto dr_accessor = dr.accessor<scalar_t, 1>();

        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t j = 0; j < n_y; j++) {
                scalar_t y_val = y_accessor[j];
                scalar_t sum = 0.0;

                for (int64_t i = 0; i < n_r; i++) {
                    scalar_t r_val = r_accessor[i];
                    if (r_val > y_val + eps) {
                        scalar_t denom = std::sqrt(r_val * r_val - y_val * y_val);
                        sum += input_accessor[b][i] * r_val / denom * dr_accessor[i];
                    }
                }
                result_accessor[b][j] = 2.0 * sum;
            }
        }
    });

    // Reshape result
    std::vector<int64_t> final_batch_shape = batch_shape;
    final_batch_shape.push_back(n_y);
    result = result.view(final_batch_shape);

    // Reshape to match y_out's original shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            output_shape.push_back(input_c.size(i));
        } else {
            output_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < y_c.dim(); i++) {
        output_shape.push_back(y_c.size(i));
    }

    result = result.view(output_shape);

    // Move the y dimensions to where dim was
    if (y_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t y_ndim = y_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - y_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - y_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for Abel transform.
 */
inline at::Tensor abel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n_r = r_in.size(0);
    at::Tensor r_c = r_in.contiguous();
    at::Tensor y_c = y_out.contiguous();
    at::Tensor y_flat = y_c.flatten();
    int64_t n_y = y_flat.size(0);

    // Compute dr weights using tensor operations (dtype-agnostic)
    at::Tensor dr = at::zeros({n_r}, r_c.options());
    at::Tensor r_diff = r_c.slice(0, 1, n_r) - r_c.slice(0, 0, n_r - 1);
    dr[0] = r_diff[0] / 2.0;
    if (n_r > 2) {
        dr.slice(0, 1, n_r - 1) = (r_c.slice(0, 2, n_r) - r_c.slice(0, 0, n_r - 2)) / 2.0;
    }
    dr[n_r - 1] = r_diff[n_r - 2] / 2.0;

    // Move grad_output's dim to the last position
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten batch and y dimensions (handle 1D specially)
    int64_t y_ndim = y_c.dim();
    at::Tensor grad_flat;
    if (grad_out_moved.dim() <= y_ndim) {
        // No batch dimensions: just flatten y dims and add batch dim
        grad_flat = grad_out_moved.flatten().unsqueeze(0);
    } else {
        grad_flat = grad_out_moved.flatten(0, -y_ndim - 1).flatten(-y_ndim);
    }

    int64_t batch_size = grad_flat.size(0);

    // Allocate grad_input
    at::Tensor grad_input = at::zeros({batch_size, n_r}, input.options());

    double eps = 1e-10;

    // Compute gradient: d(F)/d(f_i) = 2 * r_i / sqrt(r_i^2 - y^2) * dr_i
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "abel_transform_backward", [&] {
        auto grad_input_accessor = grad_input.accessor<scalar_t, 2>();
        auto grad_flat_accessor = grad_flat.accessor<scalar_t, 2>();
        auto y_accessor = y_flat.accessor<scalar_t, 1>();
        auto r_accessor = r_c.accessor<scalar_t, 1>();
        auto dr_accessor = dr.accessor<scalar_t, 1>();

        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t i = 0; i < n_r; i++) {
                scalar_t r_val = r_accessor[i];
                scalar_t sum = 0.0;

                for (int64_t j = 0; j < n_y; j++) {
                    scalar_t y_val = y_accessor[j];
                    if (r_val > y_val + eps) {
                        scalar_t denom = std::sqrt(r_val * r_val - y_val * y_val);
                        sum += grad_flat_accessor[b][j] * 2.0 * r_val / denom * dr_accessor[i];
                    }
                }
                grad_input_accessor[b][i] = sum;
            }
        }
    });

    // Reshape grad_input back to input shape
    grad_input = grad_input.view(input.sizes());
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for Abel transform.
 */
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

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "abel_transform",
        &torchscience::cpu::transform::abel_transform
    );

    module.impl(
        "abel_transform_backward",
        &torchscience::cpu::transform::abel_transform_backward
    );

    module.impl(
        "abel_transform_backward_backward",
        &torchscience::cpu::transform::abel_transform_backward_backward
    );
}
