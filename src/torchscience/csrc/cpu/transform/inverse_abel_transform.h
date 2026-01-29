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
 * CPU implementation of numerical inverse Abel transform.
 *
 * Computes f(r) = -(1/pi) * integral_r^inf (dF/dy) / sqrt(y^2 - r^2) dy
 * using numerical differentiation and quadrature with singularity handling.
 *
 * @param input Input tensor F(y) sampled at points y_in
 * @param r_out Radial points where to evaluate the inverse transform
 * @param y_in Impact parameter points where input is sampled (non-negative, sorted)
 * @param dim Dimension along which to integrate
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Inverse Abel transform evaluated at r_out
 */
inline at::Tensor inverse_abel_transform(
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& y_in,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "inverse_abel_transform: input tensor must be non-empty");
    TORCH_CHECK(y_in.dim() == 1, "inverse_abel_transform: y_in must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "inverse_abel_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == y_in.size(0),
        "inverse_abel_transform: input size along dim must match y_in length");

    int64_t n_y = y_in.size(0);
    TORCH_CHECK(n_y >= 2, "inverse_abel_transform: need at least 2 y points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor r_c = r_out.contiguous();
    at::Tensor y_c = y_in.contiguous();

    // Flatten r_out for computation
    at::Tensor r_flat = r_c.flatten();
    int64_t n_r = r_flat.size(0);

    // Move input's dim to the last position
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_y]

    // Get batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < input_moved.dim() - 1; i++) {
        batch_shape.push_back(input_moved.size(i));
    }

    // Flatten batch dimensions (handle 1D input specially)
    at::Tensor input_flat;
    if (input_moved.dim() == 1) {
        // 1D input: add batch dimension
        input_flat = input_moved.unsqueeze(0);  // [1, n_y]
    } else {
        input_flat = input_moved.flatten(0, -2);  // [batch, n_y]
    }
    int64_t batch_size = input_flat.size(0);

    // Compute dy for numerical differentiation using tensor operations (dtype-agnostic)
    at::Tensor dy = at::zeros({n_y}, y_c.options());
    at::Tensor y_diff = y_c.slice(0, 1, n_y) - y_c.slice(0, 0, n_y - 1);
    dy[0] = y_diff[0];
    if (n_y > 2) {
        dy.slice(0, 1, n_y - 1) = (y_c.slice(0, 2, n_y) - y_c.slice(0, 0, n_y - 2)) / 2.0;
    }
    dy[n_y - 1] = y_diff[n_y - 2];

    // Allocate output
    at::Tensor result = at::zeros({batch_size, n_r}, input_c.options());

    // Small epsilon to avoid division by zero at singularity
    double eps = 1e-10;

    // Compute inverse Abel transform for each r value
    // f(r) = -(1/pi) * sum_j (dF/dy)_j / sqrt(y_j^2 - r^2) * dy_j  (for y_j > r)
    AT_DISPATCH_FLOATING_TYPES(input_c.scalar_type(), "inverse_abel_transform", [&] {
        auto result_accessor = result.accessor<scalar_t, 2>();
        auto input_accessor = input_flat.accessor<scalar_t, 2>();
        auto r_accessor = r_flat.accessor<scalar_t, 1>();
        auto y_accessor = y_c.accessor<scalar_t, 1>();
        auto dy_accessor = dy.accessor<scalar_t, 1>();

        for (int64_t b = 0; b < batch_size; b++) {
            // Compute numerical derivative dF/dy using central differences
            std::vector<scalar_t> dF_dy(n_y);

            // Forward difference at start
            dF_dy[0] = (input_accessor[b][1] - input_accessor[b][0]) / dy_accessor[0];

            // Central differences in the middle
            for (int64_t j = 1; j < n_y - 1; j++) {
                dF_dy[j] = (input_accessor[b][j + 1] - input_accessor[b][j - 1]) /
                           (y_accessor[j + 1] - y_accessor[j - 1]);
            }

            // Backward difference at end
            dF_dy[n_y - 1] = (input_accessor[b][n_y - 1] - input_accessor[b][n_y - 2]) /
                             dy_accessor[n_y - 1];

            for (int64_t i = 0; i < n_r; i++) {
                scalar_t r_val = r_accessor[i];
                scalar_t sum = 0.0;

                for (int64_t j = 0; j < n_y; j++) {
                    scalar_t y_val = y_accessor[j];
                    if (y_val > r_val + eps) {
                        scalar_t denom = std::sqrt(y_val * y_val - r_val * r_val);
                        sum += dF_dy[j] / denom * dy_accessor[j];
                    }
                }
                result_accessor[b][i] = -sum / M_PI;
            }
        }
    });

    // Reshape result
    std::vector<int64_t> final_batch_shape = batch_shape;
    final_batch_shape.push_back(n_r);
    result = result.view(final_batch_shape);

    // Reshape to match r_out's original shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            output_shape.push_back(input_c.size(i));
        } else {
            output_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < r_c.dim(); i++) {
        output_shape.push_back(r_c.size(i));
    }

    result = result.view(output_shape);

    // Move the r dimensions to where dim was
    if (r_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t r_ndim = r_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - r_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - r_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for inverse Abel transform.
 *
 * The forward pass computes:
 *   output[i] = -(1/pi) * sum_j (dF/dy)[j] * K[i,j] * dy[j]
 * where K[i,j] = 1/sqrt(y[j]^2 - r[i]^2) for y[j] > r[i], else 0
 * and dF/dy is computed via finite differences.
 *
 * The backward computes:
 *   grad_F[k] = sum_j D^T[k,j] * (-(1/pi) * sum_i grad_output[i] * K[i,j] * dy[j])
 * where D^T is the adjoint of the differentiation operator.
 */
inline at::Tensor inverse_abel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& y_in,
    int64_t dim,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n_y = y_in.size(0);
    at::Tensor y_c = y_in.contiguous();
    at::Tensor r_c = r_out.contiguous();
    at::Tensor r_flat = r_c.flatten();
    int64_t n_r = r_flat.size(0);

    // Compute dy weights (same as forward)
    at::Tensor dy = at::zeros({n_y}, y_c.options());

    double eps = 1e-10;

    // Move grad_output's r dimension to the last position
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten batch dimensions
    int64_t r_ndim = r_c.dim();
    at::Tensor grad_flat;
    if (grad_out_moved.dim() > r_ndim) {
        grad_flat = grad_out_moved.flatten(0, -r_ndim - 1).flatten(-r_ndim);
    } else {
        grad_flat = grad_out_moved.flatten();
        grad_flat = grad_flat.unsqueeze(0);
    }

    int64_t batch_size = grad_flat.size(0);

    // Allocate grad_input
    at::Tensor grad_input = at::zeros({batch_size, n_y}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inverse_abel_transform_backward", [&] {
        auto dy_accessor = dy.accessor<scalar_t, 1>();
        auto y_accessor = y_c.accessor<scalar_t, 1>();

        // Compute dy
        dy_accessor[0] = y_accessor[1] - y_accessor[0];
        for (int64_t j = 1; j < n_y - 1; j++) {
            dy_accessor[j] = (y_accessor[j + 1] - y_accessor[j - 1]) / 2.0;
        }
        dy_accessor[n_y - 1] = y_accessor[n_y - 1] - y_accessor[n_y - 2];

        auto grad_input_accessor = grad_input.accessor<scalar_t, 2>();
        auto grad_flat_accessor = grad_flat.accessor<scalar_t, 2>();
        auto r_accessor = r_flat.accessor<scalar_t, 1>();

        for (int64_t b = 0; b < batch_size; b++) {
            // Step 1: Compute intermediate = -(1/pi) * sum_i grad_output[i] * K[i,j] for each j
            std::vector<scalar_t> intermediate(n_y, 0.0);

            for (int64_t j = 0; j < n_y; j++) {
                scalar_t y_val = y_accessor[j];
                scalar_t sum = 0.0;

                for (int64_t i = 0; i < n_r; i++) {
                    scalar_t r_val = r_accessor[i];
                    if (y_val > r_val + eps) {
                        scalar_t denom = std::sqrt(y_val * y_val - r_val * r_val);
                        sum += grad_flat_accessor[b][i] / denom;
                    }
                }
                intermediate[j] = -sum / M_PI * dy_accessor[j];
            }

            // Step 2: Apply adjoint of differentiation operator D^T
            // D^T is the transpose of the finite difference matrix
            // For forward diff at 0: D[0,0] = -1/dy[0], D[0,1] = 1/dy[0]
            //   -> D^T[0,0] = -1/dy[0], D^T[1,0] = 1/dy[0]
            // For central diff at j: D[j,j-1] = -1/(y[j+1]-y[j-1]), D[j,j+1] = 1/(y[j+1]-y[j-1])
            //   -> D^T[j-1,j] = -1/(y[j+1]-y[j-1]), D^T[j+1,j] = 1/(y[j+1]-y[j-1])
            // For backward diff at n-1: D[n-1,n-2] = -1/dy[n-1], D[n-1,n-1] = 1/dy[n-1]
            //   -> D^T[n-2,n-1] = -1/dy[n-1], D^T[n-1,n-1] = 1/dy[n-1]

            // grad_F[k] = sum_j D^T[k,j] * intermediate[j]

            // Contribution from j=0 (forward difference)
            scalar_t coeff_0 = 1.0 / dy_accessor[0];
            grad_input_accessor[b][0] += -coeff_0 * intermediate[0];
            grad_input_accessor[b][1] += coeff_0 * intermediate[0];

            // Contribution from j=1 to n_y-2 (central differences)
            for (int64_t j = 1; j < n_y - 1; j++) {
                scalar_t coeff_j = 1.0 / (y_accessor[j + 1] - y_accessor[j - 1]);
                grad_input_accessor[b][j - 1] += -coeff_j * intermediate[j];
                grad_input_accessor[b][j + 1] += coeff_j * intermediate[j];
            }

            // Contribution from j=n_y-1 (backward difference)
            scalar_t coeff_last = 1.0 / dy_accessor[n_y - 1];
            grad_input_accessor[b][n_y - 2] += -coeff_last * intermediate[n_y - 1];
            grad_input_accessor[b][n_y - 1] += coeff_last * intermediate[n_y - 1];
        }
    });

    // Reshape grad_input back to input shape
    std::vector<int64_t> input_shape;
    for (int64_t i = 0; i < input.dim(); i++) {
        input_shape.push_back(input.size(i));
    }

    // Reshape: [batch, n_y] -> original shape with n_y at dim
    at::Tensor input_c = input.contiguous();
    at::Tensor input_moved = input_c.movedim(dim, -1);

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < input_moved.dim() - 1; i++) {
        batch_shape.push_back(input_moved.size(i));
    }
    batch_shape.push_back(n_y);

    grad_input = grad_input.view(batch_shape);
    grad_input = grad_input.movedim(-1, dim);

    return grad_input.contiguous();
}

/**
 * Double backward pass for inverse Abel transform.
 *
 * For a linear transform, the second derivative w.r.t. input is zero.
 * The gradient w.r.t. grad_output is the forward transform applied to grad_grad_input.
 */
inline std::tuple<at::Tensor, at::Tensor>
inverse_abel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& y_in,
    int64_t dim,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        // The backward of backward w.r.t. grad_output is applying the forward transform
        // to grad_grad_input (since the transform is linear)
        grad_grad_output = inverse_abel_transform(
            grad_grad_input, r_out, y_in, dim, integration_method
        );
    }

    // Second derivative w.r.t. input is zero for linear transforms
    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_abel_transform",
        &torchscience::cpu::transform::inverse_abel_transform
    );

    module.impl(
        "inverse_abel_transform_backward",
        &torchscience::cpu::transform::inverse_abel_transform_backward
    );

    module.impl(
        "inverse_abel_transform_backward_backward",
        &torchscience::cpu::transform::inverse_abel_transform_backward_backward
    );
}
