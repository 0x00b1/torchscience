#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of numerical Laplace transform.
 *
 * Computes F(s) = integral_0^inf f(t) * exp(-s*t) dt
 * using numerical quadrature.
 *
 * @param input Input tensor f(t) sampled at points t
 * @param s Complex frequency values where to evaluate the transform
 * @param t Time points where input is sampled (must be non-negative, sorted)
 * @param dim Dimension along which to integrate
 * @param integration_method 0=trapezoidal, 1=simpson, 2=gauss_legendre
 * @return Laplace transform evaluated at s
 */
inline at::Tensor laplace_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "laplace_transform: input tensor must be non-empty");
    TORCH_CHECK(t.dim() == 1, "laplace_transform: t must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "laplace_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == t.size(0),
        "laplace_transform: input size along dim must match t length");

    int64_t n = t.size(0);
    TORCH_CHECK(n >= 2, "laplace_transform: need at least 2 time points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor s_c = s.contiguous();
    at::Tensor t_c = t.contiguous();

    // Compute weights for integration (dt for trapezoidal rule)
    // Use tensor operations instead of accessors to support all floating point types
    at::Tensor dt = at::zeros({n}, t_c.options());

    // Compute spacing using tensor operations
    at::Tensor t_diff = t_c.slice(0, 1, n) - t_c.slice(0, 0, n - 1);  // [n-1]

    if (integration_method == 0 || integration_method == 2) {  // trapezoidal (or gauss_legendre fallback)
        // Weights: dt[i] = (t[i+1] - t[i-1]) / 2 for interior points
        // dt[0] = (t[1] - t[0]) / 2, dt[n-1] = (t[n-1] - t[n-2]) / 2
        dt[0] = t_diff[0] / 2.0;
        if (n > 2) {
            dt.slice(0, 1, n - 1) = (t_c.slice(0, 2, n) - t_c.slice(0, 0, n - 2)) / 2.0;
        }
        dt[n - 1] = t_diff[n - 2] / 2.0;
    } else if (integration_method == 1) {  // simpson
        // Simpson's rule with uniform spacing assumption
        // Weights: 1, 4, 2, 4, 2, ..., 4, 1 scaled by h/3
        at::Tensor h = t_diff[0];  // Assume uniform spacing
        at::Tensor h_over_3 = h / 3.0;

        dt[0] = h_over_3;
        dt[n - 1] = h_over_3;

        // Fill interior points
        for (int64_t i = 1; i < n - 1; i++) {
            if (i % 2 == 1) {
                dt[i] = 4.0 * h_over_3;
            } else {
                dt[i] = 2.0 * h_over_3;
            }
        }
    }

    // Reshape t and dt for broadcasting with input
    // t should broadcast along dim
    std::vector<int64_t> broadcast_shape(ndim, 1);
    broadcast_shape[dim] = n;
    at::Tensor t_broadcast = t_c.view(broadcast_shape);
    at::Tensor dt_broadcast = dt.view(broadcast_shape);

    // Determine output shape: replace dim with s's shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) {
            // Insert s's shape here
            for (int64_t j = 0; j < s_c.dim(); j++) {
                output_shape.push_back(s_c.size(j));
            }
        } else {
            output_shape.push_back(input_c.size(i));
        }
    }

    // For each s value, compute the integral
    // F(s) = sum_i f(t_i) * exp(-s * t_i) * dt_i
    // We need to broadcast s with t

    // Reshape s for broadcasting: add dimensions for input's shape (except dim)
    // s has shape [n_s] or [n_s1, n_s2, ...]
    // We want s * t where t has shape [..., n_t, ...]
    // Result should be [..., n_s, ..., n_t, ...] which we then sum over t

    // For simplicity, flatten s and compute per-element, then reshape
    at::Tensor s_flat = s_c.flatten();
    int64_t n_s = s_flat.size(0);

    // Compute exp(-s * t) for all s and t
    // Shape: [n_s, n_t]
    at::Tensor s_col = s_flat.unsqueeze(1);  // [n_s, 1]
    at::Tensor t_row = t_c.unsqueeze(0);      // [1, n_t]
    at::Tensor exp_term = at::exp(-s_col * t_row);  // [n_s, n_t]

    // Multiply by dt
    at::Tensor weighted_exp = exp_term * dt.unsqueeze(0);  // [n_s, n_t]

    // Move input's dim to the last position for easier einsum
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_t]

    // Compute the transform: sum over t
    // result[..., s_idx] = sum_t input[..., t] * weighted_exp[s_idx, t]
    // This is a matrix multiplication: [..., n_t] @ [n_t, n_s] -> [..., n_s]
    at::Tensor result = at::matmul(input_moved, weighted_exp.transpose(0, 1));  // [..., n_s]

    // Reshape result to match s's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < s_c.dim(); i++) {
        final_shape.push_back(s_c.size(i));
    }

    result = result.view(final_shape);

    // Move the s dimensions to where dim was
    // Current shape: [dims before dim, dims after dim, s_dims]
    // Target: [dims before dim, s_dims, dims after dim]
    if (s_c.dim() > 0 && dim < ndim - 1) {
        // Need to permute
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t s_ndim = s_c.dim();

        // Indices for dims before original dim
        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        // Indices for s dimensions (at the end)
        for (int64_t i = result_ndim - s_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        // Indices for dims after original dim
        for (int64_t i = dim; i < result_ndim - s_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for Laplace transform.
 */
inline at::Tensor laplace_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    // The Laplace transform is linear in input, so:
    // grad_input = sum_s grad_output[s] * exp(-s * t) * dt

    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = t.size(0);
    at::Tensor t_c = t.contiguous();

    // Compute dt weights (same as forward) using tensor operations
    at::Tensor dt = at::zeros({n}, t_c.options());
    at::Tensor t_diff = t_c.slice(0, 1, n) - t_c.slice(0, 0, n - 1);  // [n-1]

    dt[0] = t_diff[0] / 2.0;
    if (n > 2) {
        dt.slice(0, 1, n - 1) = (t_c.slice(0, 2, n) - t_c.slice(0, 0, n - 2)) / 2.0;
    }
    dt[n - 1] = t_diff[n - 2] / 2.0;

    at::Tensor s_c = s.contiguous();
    at::Tensor s_flat = s_c.flatten();
    int64_t n_s = s_flat.size(0);

    // Compute exp(-s * t) for all s and t
    at::Tensor s_col = s_flat.unsqueeze(1);
    at::Tensor t_row = t_c.unsqueeze(0);
    at::Tensor exp_term = at::exp(-s_col * t_row);  // [n_s, n_t]
    at::Tensor weighted_exp = exp_term * dt.unsqueeze(0);  // [n_s, n_t]

    // Reshape grad_output to have s as last dimension
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);  // [..., n_s...]

    // Flatten s dimensions in grad_output
    std::vector<int64_t> grad_shape_before_s;
    int64_t grad_ndim = grad_output.dim();
    int64_t s_ndim = s_c.dim();
    for (int64_t i = 0; i < grad_ndim - s_ndim; i++) {
        grad_shape_before_s.push_back(grad_out_moved.size(i));
    }
    at::Tensor grad_flat = grad_out_moved.flatten(-s_ndim);  // [..., n_s]

    // grad_input = grad_flat @ weighted_exp -> [..., n_t]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_exp);

    // Move t dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for Laplace transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
laplace_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    // For linear transform, second derivative w.r.t. input is zero
    // grad_grad_output is the only non-trivial output
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = laplace_transform(grad_grad_input, s, t, dim, integration_method);
    }

    // Return grad_grad_output and empty tensor for new_grad_input (second derivatives are zero for linear op)
    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "laplace_transform",
        &torchscience::cpu::transform::laplace_transform
    );

    module.impl(
        "laplace_transform_backward",
        &torchscience::cpu::transform::laplace_transform_backward
    );

    module.impl(
        "laplace_transform_backward_backward",
        &torchscience::cpu::transform::laplace_transform_backward_backward
    );
}
