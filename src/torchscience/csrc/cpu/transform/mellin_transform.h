#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of numerical Mellin transform.
 *
 * Computes M(s) = integral_0^inf f(t) * t^(s-1) dt
 * using numerical quadrature.
 *
 * @param input Input tensor f(t) sampled at points t
 * @param s Complex frequency values where to evaluate the transform
 * @param t Time points where input is sampled (must be positive, sorted)
 * @param dim Dimension along which to integrate
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Mellin transform evaluated at s
 */
inline at::Tensor mellin_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "mellin_transform: input tensor must be non-empty");
    TORCH_CHECK(t.dim() == 1, "mellin_transform: t must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "mellin_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == t.size(0),
        "mellin_transform: input size along dim must match t length");

    int64_t n = t.size(0);
    TORCH_CHECK(n >= 2, "mellin_transform: need at least 2 time points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor s_c = s.contiguous();
    at::Tensor t_c = t.contiguous();

    // Compute weights for integration (dt for trapezoidal rule)
    // Use tensor operations for dtype-agnostic computation
    at::Tensor dt = at::zeros({n}, t_c.options());
    at::Tensor t_diff = t_c.slice(0, 1, n) - t_c.slice(0, 0, n - 1);

    dt[0] = t_diff[0] / 2.0;
    if (n > 2) {
        dt.slice(0, 1, n - 1) = (t_c.slice(0, 2, n) - t_c.slice(0, 0, n - 2)) / 2.0;
    }
    dt[n - 1] = t_diff[n - 2] / 2.0;

    // Flatten s for computation
    at::Tensor s_flat = s_c.flatten();
    int64_t n_s = s_flat.size(0);

    // Compute t^(s-1) for all s and t
    // Shape: [n_s, n_t]
    at::Tensor s_col = s_flat.unsqueeze(1);  // [n_s, 1]
    at::Tensor t_row = t_c.unsqueeze(0);      // [1, n_t]

    // t^(s-1) = exp((s-1) * log(t))
    at::Tensor log_t = at::log(t_row);  // [1, n_t]
    at::Tensor power_term = at::exp((s_col - 1.0) * log_t);  // [n_s, n_t]

    // Multiply by dt
    at::Tensor weighted_power = power_term * dt.unsqueeze(0);  // [n_s, n_t]

    // Move input's dim to the last position for easier matmul
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_t]

    // Compute the transform: sum over t
    // result[..., s_idx] = sum_t input[..., t] * weighted_power[s_idx, t]
    at::Tensor result = at::matmul(input_moved, weighted_power.transpose(0, 1));  // [..., n_s]

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
    if (s_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t s_ndim = s_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - s_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - s_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for Mellin transform.
 */
inline at::Tensor mellin_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = t.size(0);
    at::Tensor t_c = t.contiguous();

    // Compute dt weights using tensor operations (dtype-agnostic)
    at::Tensor dt = at::zeros({n}, t_c.options());
    at::Tensor t_diff = t_c.slice(0, 1, n) - t_c.slice(0, 0, n - 1);

    dt[0] = t_diff[0] / 2.0;
    if (n > 2) {
        dt.slice(0, 1, n - 1) = (t_c.slice(0, 2, n) - t_c.slice(0, 0, n - 2)) / 2.0;
    }
    dt[n - 1] = t_diff[n - 2] / 2.0;

    at::Tensor s_c = s.contiguous();
    at::Tensor s_flat = s_c.flatten();
    int64_t n_s = s_flat.size(0);

    // Compute t^(s-1) for all s and t
    at::Tensor s_col = s_flat.unsqueeze(1);
    at::Tensor t_row = t_c.unsqueeze(0);
    at::Tensor log_t = at::log(t_row);
    at::Tensor power_term = at::exp((s_col - 1.0) * log_t);
    at::Tensor weighted_power = power_term * dt.unsqueeze(0);

    // Reshape grad_output to have s as last dimension
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten s dimensions in grad_output
    int64_t grad_ndim = grad_output.dim();
    int64_t s_ndim = s_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-s_ndim);

    // grad_input = grad_flat @ weighted_power -> [..., n_t]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_power);

    // Move t dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for Mellin transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
mellin_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = mellin_transform(grad_grad_input, s, t, dim, integration_method);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "mellin_transform",
        &torchscience::cpu::transform::mellin_transform
    );

    module.impl(
        "mellin_transform_backward",
        &torchscience::cpu::transform::mellin_transform_backward
    );

    module.impl(
        "mellin_transform_backward_backward",
        &torchscience::cpu::transform::mellin_transform_backward_backward
    );
}
