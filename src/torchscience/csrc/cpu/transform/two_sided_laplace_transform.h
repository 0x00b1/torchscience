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
 * CPU implementation of numerical two-sided (bilateral) Laplace transform.
 *
 * Computes F(s) = integral_{-inf}^{+inf} f(t) * exp(-s*t) dt
 * using numerical quadrature.
 *
 * @param input Input tensor f(t) sampled at points t
 * @param s Complex frequency values where to evaluate the transform
 * @param t Time points where input is sampled (can be negative, must be sorted)
 * @param dim Dimension along which to integrate
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Two-sided Laplace transform evaluated at s
 */
inline at::Tensor two_sided_laplace_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "two_sided_laplace_transform: input tensor must be non-empty");
    TORCH_CHECK(t.dim() == 1, "two_sided_laplace_transform: t must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "two_sided_laplace_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == t.size(0),
        "two_sided_laplace_transform: input size along dim must match t length");

    int64_t n = t.size(0);
    TORCH_CHECK(n >= 2, "two_sided_laplace_transform: need at least 2 time points");

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

    // Compute exp(-s * t) for all s and t
    // Shape: [n_s, n_t]
    at::Tensor s_col = s_flat.unsqueeze(1);  // [n_s, 1]
    at::Tensor t_row = t_c.unsqueeze(0);      // [1, n_t]

    // exp(-s * t)
    at::Tensor exp_term = at::exp(-s_col * t_row);  // [n_s, n_t]

    // Multiply by dt
    at::Tensor weighted_exp = exp_term * dt.unsqueeze(0);  // [n_s, n_t]

    // Move input's dim to the last position for easier matmul
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_t]

    // Compute the transform: sum over t
    // result[..., s_idx] = sum_t input[..., t] * weighted_exp[s_idx, t]
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
 * Backward pass for two-sided Laplace transform.
 */
inline at::Tensor two_sided_laplace_transform_backward(
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

    // Compute exp(-s * t) for all s and t
    at::Tensor s_col = s_flat.unsqueeze(1);
    at::Tensor t_row = t_c.unsqueeze(0);
    at::Tensor exp_term = at::exp(-s_col * t_row);
    at::Tensor weighted_exp = exp_term * dt.unsqueeze(0);

    // Reshape grad_output to have s as last dimension
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten s dimensions in grad_output
    int64_t grad_ndim = grad_output.dim();
    int64_t s_ndim = s_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-s_ndim);

    // grad_input = grad_flat @ weighted_exp -> [..., n_t]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_exp);

    // Move t dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for two-sided Laplace transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
two_sided_laplace_transform_backward_backward(
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
        grad_grad_output = two_sided_laplace_transform(grad_grad_input, s, t, dim, integration_method);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "two_sided_laplace_transform",
        &torchscience::cpu::transform::two_sided_laplace_transform
    );

    module.impl(
        "two_sided_laplace_transform_backward",
        &torchscience::cpu::transform::two_sided_laplace_transform_backward
    );

    module.impl(
        "two_sided_laplace_transform_backward_backward",
        &torchscience::cpu::transform::two_sided_laplace_transform_backward_backward
    );
}
