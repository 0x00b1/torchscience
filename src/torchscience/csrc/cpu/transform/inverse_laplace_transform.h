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
#include <ATen/ops/real.h>
#include <ATen/ops/imag.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of numerical inverse Laplace transform.
 *
 * Computes f(t) = (1/2pi*i) * integral F(s) * exp(s*t) ds
 * using the Bromwich integral approximation.
 *
 * @param input Input tensor F(s) sampled at frequency points s
 * @param t_out Time points where to evaluate the inverse transform
 * @param s_in Complex frequency values where input is sampled
 * @param dim Dimension along which to integrate
 * @param sigma Real part of the Bromwich contour
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Inverse Laplace transform evaluated at t_out
 */
inline at::Tensor inverse_laplace_transform(
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double sigma,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "inverse_laplace_transform: input tensor must be non-empty");
    TORCH_CHECK(s_in.dim() == 1, "inverse_laplace_transform: s must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "inverse_laplace_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == s_in.size(0),
        "inverse_laplace_transform: input size along dim must match s length");

    int64_t n = s_in.size(0);
    TORCH_CHECK(n >= 2, "inverse_laplace_transform: need at least 2 frequency points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor t_c = t_out.contiguous();
    at::Tensor s_c = s_in.contiguous();

    // Extract omega (imaginary part of s)
    at::Tensor omega;
    if (s_c.is_complex()) {
        omega = at::imag(s_c);
    } else {
        // If s is real, treat as omega = 0 (degenerate case)
        omega = at::zeros_like(s_c);
    }

    // Compute weights for integration (d_omega for trapezoidal rule)
    at::Tensor d_omega = at::zeros({n}, omega.options());

    // Use AT_DISPATCH for proper type handling
    AT_DISPATCH_FLOATING_TYPES(omega.scalar_type(), "inverse_laplace_transform_weights", [&] {
        auto d_omega_accessor = d_omega.accessor<scalar_t, 1>();
        auto omega_accessor = omega.accessor<scalar_t, 1>();

        if (integration_method == 0) {  // trapezoidal
            d_omega_accessor[0] = (omega_accessor[1] - omega_accessor[0]) / 2.0;
            for (int64_t i = 1; i < n - 1; i++) {
                d_omega_accessor[i] = (omega_accessor[i + 1] - omega_accessor[i - 1]) / 2.0;
            }
            d_omega_accessor[n - 1] = (omega_accessor[n - 1] - omega_accessor[n - 2]) / 2.0;
        } else {  // simpson (fall back to trapezoidal for non-uniform)
            d_omega_accessor[0] = (omega_accessor[1] - omega_accessor[0]) / 2.0;
            for (int64_t i = 1; i < n - 1; i++) {
                d_omega_accessor[i] = (omega_accessor[i + 1] - omega_accessor[i - 1]) / 2.0;
            }
            d_omega_accessor[n - 1] = (omega_accessor[n - 1] - omega_accessor[n - 2]) / 2.0;
        }
    });

    // Flatten t_out for computation
    at::Tensor t_flat = t_c.flatten();
    int64_t n_t = t_flat.size(0);

    // Compute the inverse transform using Bromwich integral:
    // f(t) = (exp(sigma*t) / (2*pi)) * integral F(sigma + i*omega) * exp(i*omega*t) d_omega
    //
    // This is essentially an inverse Fourier transform scaled by exp(sigma*t)

    // Compute exp(i*omega*t) for all omega and t
    // Shape: [n_t, n_omega]
    at::Tensor t_col = t_flat.unsqueeze(1);  // [n_t, 1]
    at::Tensor omega_row = omega.unsqueeze(0);  // [1, n_omega]

    // i * omega * t
    at::Tensor i_omega_t = at::complex(
        at::zeros_like(t_col * omega_row),
        t_col * omega_row
    );  // [n_t, n_omega]

    at::Tensor exp_term = at::exp(i_omega_t);  // [n_t, n_omega]

    // Multiply by d_omega / (2*pi)
    double scale = 1.0 / (2.0 * M_PI);
    at::Tensor weighted_exp = exp_term * d_omega.unsqueeze(0) * scale;  // [n_t, n_omega]

    // Move input's dim to the last position for easier matmul
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_omega]

    // Compute the integral: sum over omega
    // result[..., t_idx] = sum_omega input[..., omega] * weighted_exp[t_idx, omega]
    // This is a matrix multiplication: [..., n_omega] @ [n_omega, n_t] -> [..., n_t]
    at::Tensor result = at::matmul(input_moved, weighted_exp.transpose(0, 1).conj());  // [..., n_t]

    // Multiply by exp(sigma * t)
    at::Tensor sigma_t = sigma * t_flat;  // [n_t]
    at::Tensor exp_sigma_t = at::exp(sigma_t);  // [n_t]

    // Broadcast exp_sigma_t to result shape
    std::vector<int64_t> broadcast_shape(result.dim(), 1);
    broadcast_shape[result.dim() - 1] = n_t;
    result = result * exp_sigma_t.view(broadcast_shape);

    // Take real part (inverse Laplace transform of real functions is real)
    if (result.is_complex()) {
        result = at::real(result);
    }

    // Reshape result to match t_out's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < t_c.dim(); i++) {
        final_shape.push_back(t_c.size(i));
    }

    result = result.view(final_shape);

    // Move the t dimensions to where dim was
    if (t_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t t_ndim = t_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - t_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - t_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for inverse Laplace transform.
 */
inline at::Tensor inverse_laplace_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double sigma,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = s_in.size(0);
    at::Tensor s_c = s_in.contiguous();

    // Extract omega
    at::Tensor omega;
    if (s_c.is_complex()) {
        omega = at::imag(s_c);
    } else {
        omega = at::zeros_like(s_c);
    }

    // Compute d_omega weights
    at::Tensor d_omega = at::zeros({n}, omega.options());

    AT_DISPATCH_FLOATING_TYPES(omega.scalar_type(), "inverse_laplace_transform_backward_weights", [&] {
        auto d_omega_accessor = d_omega.accessor<scalar_t, 1>();
        auto omega_accessor = omega.accessor<scalar_t, 1>();

        d_omega_accessor[0] = (omega_accessor[1] - omega_accessor[0]) / 2.0;
        for (int64_t i = 1; i < n - 1; i++) {
            d_omega_accessor[i] = (omega_accessor[i + 1] - omega_accessor[i - 1]) / 2.0;
        }
        d_omega_accessor[n - 1] = (omega_accessor[n - 1] - omega_accessor[n - 2]) / 2.0;
    });

    at::Tensor t_c = t_out.contiguous();
    at::Tensor t_flat = t_c.flatten();
    int64_t n_t = t_flat.size(0);

    // Compute exp(-i*omega*t) for backward (conjugate of forward)
    at::Tensor t_col = t_flat.unsqueeze(1);
    at::Tensor omega_row = omega.unsqueeze(0);

    at::Tensor neg_i_omega_t = at::complex(
        at::zeros_like(t_col * omega_row),
        -(t_col * omega_row)
    );

    at::Tensor exp_term = at::exp(neg_i_omega_t);

    double scale = 1.0 / (2.0 * M_PI);
    at::Tensor weighted_exp = exp_term * d_omega.unsqueeze(0) * scale;

    // Multiply grad_output by exp(sigma * t)
    at::Tensor sigma_t = sigma * t_flat;
    at::Tensor exp_sigma_t = at::exp(sigma_t);

    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Scale by exp(sigma * t)
    std::vector<int64_t> broadcast_shape(grad_out_moved.dim(), 1);
    broadcast_shape[grad_out_moved.dim() - 1] = n_t;
    grad_out_moved = grad_out_moved * exp_sigma_t.view(broadcast_shape);

    // Flatten t dimensions
    int64_t t_ndim = t_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-t_ndim);

    // grad_input = grad_flat @ weighted_exp -> [..., n_omega]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_exp);

    // Move omega dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for inverse Laplace transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
inverse_laplace_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double sigma,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = inverse_laplace_transform(
            grad_grad_input, t_out, s_in, dim, sigma, integration_method
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_laplace_transform",
        &torchscience::cpu::transform::inverse_laplace_transform
    );

    module.impl(
        "inverse_laplace_transform_backward",
        &torchscience::cpu::transform::inverse_laplace_transform_backward
    );

    module.impl(
        "inverse_laplace_transform_backward_backward",
        &torchscience::cpu::transform::inverse_laplace_transform_backward_backward
    );
}
