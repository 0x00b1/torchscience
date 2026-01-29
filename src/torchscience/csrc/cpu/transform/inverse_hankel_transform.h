#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of numerical inverse Hankel transform.
 *
 * Computes f(r) = integral_0^inf F(k) * J_nu(k*r) * k dk
 * using numerical quadrature.
 *
 * For order 0, this is identical to the forward Hankel transform
 * (self-reciprocal).
 *
 * @param input Input tensor F(k) sampled at points k_in
 * @param r_out Output radial points where to evaluate the transform
 * @param k_in Input radial frequency points where input is sampled
 * @param dim Dimension along which to integrate
 * @param order Order nu of the Hankel transform (default 0)
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Inverse Hankel transform evaluated at r_out
 */
inline at::Tensor inverse_hankel_transform(
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& k_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "inverse_hankel_transform: input tensor must be non-empty");
    TORCH_CHECK(k_in.dim() == 1, "inverse_hankel_transform: k_in must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "inverse_hankel_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == k_in.size(0),
        "inverse_hankel_transform: input size along dim must match k_in length");

    int64_t n = k_in.size(0);
    TORCH_CHECK(n >= 2, "inverse_hankel_transform: need at least 2 frequency points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor r_c = r_out.contiguous();
    at::Tensor k_c = k_in.contiguous();

    // Compute weights for integration (dk for trapezoidal rule) using tensor operations
    at::Tensor dk = at::zeros({n}, k_c.options());
    at::Tensor k_diff = k_c.slice(0, 1, n) - k_c.slice(0, 0, n - 1);
    dk[0] = k_diff[0] / 2.0;
    if (n > 2) {
        dk.slice(0, 1, n - 1) = (k_c.slice(0, 2, n) - k_c.slice(0, 0, n - 2)) / 2.0;
    }
    dk[n - 1] = k_diff[n - 2] / 2.0;

    // Flatten r_out for computation
    at::Tensor r_flat = r_c.flatten();
    int64_t n_r = r_flat.size(0);

    // Compute J_nu(k*r) for all k and r
    // Shape: [n_r, n_k]
    at::Tensor r_col = r_flat.unsqueeze(1);  // [n_r, 1]
    at::Tensor k_row = k_c.unsqueeze(0);      // [1, n_k]

    // k * r
    at::Tensor kr = r_col * k_row;  // [n_r, n_k]

    // J_nu(k*r) via dispatcher
    at::Tensor nu_tensor = at::full({}, order, kr.options());
    at::Tensor bessel_term = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::bessel_j", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(nu_tensor, kr);  // [n_r, n_k]

    // Multiply by k * dk for the integral
    at::Tensor weighted_bessel = bessel_term * k_row * dk.unsqueeze(0);  // [n_r, n_k]

    // Move input's dim to the last position for easier matmul
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_k]

    // Compute the transform: sum over k
    // result[..., r_idx] = sum_k input[..., k] * weighted_bessel[r_idx, k]
    at::Tensor result = at::matmul(input_moved, weighted_bessel.transpose(0, 1));  // [..., n_r]

    // Reshape result to match r_out's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < r_c.dim(); i++) {
        final_shape.push_back(r_c.size(i));
    }

    result = result.view(final_shape);

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
 * Backward pass for inverse Hankel transform.
 */
inline at::Tensor inverse_hankel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& k_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = k_in.size(0);
    at::Tensor k_c = k_in.contiguous();

    // Compute dk weights using tensor operations (dtype-agnostic)
    at::Tensor dk = at::zeros({n}, k_c.options());
    at::Tensor k_diff = k_c.slice(0, 1, n) - k_c.slice(0, 0, n - 1);
    dk[0] = k_diff[0] / 2.0;
    if (n > 2) {
        dk.slice(0, 1, n - 1) = (k_c.slice(0, 2, n) - k_c.slice(0, 0, n - 2)) / 2.0;
    }
    dk[n - 1] = k_diff[n - 2] / 2.0;

    at::Tensor r_c = r_out.contiguous();
    at::Tensor r_flat = r_c.flatten();
    int64_t n_r = r_flat.size(0);

    // Compute J_nu(k*r) for all k and r
    at::Tensor r_col = r_flat.unsqueeze(1);
    at::Tensor k_row = k_c.unsqueeze(0);
    at::Tensor kr = r_col * k_row;

    at::Tensor nu_tensor = at::full({}, order, kr.options());
    at::Tensor bessel_term = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::bessel_j", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(nu_tensor, kr);

    at::Tensor weighted_bessel = bessel_term * k_row * dk.unsqueeze(0);

    // Reshape grad_output to have r as last dimension
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten r dimensions in grad_output
    int64_t r_ndim = r_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-r_ndim);

    // grad_input = grad_flat @ weighted_bessel -> [..., n_k]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_bessel);

    // Move k dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for inverse Hankel transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
inverse_hankel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& r_out,
    const at::Tensor& k_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = inverse_hankel_transform(
            grad_grad_input, r_out, k_in, dim, order, integration_method
        );
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_hankel_transform",
        &torchscience::cpu::transform::inverse_hankel_transform
    );

    module.impl(
        "inverse_hankel_transform_backward",
        &torchscience::cpu::transform::inverse_hankel_transform_backward
    );

    module.impl(
        "inverse_hankel_transform_backward_backward",
        &torchscience::cpu::transform::inverse_hankel_transform_backward_backward
    );
}
