#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/pow.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of inverse Z-transform.
 *
 * Computes x[n] = (1/M) * sum_{k=0}^{M-1} X(z_k) z_k^{n}
 * where M is the number of z samples (typically on a circle).
 *
 * This is exact when z_k are uniformly spaced on a circle and X(z)
 * is sampled correctly.
 *
 * @param input Input tensor X(z) sampled at points z_in
 * @param n_out Sample indices where to evaluate the inverse transform
 * @param z_in Complex values where input is sampled
 * @param dim Dimension along which to transform
 * @return Inverse Z-transform evaluated at n_out
 */
inline at::Tensor inverse_z_transform(
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    TORCH_CHECK(input.numel() > 0, "inverse_z_transform: input tensor must be non-empty");
    TORCH_CHECK(z_in.dim() == 1, "inverse_z_transform: z_in must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "inverse_z_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == z_in.size(0),
        "inverse_z_transform: input size along dim must match z_in length");

    int64_t M = z_in.size(0);

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor n_c = n_out.contiguous();
    at::Tensor z_c = z_in.contiguous();

    // Flatten n_out for computation
    at::Tensor n_flat = n_c.flatten();
    int64_t n_n = n_flat.size(0);

    // Compute z^n for all z and n
    // Shape: [n_n, M]
    at::Tensor n_col = n_flat.unsqueeze(1);  // [n_n, 1]
    at::Tensor z_row = z_c.unsqueeze(0);      // [1, M]

    at::Tensor z_pow_n = at::pow(z_row, n_col);  // [n_n, M]

    // Move input's dim to the last position
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., M]

    // Convert input to complex if z is complex (for matmul compatibility)
    if (z_c.is_complex() && !input_moved.is_complex()) {
        if (input_moved.scalar_type() == at::kFloat) {
            input_moved = input_moved.to(at::kComplexFloat);
        } else {
            input_moved = input_moved.to(at::kComplexDouble);
        }
    }

    // Compute the inverse transform: (1/M) * sum over z
    // result[..., n_idx] = (1/M) * sum_k input[..., k] * z_k^{n_idx}
    at::Tensor result = at::matmul(input_moved, z_pow_n.transpose(0, 1)) / static_cast<double>(M);  // [..., n_n]

    // Take real part if input was from real sequence
    if (!input_c.is_complex() || result.is_complex()) {
        result = at::real(result);
    }

    // Reshape result to match n_out's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < n_c.dim(); i++) {
        final_shape.push_back(n_c.size(i));
    }

    result = result.view(final_shape);

    // Move the n dimensions to where dim was
    if (n_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t n_ndim = n_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - n_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - n_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for inverse Z-transform.
 */
inline at::Tensor inverse_z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t M = z_in.size(0);
    at::Tensor n_c = n_out.contiguous();
    at::Tensor z_c = z_in.contiguous();
    at::Tensor n_flat = n_c.flatten();
    int64_t n_n = n_flat.size(0);

    // Compute z^n
    at::Tensor n_col = n_flat.unsqueeze(1);
    at::Tensor z_row = z_c.unsqueeze(0);
    at::Tensor z_pow_n = at::pow(z_row, n_col);  // [n_n, M]

    // Move grad_output's dim to the last position
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten n dimensions
    int64_t n_ndim = n_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-n_ndim);

    // Convert to compatible dtype for matmul
    at::Tensor z_pow_n_conj = z_pow_n.conj();
    if (z_pow_n_conj.is_complex() && !grad_flat.is_complex()) {
        if (grad_flat.scalar_type() == at::kFloat) {
            grad_flat = grad_flat.to(at::kComplexFloat);
        } else {
            grad_flat = grad_flat.to(at::kComplexDouble);
        }
    }

    // grad_input = (1/M) * grad_flat @ conj(z_pow_n)
    at::Tensor grad_input = at::matmul(grad_flat, z_pow_n_conj) / static_cast<double>(M);

    // Move z dimension back
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for inverse Z-transform.
 */
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

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_z_transform",
        &torchscience::cpu::transform::inverse_z_transform
    );

    module.impl(
        "inverse_z_transform_backward",
        &torchscience::cpu::transform::inverse_z_transform_backward
    );

    module.impl(
        "inverse_z_transform_backward_backward",
        &torchscience::cpu::transform::inverse_z_transform_backward_backward
    );
}
