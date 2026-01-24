#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/pow.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of Z-transform.
 *
 * Computes X(z) = sum_{n=0}^{N-1} x[n] z^{-n}
 *
 * @param input Input tensor x[n] (discrete-time sequence)
 * @param z_out Complex values where to evaluate the transform
 * @param dim Dimension along which to transform
 * @return Z-transform evaluated at z_out
 */
inline at::Tensor z_transform(
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    TORCH_CHECK(input.numel() > 0, "z_transform: input tensor must be non-empty");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "z_transform: dim out of range");

    int64_t N = input.size(dim);

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor z_c = z_out.contiguous();

    // Flatten z_out for computation
    at::Tensor z_flat = z_c.flatten();
    int64_t n_z = z_flat.size(0);

    // Create index array n = [0, 1, 2, ..., N-1]
    at::Tensor n_indices = at::arange(N, input_c.options());

    // Compute z^{-n} for all z and n
    // Shape: [n_z, N]
    at::Tensor z_col = z_flat.unsqueeze(1);  // [n_z, 1]
    at::Tensor n_row = n_indices.unsqueeze(0);  // [1, N]

    // z^{-n} = z^{-1}^n
    at::Tensor z_inv = at::pow(z_col, -n_row);  // [n_z, N]

    // Move input's dim to the last position
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., N]

    // Convert input to complex if z is complex (for matmul compatibility)
    if (z_c.is_complex() && !input_moved.is_complex()) {
        if (input_moved.scalar_type() == at::kFloat) {
            input_moved = input_moved.to(at::kComplexFloat);
        } else {
            input_moved = input_moved.to(at::kComplexDouble);
        }
    }

    // Compute the transform: sum over n
    // result[..., z_idx] = sum_n input[..., n] * z^{-n}[z_idx, n]
    at::Tensor result = at::matmul(input_moved, z_inv.transpose(0, 1));  // [..., n_z]

    // Reshape result to match z_out's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < z_c.dim(); i++) {
        final_shape.push_back(z_c.size(i));
    }

    result = result.view(final_shape);

    // Move the z dimensions to where dim was
    if (z_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t z_ndim = z_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - z_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - z_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for Z-transform.
 */
inline at::Tensor z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t N = input.size(dim);
    at::Tensor z_c = z_out.contiguous();
    at::Tensor z_flat = z_c.flatten();
    int64_t n_z = z_flat.size(0);

    // Create index array
    at::Tensor n_indices = at::arange(N, input.options());

    // Compute z^{-n}
    at::Tensor z_col = z_flat.unsqueeze(1);
    at::Tensor n_row = n_indices.unsqueeze(0);
    at::Tensor z_inv = at::pow(z_col, -n_row);  // [n_z, N]

    // Move grad_output's dim to the last position
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten z dimensions
    int64_t z_ndim = z_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-z_ndim);

    // Convert to compatible dtype for matmul
    at::Tensor z_inv_conj = z_inv.conj();
    if (z_inv_conj.is_complex() && !grad_flat.is_complex()) {
        if (grad_flat.scalar_type() == at::kFloat) {
            grad_flat = grad_flat.to(at::kComplexFloat);
        } else {
            grad_flat = grad_flat.to(at::kComplexDouble);
        }
    }

    // grad_input = grad_flat @ conj(z_inv)
    at::Tensor grad_input = at::matmul(grad_flat, z_inv_conj);

    // Move n dimension back
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for Z-transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
z_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = z_transform(grad_grad_input, z_out, dim);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "z_transform",
        &torchscience::cpu::transform::z_transform
    );

    module.impl(
        "z_transform_backward",
        &torchscience::cpu::transform::z_transform_backward
    );

    module.impl(
        "z_transform_backward_backward",
        &torchscience::cpu::transform::z_transform_backward_backward
    );
}
