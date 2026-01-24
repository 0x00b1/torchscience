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
 * CPU implementation of numerical Hankel transform.
 *
 * Computes H_nu{f}(k) = integral_0^inf f(r) * J_nu(k*r) * r dr
 * using numerical quadrature.
 *
 * @param input Input tensor f(r) sampled at points r_in
 * @param k_out Output radial frequency values where to evaluate the transform
 * @param r_in Radial points where input is sampled (must be positive, sorted)
 * @param dim Dimension along which to integrate
 * @param order Order nu of the Hankel transform (default 0)
 * @param integration_method 0=trapezoidal, 1=simpson
 * @return Hankel transform evaluated at k_out
 */
inline at::Tensor hankel_transform(
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    TORCH_CHECK(input.numel() > 0, "hankel_transform: input tensor must be non-empty");
    TORCH_CHECK(r_in.dim() == 1, "hankel_transform: r_in must be 1-dimensional");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "hankel_transform: dim out of range");
    TORCH_CHECK(input.size(dim) == r_in.size(0),
        "hankel_transform: input size along dim must match r_in length");

    int64_t n = r_in.size(0);
    TORCH_CHECK(n >= 2, "hankel_transform: need at least 2 radial points");

    // Ensure contiguous tensors
    at::Tensor input_c = input.contiguous();
    at::Tensor k_c = k_out.contiguous();
    at::Tensor r_c = r_in.contiguous();

    // Compute weights for integration (dr for trapezoidal rule)
    at::Tensor dr = at::zeros({n}, r_c.options());
    auto dr_accessor = dr.accessor<double, 1>();
    auto r_accessor = r_c.accessor<double, 1>();

    if (integration_method == 0) {  // trapezoidal
        dr_accessor[0] = (r_accessor[1] - r_accessor[0]) / 2.0;
        for (int64_t i = 1; i < n - 1; i++) {
            dr_accessor[i] = (r_accessor[i + 1] - r_accessor[i - 1]) / 2.0;
        }
        dr_accessor[n - 1] = (r_accessor[n - 1] - r_accessor[n - 2]) / 2.0;
    } else {  // simpson or default to trapezoidal
        dr_accessor[0] = (r_accessor[1] - r_accessor[0]) / 2.0;
        for (int64_t i = 1; i < n - 1; i++) {
            dr_accessor[i] = (r_accessor[i + 1] - r_accessor[i - 1]) / 2.0;
        }
        dr_accessor[n - 1] = (r_accessor[n - 1] - r_accessor[n - 2]) / 2.0;
    }

    // Flatten k_out for computation
    at::Tensor k_flat = k_c.flatten();
    int64_t n_k = k_flat.size(0);

    // Compute J_nu(k*r) for all k and r
    // Shape: [n_k, n_r]
    at::Tensor k_col = k_flat.unsqueeze(1);  // [n_k, 1]
    at::Tensor r_row = r_c.unsqueeze(0);      // [1, n_r]

    // k * r
    at::Tensor kr = k_col * r_row;  // [n_k, n_r]

    // J_nu(k*r) via dispatcher
    at::Tensor nu_tensor = at::full({}, order, kr.options());
    at::Tensor bessel_term = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::bessel_j", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(nu_tensor, kr);  // [n_k, n_r]

    // Multiply by r * dr for the integral
    at::Tensor weighted_bessel = bessel_term * r_row * dr.unsqueeze(0);  // [n_k, n_r]

    // Move input's dim to the last position for easier matmul
    at::Tensor input_moved = input_c.movedim(dim, -1);  // [..., n_r]

    // Compute the transform: sum over r
    // result[..., k_idx] = sum_r input[..., r] * weighted_bessel[k_idx, r]
    at::Tensor result = at::matmul(input_moved, weighted_bessel.transpose(0, 1));  // [..., n_k]

    // Reshape result to match k_out's original shape
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < input_c.dim() - 1; i++) {
        if (i < dim) {
            final_shape.push_back(input_c.size(i));
        } else {
            final_shape.push_back(input_c.size(i + 1));
        }
    }
    for (int64_t i = 0; i < k_c.dim(); i++) {
        final_shape.push_back(k_c.size(i));
    }

    result = result.view(final_shape);

    // Move the k dimensions to where dim was
    if (k_c.dim() > 0 && dim < ndim - 1) {
        std::vector<int64_t> perm;
        int64_t result_ndim = result.dim();
        int64_t k_ndim = k_c.dim();

        for (int64_t i = 0; i < dim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = result_ndim - k_ndim; i < result_ndim; i++) {
            perm.push_back(i);
        }
        for (int64_t i = dim; i < result_ndim - k_ndim; i++) {
            perm.push_back(i);
        }

        result = result.permute(perm);
    }

    return result.contiguous();
}

/**
 * Backward pass for Hankel transform.
 */
inline at::Tensor hankel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }

    int64_t n = r_in.size(0);
    at::Tensor r_c = r_in.contiguous();

    // Compute dr weights
    at::Tensor dr = at::zeros({n}, r_c.options());
    auto dr_accessor = dr.accessor<double, 1>();
    auto r_accessor = r_c.accessor<double, 1>();

    dr_accessor[0] = (r_accessor[1] - r_accessor[0]) / 2.0;
    for (int64_t i = 1; i < n - 1; i++) {
        dr_accessor[i] = (r_accessor[i + 1] - r_accessor[i - 1]) / 2.0;
    }
    dr_accessor[n - 1] = (r_accessor[n - 1] - r_accessor[n - 2]) / 2.0;

    at::Tensor k_c = k_out.contiguous();
    at::Tensor k_flat = k_c.flatten();
    int64_t n_k = k_flat.size(0);

    // Compute J_nu(k*r) for all k and r
    at::Tensor k_col = k_flat.unsqueeze(1);
    at::Tensor r_row = r_c.unsqueeze(0);
    at::Tensor kr = k_col * r_row;

    at::Tensor nu_tensor = at::full({}, order, kr.options());
    at::Tensor bessel_term = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::bessel_j", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(nu_tensor, kr);

    at::Tensor weighted_bessel = bessel_term * r_row * dr.unsqueeze(0);

    // Reshape grad_output to have k as last dimension
    at::Tensor grad_out_moved = grad_output.movedim(dim, -1);

    // Flatten k dimensions in grad_output
    int64_t grad_ndim = grad_output.dim();
    int64_t k_ndim = k_c.dim();
    at::Tensor grad_flat = grad_out_moved.flatten(-k_ndim);

    // grad_input = grad_flat @ weighted_bessel -> [..., n_r]
    at::Tensor grad_input = at::matmul(grad_flat, weighted_bessel);

    // Move r dimension back to original position
    grad_input = grad_input.movedim(-1, dim);

    return grad_input;
}

/**
 * Double backward pass for Hankel transform.
 */
inline std::tuple<at::Tensor, at::Tensor>
hankel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& k_out,
    const at::Tensor& r_in,
    int64_t dim,
    double order,
    int64_t integration_method
) {
    at::Tensor grad_grad_output = at::Tensor();

    if (grad_grad_input.defined()) {
        grad_grad_output = hankel_transform(grad_grad_input, k_out, r_in, dim, order, integration_method);
    }

    return std::make_tuple(grad_grad_output, at::Tensor());
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "hankel_transform",
        &torchscience::cpu::transform::hankel_transform
    );

    module.impl(
        "hankel_transform_backward",
        &torchscience::cpu::transform::hankel_transform_backward
    );

    module.impl(
        "hankel_transform_backward_backward",
        &torchscience::cpu::transform::hankel_transform_backward_backward
    );
}
