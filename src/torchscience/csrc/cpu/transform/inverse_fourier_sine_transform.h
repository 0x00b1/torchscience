#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

#include "fourier_sine_transform.h"

namespace torchscience::cpu::transform {

/**
 * CPU implementation of Inverse Discrete Sine Transform (IDST).
 *
 * The inverse of DST-II is DST-III, and vice versa.
 * DST-I and DST-IV are their own inverses (up to scaling).
 *
 * With ortho normalization, DST and IDST are true inverses.
 * With backward normalization, IDST includes the 1/(2N) scaling factor
 * to properly invert the forward transform.
 */
inline at::Tensor inverse_fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(input.numel() > 0, "inverse_fourier_sine_transform: input tensor must be non-empty");
    TORCH_CHECK(type >= 1 && type <= 4, "inverse_fourier_sine_transform: type must be 1, 2, 3, or 4");

    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim;
    if (norm_dim < 0) {
        norm_dim += ndim;
    }

    // Determine signal length
    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // IDST is computed by applying the corresponding inverse DST type
    // IDST-II = DST-III, IDST-III = DST-II
    // IDST-I = DST-I, IDST-IV = DST-IV (self-inverse)

    int64_t inverse_type;
    if (type == 2) {
        inverse_type = 3;
    } else if (type == 3) {
        inverse_type = 2;
    } else {
        inverse_type = type;  // Type I and IV are self-inverse
    }

    at::Tensor result = fourier_sine_transform(input, n_param, dim, inverse_type, norm);

    // For backward (unnormalized) mode, apply the 1/(2N) scaling factor
    // to properly invert the forward DST
    if (norm == 0) {  // backward normalization
        if (type == 1) {
            // DST-I: scaling is 1/(2*(N+1))
            result = result / (2.0 * (n + 1));
        } else {
            // DST-II, III, IV: scaling is 1/(2N)
            result = result / (2.0 * n);
        }
    }

    return result;
}

/**
 * Backward pass for IDST on CPU.
 *
 * IDST forward:
 * - ortho: result = DST_inverse_type(input)
 * - backward: result = DST_inverse_type(input) / (2N)
 *
 * The gradient must account for the 1/(2N) scaling factor in backward mode.
 */
inline at::Tensor inverse_fourier_sine_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    int64_t inverse_type;
    if (type == 2) {
        inverse_type = 3;
    } else if (type == 3) {
        inverse_type = 2;
    } else {
        inverse_type = type;
    }

    at::Tensor grad = fourier_sine_transform_backward(grad_output, input, n_param, dim, inverse_type, norm);

    // For backward (unnormalized) mode, apply the 1/(2N) scaling factor
    // that was applied in the forward pass
    if (norm == 0) {
        if (type == 1) {
            grad = grad / (2.0 * (n + 1));
        } else {
            grad = grad / (2.0 * n);
        }
    }

    return grad;
}

/**
 * Double backward pass for IDST on CPU.
 */
inline std::tuple<at::Tensor, at::Tensor> inverse_fourier_sine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    int64_t inverse_type;
    if (type == 2) {
        inverse_type = 3;
    } else if (type == 3) {
        inverse_type = 2;
    } else {
        inverse_type = type;
    }

    return fourier_sine_transform_backward_backward(
        grad_grad_input, grad_output, input, n_param, dim, inverse_type, norm
    );
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "inverse_fourier_sine_transform",
        &torchscience::cpu::transform::inverse_fourier_sine_transform
    );

    module.impl(
        "inverse_fourier_sine_transform_backward",
        &torchscience::cpu::transform::inverse_fourier_sine_transform_backward
    );

    module.impl(
        "inverse_fourier_sine_transform_backward_backward",
        &torchscience::cpu::transform::inverse_fourier_sine_transform_backward_backward
    );
}
