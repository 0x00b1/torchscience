#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <torch/library.h>

namespace torchscience::cpu::transform {

/**
 * CPU implementation of FFT-based convolution.
 *
 * Computes the convolution of input with kernel along the specified dimension.
 *
 * @param input Input tensor
 * @param kernel Convolution kernel
 * @param dim Dimension along which to convolve
 * @param mode 0=full, 1=same, 2=valid
 * @return Convolved tensor
 */
inline at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    TORCH_CHECK(input.numel() > 0, "convolution: input tensor must be non-empty");
    TORCH_CHECK(kernel.numel() > 0, "convolution: kernel tensor must be non-empty");

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "convolution: dim out of range");

    // Get sizes - handle 1D kernel case
    int64_t n_input = input.size(dim);
    int64_t n_kernel = kernel.dim() == 1 ? kernel.size(0) : kernel.size(dim);

    // Determine output size based on mode
    int64_t n_output;
    if (mode == 0) {  // full
        n_output = n_input + n_kernel - 1;
    } else if (mode == 1) {  // same
        n_output = n_input;
    } else {  // valid
        TORCH_CHECK(n_input >= n_kernel, "convolution: for 'valid' mode, input must be at least as long as kernel");
        n_output = n_input - n_kernel + 1;
    }

    // FFT size (power of 2 for efficiency)
    int64_t n_fft = n_input + n_kernel - 1;
    // Round up to nearest power of 2
    int64_t n_fft_padded = 1;
    while (n_fft_padded < n_fft) {
        n_fft_padded *= 2;
    }

    // Pad input and kernel to n_fft_padded
    at::Tensor input_padded = input.contiguous();
    at::Tensor kernel_padded = kernel.contiguous();

    // Ensure kernel has same number of dimensions as input for broadcasting
    if (kernel.dim() == 1 && input.dim() > 1) {
        std::vector<int64_t> new_shape(input.dim(), 1);
        new_shape[dim] = n_kernel;
        kernel_padded = kernel_padded.view(new_shape);
    }

    // Zero-pad along dim
    if (n_fft_padded > n_input) {
        std::vector<int64_t> pad_shape(input_padded.sizes().begin(), input_padded.sizes().end());
        pad_shape[dim] = n_fft_padded;
        at::Tensor padded = at::zeros(pad_shape, input_padded.options());
        padded.narrow(dim, 0, n_input).copy_(input_padded);
        input_padded = padded;
    }

    if (n_fft_padded > n_kernel) {
        std::vector<int64_t> pad_shape(kernel_padded.sizes().begin(), kernel_padded.sizes().end());
        pad_shape[dim] = n_fft_padded;
        at::Tensor padded = at::zeros(pad_shape, kernel_padded.options());
        padded.narrow(dim, 0, n_kernel).copy_(kernel_padded);
        kernel_padded = padded;
    }

    // Compute FFTs
    at::Tensor input_fft = at::fft_fft(input_padded, c10::nullopt, dim);
    at::Tensor kernel_fft = at::fft_fft(kernel_padded, c10::nullopt, dim);

    // Multiply in frequency domain
    at::Tensor result_fft = input_fft * kernel_fft;

    // Inverse FFT
    at::Tensor result = at::fft_ifft(result_fft, c10::nullopt, dim);
    result = at::real(result);

    // Extract appropriate portion based on mode
    if (mode == 0) {  // full
        result = result.narrow(dim, 0, n_output);
    } else if (mode == 1) {  // same
        // Center the output
        int64_t start = (n_kernel - 1) / 2;
        result = result.narrow(dim, start, n_output);
    } else {  // valid
        result = result.narrow(dim, n_kernel - 1, n_output);
    }

    return result.contiguous();
}

/**
 * Backward pass for convolution.
 *
 * For convolution y = conv(x, h):
 * - grad_input = conv(grad_output, flip(h), mode='full') with appropriate extraction
 * - grad_kernel = conv(x, grad_output, mode='valid') for valid forward mode
 *                 or extracted from full convolution for other modes
 */
inline std::tuple<at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t n_input = input.size(norm_dim);
    int64_t n_kernel = kernel.dim() == 1 ? kernel.size(0) : kernel.size(norm_dim);
    int64_t n_grad = grad_output.size(norm_dim);

    // Gradient w.r.t. input: conv(grad_output, flip(kernel)) in full mode
    at::Tensor kernel_flipped;
    if (kernel.dim() == 1) {
        kernel_flipped = at::flip(kernel, {0});
    } else {
        kernel_flipped = at::flip(kernel, {norm_dim});
    }

    at::Tensor grad_input = convolution(grad_output, kernel_flipped, norm_dim, 0);  // full mode

    // Extract the valid portion that corresponds to the input size
    if (mode == 0) {  // full: grad_output has n_input + n_kernel - 1 elements
        int64_t start = n_kernel - 1;
        grad_input = grad_input.narrow(norm_dim, start, n_input);
    } else if (mode == 1) {  // same: grad_output has n_input elements
        int64_t start = (n_kernel - 1) / 2;
        grad_input = grad_input.narrow(norm_dim, start, n_input);
    } else {  // valid: grad_output has n_input - n_kernel + 1 elements
        grad_input = grad_input.narrow(norm_dim, 0, n_input);
    }

    // Gradient w.r.t. kernel: grad_h[j] = sum_k grad_y[k] * x[k - j + offset]
    // This is a cross-correlation, computed as conv(flip(x), grad_y, full) with extraction
    // The offset depends on the forward mode
    at::Tensor grad_kernel;

    // Flip input along the convolution dimension
    at::Tensor input_flipped;
    if (input.dim() == 1) {
        input_flipped = at::flip(input, {0});
    } else {
        input_flipped = at::flip(input, {norm_dim});
    }

    // Compute full convolution of flipped input with grad_output
    at::Tensor full_corr = convolution(input_flipped, grad_output, norm_dim, 0);  // full mode

    // Extract based on forward mode
    int64_t start;
    if (mode == 0) {  // full: forward output had n_input + n_kernel - 1 elements
        start = n_input - 1;
    } else if (mode == 1) {  // same: forward output had n_input elements
        start = (n_input - 1) - (n_kernel - 1) / 2;
    } else {  // valid: forward output had n_input - n_kernel + 1 elements
        start = n_input - n_kernel;
    }
    grad_kernel = full_corr.narrow(norm_dim, start, n_kernel);

    // Sum over batch dimensions if kernel was 1D (broadcast)
    if (kernel.dim() == 1 && input.dim() > 1) {
        std::vector<int64_t> sum_dims;
        for (int64_t i = 0; i < ndim; i++) {
            if (i != norm_dim) {
                sum_dims.push_back(i);
            }
        }
        if (!sum_dims.empty()) {
            grad_kernel = grad_kernel.sum(sum_dims);
        }
    }

    return std::make_tuple(grad_input, grad_kernel);
}

/**
 * Double backward pass for convolution.
 *
 * For bilinear convolution y = conv(x, h):
 * - First backward: grad_x = conv(grad_y, flip(h)), grad_h = corr(grad_y, x)
 * - Second backward computes gradients from the cross-derivatives:
 *   - new_grad_h from d(grad_x)/d(h) which involves grad_y
 *   - new_grad_x from d(grad_h)/d(x) which involves grad_y
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_grad_kernel,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    // grad_grad_output from grad_grad_input: convolve with kernel
    // grad_grad_output from grad_grad_kernel: convolve input with grad_grad_kernel
    at::Tensor grad_grad_output = at::zeros_like(grad_output);

    if (grad_grad_input.defined()) {
        grad_grad_output = grad_grad_output + convolution(grad_grad_input, kernel, norm_dim, mode);
    }

    if (grad_grad_kernel.defined()) {
        grad_grad_output = grad_grad_output + convolution(input, grad_grad_kernel, norm_dim, mode);
    }

    // Cross-derivative terms for bilinear convolution
    at::Tensor new_grad_input = at::Tensor();
    at::Tensor new_grad_kernel = at::Tensor();

    // Flip grad_output for the cross-derivative computations
    at::Tensor grad_output_flipped = at::flip(grad_output, {norm_dim});

    if (grad_grad_input.defined()) {
        // new_grad_kernel from d(grad_x)/d(h)
        // grad_x = conv(grad_y, flip(h), valid_for_input_size)
        // d(grad_x)/d(h)^T @ grad_grad_x involves convolving grad_y with grad_grad_x
        // The formula: conv(flip(grad_y), grad_grad_x, valid), then flip result
        new_grad_kernel = convolution(grad_output_flipped, grad_grad_input, norm_dim, 2);  // valid mode

        // Flip along dimension because we differentiated w.r.t. flip(h), not h
        if (kernel.dim() == 1) {
            new_grad_kernel = at::flip(new_grad_kernel, {0});
        } else {
            new_grad_kernel = at::flip(new_grad_kernel, {norm_dim});
        }

        // Sum over batch dimensions if kernel was 1D
        if (kernel.dim() == 1 && input.dim() > 1) {
            std::vector<int64_t> sum_dims;
            for (int64_t i = 0; i < ndim; i++) {
                if (i != norm_dim) {
                    sum_dims.push_back(i);
                }
            }
            if (!sum_dims.empty()) {
                new_grad_kernel = new_grad_kernel.sum(sum_dims);
            }
        }
    }

    if (grad_grad_kernel.defined()) {
        // new_grad_input from d(grad_h)/d(x)
        // grad_h = conv(flip(x), grad_y, full) with extraction
        // d(grad_h)/d(x)^T @ grad_grad_h involves convolving grad_y with grad_grad_h
        // The formula: conv(flip(grad_y), grad_grad_h, valid), then flip result
        new_grad_input = convolution(grad_output_flipped, grad_grad_kernel, norm_dim, 2);  // valid mode

        // Flip along dimension because we differentiated w.r.t. flip(x), not x
        new_grad_input = at::flip(new_grad_input, {norm_dim});
    }

    return std::make_tuple(grad_grad_output, new_grad_input, new_grad_kernel);
}

}  // namespace torchscience::cpu::transform

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "convolution",
        &torchscience::cpu::transform::convolution
    );

    module.impl(
        "convolution_backward",
        &torchscience::cpu::transform::convolution_backward
    );

    module.impl(
        "convolution_backward_backward",
        &torchscience::cpu::transform::convolution_backward_backward
    );
}
