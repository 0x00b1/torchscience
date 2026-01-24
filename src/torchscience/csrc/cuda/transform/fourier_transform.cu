#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../cpu/transform/_utils.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of Fourier transform with padding and windowing.
 *
 * Since FFT operations are already optimized for CUDA via cuFFT,
 * this implementation reuses the shared utilities for padding/windowing
 * and delegates the FFT to PyTorch's at::fft_fft which automatically
 * uses cuFFT on CUDA tensors.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @param window Optional window tensor to apply before FFT
 * @param norm Normalization mode (0=backward, 1=ortho, 2=forward)
 * @return Fourier transform of the input
 */
at::Tensor fourier_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window,
    int64_t norm
) {
    TORCH_CHECK(input.is_cuda(), "fourier_transform: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "fourier_transform: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "fourier_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "fourier_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "fourier_transform: n must be positive");

    // Ensure contiguous for efficient operations
    at::Tensor processed = input.contiguous();

    // Apply padding if needed (uses cpu::transform utilities which work on any device)
    if (n > input_size) {
        processed = cpu::transform::apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        // Truncation
        processed = processed.narrow(dim, 0, n);
    }

    // Apply window if provided
    if (window.has_value()) {
        processed = cpu::transform::apply_window(processed, window.value(), dim);
    }

    // Determine normalization string
    c10::optional<c10::string_view> norm_str = c10::nullopt;
    if (norm == 0) {
        norm_str = "backward";
    } else if (norm == 1) {
        norm_str = "ortho";
    } else if (norm == 2) {
        norm_str = "forward";
    }

    // Compute FFT (PyTorch automatically uses cuFFT for CUDA tensors)
    return at::fft_fft(processed, c10::nullopt, dim, norm_str);
}

/**
 * CUDA backward pass for Fourier transform.
 */
at::Tensor fourier_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window,
    int64_t norm
) {
    TORCH_CHECK(grad_output.is_cuda(), "fourier_transform_backward: grad_output must be a CUDA tensor");

    c10::cuda::CUDAGuard device_guard(grad_output.device());

    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Determine normalization
    c10::optional<c10::string_view> norm_str = c10::nullopt;
    if (norm == 0) {
        norm_str = "backward";
    } else if (norm == 1) {
        norm_str = "ortho";
    } else if (norm == 2) {
        norm_str = "forward";
    }

    // The adjoint of FFT is IFFT (with appropriate normalization)
    at::Tensor grad = at::fft_ifft(grad_output, c10::nullopt, norm_dim, norm_str);

    // Adjust scaling for adjoint vs inverse
    if (norm == 0) {
        // backward norm: FFT = sum, IFFT = sum/n
        // adjoint is IFFT * n
        grad = grad * static_cast<double>(n);
    } else if (norm == 2) {
        // forward norm: FFT = sum/n, IFFT = sum
        // adjoint is IFFT / n
        grad = grad / static_cast<double>(n);
    }
    // ortho is self-adjoint

    // If window was applied in forward, multiply gradient by window
    if (window.has_value()) {
        grad = cpu::transform::apply_window(grad, window.value(), norm_dim);
    }

    // Handle real input case - take real part
    if (!input.is_complex()) {
        grad = at::real(grad);
    }

    // Adjust size to match input shape
    grad = cpu::transform::adjust_backward_gradient_size(
        grad, input_size, n, norm_dim, padding_mode
    );

    return grad;
}

/**
 * CUDA double backward pass for Fourier transform.
 */
std::tuple<at::Tensor, at::Tensor> fourier_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window,
    int64_t norm
) {
    // FFT is linear, so grad_grad_output = FFT[grad_grad_input]
    at::Tensor grad_grad_output = fourier_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window, norm
    );

    // No second-order term (FFT is linear)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "fourier_transform",
        &torchscience::cuda::transform::fourier_transform
    );

    module.impl(
        "fourier_transform_backward",
        &torchscience::cuda::transform::fourier_transform_backward
    );

    module.impl(
        "fourier_transform_backward_backward",
        &torchscience::cuda::transform::fourier_transform_backward_backward
    );
}
