#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../impl/integral_transform/hilbert_transform.h"
#include "../../impl/integral_transform/inverse_hilbert_transform.h"
#include "../../impl/integral_transform/inverse_hilbert_transform_backward.h"

namespace torchscience::cuda::integral_transform {

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * CUDA kernel to apply inverse Hilbert frequency response in-place.
 */
template <typename scalar_t>
__global__ void apply_inverse_hilbert_response_kernel(
    c10::complex<scalar_t>* __restrict__ data,
    int64_t n,
    int64_t batch_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = n * batch_size;

    if (idx >= total_elements) return;

    int64_t freq_idx = idx % n;

    c10::complex<scalar_t> h;

    if (freq_idx == 0) {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (n % 2 == 0 && freq_idx == n / 2) {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (freq_idx < (n + 1) / 2) {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(1));
    } else {
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(-1));
    }

    data[idx] = data[idx] * h;
}

}  // namespace

/**
 * CUDA implementation of inverse Hilbert transform with padding and windowing.
 */
at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.is_cuda(), "inverse_hilbert_transform: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "inverse_hilbert_transform: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "inverse_hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "inverse_hilbert_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "inverse_hilbert_transform: n must be positive");

    at::Tensor processed = input.contiguous();

    // Apply padding if needed
    if (n > input_size) {
        processed = impl::integral_transform::apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        processed = processed.narrow(dim, 0, n);
    }

    // Apply window if provided
    if (window.has_value()) {
        processed = impl::integral_transform::apply_window(
            processed, window.value(), dim
        );
    }

    // Compute FFT along specified dimension
    at::Tensor spectrum = at::fft_fft(processed, c10::nullopt, dim);

    // Calculate batch size
    int64_t batch_size = spectrum.numel() / n;

    // Move transform dimension to last for efficient kernel access
    at::Tensor spectrum_transposed = spectrum.movedim(dim, -1).contiguous();

    // Apply inverse Hilbert frequency response using custom CUDA kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t total_elements = batch_size * n;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(
        at::toRealValueType(spectrum_transposed.scalar_type()),
        "inverse_hilbert_transform_cuda_kernel",
        [&]() {
            auto* data = reinterpret_cast<c10::complex<scalar_t>*>(
                spectrum_transposed.data_ptr()
            );

            apply_inverse_hilbert_response_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                data, n, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    // Move transform dimension back to original position
    at::Tensor modified_spectrum = spectrum_transposed.movedim(-1, dim).contiguous();

    // Compute inverse FFT
    at::Tensor result = at::fft_ifft(modified_spectrum, c10::nullopt, dim);

    // If input was real, return real part
    if (!input.is_complex()) {
        result = at::real(result);
    }

    return result;
}

/**
 * CUDA backward pass for inverse Hilbert transform.
 */
at::Tensor inverse_hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_hilbert_transform_backward: grad_output must be a CUDA tensor");

    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H^{-1} with size n to grad_output
    at::Tensor grad = -inverse_hilbert_transform(grad_output, n_param, dim, 0, 0.0, c10::nullopt);

    // If window was applied in forward, multiply gradient by window BEFORE size adjustment
    // (chain rule: d/dx[w*x] = w * d/dx[x], where w operates at padded size)
    if (window.has_value()) {
        grad = impl::integral_transform::apply_window(grad, window.value(), norm_dim);
    }

    // Adjust size to match input shape (with proper gradient accumulation for padding)
    grad = impl::integral_transform::adjust_backward_gradient_size(
        grad, input_size, n, norm_dim, padding_mode
    );

    return grad;
}

/**
 * CUDA double backward pass for inverse Hilbert transform.
 */
std::tuple<at::Tensor, at::Tensor> inverse_hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // grad_grad_output = H^{-1}[grad_grad_input]
    at::Tensor grad_grad_output = inverse_hilbert_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window
    );

    // new_grad_input = 0 (H^{-1} is linear, no second-order term)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform
    );

    module.impl(
        "inverse_hilbert_transform_backward",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform_backward
    );

    module.impl(
        "inverse_hilbert_transform_backward_backward",
        &torchscience::cuda::integral_transform::inverse_hilbert_transform_backward_backward
    );
}
