#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/zeros_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../impl/integral_transform/hilbert_transform.h"
#include "../../impl/integral_transform/hilbert_transform_backward.h"

namespace torchscience::cuda::integral_transform {

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * CUDA kernel to apply Hilbert frequency response in-place.
 *
 * Multiplies each frequency bin by h[k] = -i * sign(freq[k]):
 *   - k=0 (DC): multiply by 0
 *   - k=1...(n-1)/2 (positive freq): multiply by -i
 *   - k=n/2 (Nyquist, even n): multiply by 0
 *   - k=(n+1)/2...n-1 (negative freq): multiply by +i
 *
 * @param data Complex spectrum data (batch_size * n elements)
 * @param n Signal length (number of frequency bins)
 * @param batch_size Number of signals in batch
 */
template <typename scalar_t>
__global__ void apply_hilbert_response_kernel(
    c10::complex<scalar_t>* __restrict__ data,
    int64_t n,
    int64_t batch_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = n * batch_size;

    if (idx >= total_elements) return;

    int64_t freq_idx = idx % n;

    // Compute h[freq_idx] = -i * sign(freq)
    c10::complex<scalar_t> h;

    if (freq_idx == 0) {
        // DC component: h[0] = 0
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (n % 2 == 0 && freq_idx == n / 2) {
        // Nyquist for even n: h[n/2] = 0
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(0));
    } else if (freq_idx < (n + 1) / 2) {
        // Positive frequencies: h[k] = -i
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(-1));
    } else {
        // Negative frequencies: h[k] = +i
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(1));
    }

    data[idx] = data[idx] * h;
}

/**
 * CUDA kernel to apply backward Hilbert frequency response in-place.
 *
 * For backward pass, multiply by conjugate of h[k]:
 *   h_backward[k] = conj(h[k]) = +i for positive freq, -i for negative freq
 */
template <typename scalar_t>
__global__ void apply_hilbert_backward_response_kernel(
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
        // Backward: +i for positive frequencies
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(1));
    } else {
        // Backward: -i for negative frequencies
        h = c10::complex<scalar_t>(scalar_t(0), scalar_t(-1));
    }

    data[idx] = data[idx] * h;
}

}  // namespace

/**
 * CUDA implementation of Hilbert transform with padding and windowing.
 *
 * Uses cuFFT (via ATen) for FFT/IFFT and custom CUDA kernel for
 * applying the Hilbert frequency response.
 *
 * @param input Input tensor (real or complex)
 * @param n_param Signal length for FFT (-1 means use input size)
 * @param dim Dimension along which to compute the transform
 * @param padding_mode Padding mode (0=constant, 1=reflect, 2=replicate, 3=circular)
 * @param padding_value Value for constant padding
 * @param window Optional window tensor to apply before FFT
 * @return Hilbert transform of the input
 */
at::Tensor hilbert_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(input.is_cuda(), "hilbert_transform: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "hilbert_transform: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    // Normalize dimension
    int64_t ndim = input.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "hilbert_transform: dim out of range (got ", dim, " for tensor with ", ndim, " dimensions)");

    TORCH_CHECK(input.size(dim) > 0, "hilbert_transform: transform dimension must have positive size");

    // Determine FFT length
    int64_t input_size = input.size(dim);
    int64_t n = (n_param > 0) ? n_param : input_size;
    TORCH_CHECK(n > 0, "hilbert_transform: n must be positive");

    at::Tensor processed = input.contiguous();

    // Apply padding if needed
    if (n > input_size) {
        processed = impl::integral_transform::apply_padding(
            processed, n, dim, padding_mode, padding_value
        );
    } else if (n < input_size) {
        // Truncation
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

    // Calculate batch size (product of all dims except transform dim)
    int64_t batch_size = spectrum.numel() / n;

    // Move transform dimension to last for efficient kernel access
    at::Tensor spectrum_transposed = spectrum.movedim(dim, -1).contiguous();

    // Apply Hilbert frequency response using custom CUDA kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t total_elements = batch_size * n;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(
        at::toRealValueType(spectrum_transposed.scalar_type()),
        "hilbert_transform_cuda_kernel",
        [&]() {
            auto* data = reinterpret_cast<c10::complex<scalar_t>*>(
                spectrum_transposed.data_ptr()
            );

            apply_hilbert_response_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
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
 * CUDA backward pass for Hilbert transform.
 */
at::Tensor hilbert_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    TORCH_CHECK(grad_output.is_cuda(), "hilbert_transform_backward: grad_output must be a CUDA tensor");

    // Normalize dimension
    int64_t ndim = input.dim();
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;

    int64_t input_size = input.size(norm_dim);
    int64_t n = (n_param > 0) ? n_param : input_size;

    // Apply -H with size n to grad_output
    // Use the same padding_mode as forward for consistency, though in practice
    // grad_output already has size n so no padding is applied
    at::Tensor grad = -hilbert_transform(grad_output, n_param, dim, padding_mode, padding_value, c10::nullopt);

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
 * CUDA double backward pass for Hilbert transform.
 */
std::tuple<at::Tensor, at::Tensor> hilbert_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    // grad_grad_output = H[grad_grad_input]
    at::Tensor grad_grad_output = hilbert_transform(
        grad_grad_input, n_param, dim, padding_mode, padding_value, window
    );

    // new_grad_input = 0 (H is linear, no second-order term)
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::integral_transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "hilbert_transform",
        &torchscience::cuda::integral_transform::hilbert_transform
    );

    module.impl(
        "hilbert_transform_backward",
        &torchscience::cuda::integral_transform::hilbert_transform_backward
    );

    module.impl(
        "hilbert_transform_backward_backward",
        &torchscience::cuda::integral_transform::hilbert_transform_backward_backward
    );
}
