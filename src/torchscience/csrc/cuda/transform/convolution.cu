#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/convolution.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of FFT-based convolution.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations (fft_fft, fft_ifft, etc.) that
 * automatically use CUDA kernels when given CUDA tensors.
 */
at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    TORCH_CHECK(input.is_cuda(), "convolution: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::convolution(input, kernel, dim, mode);
}

std::tuple<at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    TORCH_CHECK(grad_output.is_cuda(), "convolution_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::convolution_backward(
        grad_output, input, kernel, dim, mode
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_grad_kernel,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    TORCH_CHECK(grad_grad_input.is_cuda() || grad_grad_kernel.is_cuda(),
        "convolution_backward_backward: at least one grad must be a CUDA tensor");
    at::Device device = grad_grad_input.defined() ? grad_grad_input.device() : grad_grad_kernel.device();
    c10::cuda::CUDAGuard device_guard(device);

    return cpu::transform::convolution_backward_backward(
        grad_grad_input, grad_grad_kernel, grad_output, input, kernel, dim, mode
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "convolution",
        &torchscience::cuda::transform::convolution
    );

    module.impl(
        "convolution_backward",
        &torchscience::cuda::transform::convolution_backward
    );

    module.impl(
        "convolution_backward_backward",
        &torchscience::cuda::transform::convolution_backward_backward
    );
}
