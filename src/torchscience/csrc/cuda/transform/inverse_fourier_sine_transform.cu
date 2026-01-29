#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/inverse_fourier_sine_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of Inverse Discrete Sine Transform (IDST).
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations that automatically use
 * CUDA kernels when given CUDA tensors.
 */
at::Tensor inverse_fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(input.is_cuda(), "inverse_fourier_sine_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::inverse_fourier_sine_transform(
        input, n_param, dim, type, norm
    );
}

at::Tensor inverse_fourier_sine_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_fourier_sine_transform_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::inverse_fourier_sine_transform_backward(
        grad_output, input, n_param, dim, type, norm
    );
}

std::tuple<at::Tensor, at::Tensor> inverse_fourier_sine_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "inverse_fourier_sine_transform_backward_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::inverse_fourier_sine_transform_backward_backward(
        grad_grad_input, grad_output, input, n_param, dim, type, norm
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_fourier_sine_transform",
        &torchscience::cuda::transform::inverse_fourier_sine_transform
    );

    module.impl(
        "inverse_fourier_sine_transform_backward",
        &torchscience::cuda::transform::inverse_fourier_sine_transform_backward
    );

    module.impl(
        "inverse_fourier_sine_transform_backward_backward",
        &torchscience::cuda::transform::inverse_fourier_sine_transform_backward_backward
    );
}
