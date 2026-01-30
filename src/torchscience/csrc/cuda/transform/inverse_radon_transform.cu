#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/inverse_radon_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of inverse Radon transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations that automatically use
 * CUDA kernels when given CUDA tensors.
 */
at::Tensor inverse_radon_transform(
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    TORCH_CHECK(sinogram.is_cuda(), "inverse_radon_transform: sinogram must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(sinogram.device());

    return cpu::transform::inverse_radon_transform(
        sinogram, angles, circle, output_size, filter_type
    );
}

at::Tensor inverse_radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_radon_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::inverse_radon_transform_backward(
        grad_output, sinogram, angles, circle, output_size, filter_type
    );
}

std::tuple<at::Tensor, at::Tensor> inverse_radon_transform_backward_backward(
    const at::Tensor& grad_grad_sinogram,
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    TORCH_CHECK(grad_grad_sinogram.is_cuda(), "inverse_radon_transform_backward_backward: grad_grad_sinogram must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_sinogram.device());

    return cpu::transform::inverse_radon_transform_backward_backward(
        grad_grad_sinogram, grad_output, sinogram, angles, circle, output_size, filter_type
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_radon_transform",
        &torchscience::cuda::transform::inverse_radon_transform
    );

    module.impl(
        "inverse_radon_transform_backward",
        &torchscience::cuda::transform::inverse_radon_transform_backward
    );

    module.impl(
        "inverse_radon_transform_backward_backward",
        &torchscience::cuda::transform::inverse_radon_transform_backward_backward
    );
}
