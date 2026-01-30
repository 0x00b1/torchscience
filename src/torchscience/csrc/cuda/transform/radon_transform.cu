#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/radon_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of Radon transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations that automatically use
 * CUDA kernels when given CUDA tensors.
 */
at::Tensor radon_transform(
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    TORCH_CHECK(input.is_cuda(), "radon_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::radon_transform(input, angles, circle);
}

at::Tensor radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    TORCH_CHECK(grad_output.is_cuda(), "radon_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::radon_transform_backward(
        grad_output, input, angles, circle
    );
}

std::tuple<at::Tensor, at::Tensor> radon_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "radon_transform_backward_backward: grad_grad_input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::radon_transform_backward_backward(
        grad_grad_input, grad_output, input, angles, circle
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "radon_transform",
        &torchscience::cuda::transform::radon_transform
    );

    module.impl(
        "radon_transform_backward",
        &torchscience::cuda::transform::radon_transform_backward
    );

    module.impl(
        "radon_transform_backward_backward",
        &torchscience::cuda::transform::radon_transform_backward_backward
    );
}
