#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/inverse_z_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of inverse Z-transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations (pow, matmul) that
 * automatically use CUDA kernels when given CUDA tensors.
 */
at::Tensor inverse_z_transform(
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    TORCH_CHECK(input.is_cuda(), "inverse_z_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::inverse_z_transform(input, n_out, z_in, dim);
}

at::Tensor inverse_z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_z_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::inverse_z_transform_backward(grad_output, input, n_out, z_in, dim);
}

std::tuple<at::Tensor, at::Tensor> inverse_z_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& n_out,
    const at::Tensor& z_in,
    int64_t dim
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "inverse_z_transform_backward_backward: grad_grad_input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::inverse_z_transform_backward_backward(
        grad_grad_input, grad_output, input, n_out, z_in, dim
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_z_transform",
        &torchscience::cuda::transform::inverse_z_transform
    );

    module.impl(
        "inverse_z_transform_backward",
        &torchscience::cuda::transform::inverse_z_transform_backward
    );

    module.impl(
        "inverse_z_transform_backward_backward",
        &torchscience::cuda::transform::inverse_z_transform_backward_backward
    );
}
