#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/abel_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of Abel transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations that automatically use
 * CUDA kernels when given CUDA tensors.
 */
at::Tensor abel_transform(
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.is_cuda(), "abel_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::abel_transform(input, y_out, r_in, dim, integration_method);
}

at::Tensor abel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(grad_output.is_cuda(), "abel_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::abel_transform_backward(
        grad_output, input, y_out, r_in, dim, integration_method
    );
}

std::tuple<at::Tensor, at::Tensor> abel_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "abel_transform_backward_backward: grad_grad_input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::abel_transform_backward_backward(
        grad_grad_input, grad_output, input, y_out, r_in, dim, integration_method
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "abel_transform",
        &torchscience::cuda::transform::abel_transform
    );

    module.impl(
        "abel_transform_backward",
        &torchscience::cuda::transform::abel_transform_backward
    );

    module.impl(
        "abel_transform_backward_backward",
        &torchscience::cuda::transform::abel_transform_backward_backward
    );
}
