#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/inverse_mellin_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of inverse Mellin transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations that automatically use
 * CUDA kernels when given CUDA tensors.
 */
at::Tensor inverse_mellin_transform(
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double c,
    int64_t integration_method
) {
    TORCH_CHECK(input.is_cuda(), "inverse_mellin_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::inverse_mellin_transform(
        input, t_out, s_in, dim, c, integration_method
    );
}

at::Tensor inverse_mellin_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double c,
    int64_t integration_method
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_mellin_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::inverse_mellin_transform_backward(
        grad_output, input, t_out, s_in, dim, c, integration_method
    );
}

std::tuple<at::Tensor, at::Tensor> inverse_mellin_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& t_out,
    const at::Tensor& s_in,
    int64_t dim,
    double c,
    int64_t integration_method
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "inverse_mellin_transform_backward_backward: grad_grad_input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::inverse_mellin_transform_backward_backward(
        grad_grad_input, grad_output, input, t_out, s_in, dim, c, integration_method
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_mellin_transform",
        &torchscience::cuda::transform::inverse_mellin_transform
    );

    module.impl(
        "inverse_mellin_transform_backward",
        &torchscience::cuda::transform::inverse_mellin_transform_backward
    );

    module.impl(
        "inverse_mellin_transform_backward_backward",
        &torchscience::cuda::transform::inverse_mellin_transform_backward_backward
    );
}
