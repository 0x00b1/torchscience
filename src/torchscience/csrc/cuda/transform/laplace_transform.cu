#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/laplace_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of Laplace transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations (exp, matmul) that
 * automatically use CUDA kernels when given CUDA tensors.
 */
at::Tensor laplace_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(input.is_cuda(), "laplace_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::laplace_transform(input, s, t, dim, integration_method);
}

at::Tensor laplace_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(grad_output.is_cuda(), "laplace_transform_backward: grad_output must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::laplace_transform_backward(
        grad_output, input, s, t, dim, integration_method
    );
}

std::tuple<at::Tensor, at::Tensor> laplace_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "laplace_transform_backward_backward: grad_grad_input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::laplace_transform_backward_backward(
        grad_grad_input, grad_output, input, s, t, dim, integration_method
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "laplace_transform",
        &torchscience::cuda::transform::laplace_transform
    );

    module.impl(
        "laplace_transform_backward",
        &torchscience::cuda::transform::laplace_transform_backward
    );

    module.impl(
        "laplace_transform_backward_backward",
        &torchscience::cuda::transform::laplace_transform_backward_backward
    );
}
