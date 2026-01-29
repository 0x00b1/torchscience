#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/discrete_wavelet_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of discrete wavelet transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations (conv1d, pad, flip, cat) that
 * automatically use CUDA kernels when given CUDA tensors.
 */
at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode
) {
    TORCH_CHECK(input.is_cuda(), "discrete_wavelet_transform: input must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(input.device());

    return cpu::transform::discrete_wavelet_transform(
        input, filter_lo, filter_hi, levels, mode
    );
}

at::Tensor discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    TORCH_CHECK(grad_output.is_cuda(), "discrete_wavelet_transform_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::discrete_wavelet_transform_backward(
        grad_output, input, filter_lo, filter_hi, levels, mode, input_length
    );
}

std::tuple<at::Tensor, at::Tensor> discrete_wavelet_transform_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    TORCH_CHECK(grad_grad_input.is_cuda(), "discrete_wavelet_transform_backward_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_input.device());

    return cpu::transform::discrete_wavelet_transform_backward_backward(
        grad_grad_input, grad_output, input, filter_lo, filter_hi, levels, mode, input_length
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::cuda::transform::discrete_wavelet_transform
    );

    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::cuda::transform::discrete_wavelet_transform_backward
    );

    module.impl(
        "discrete_wavelet_transform_backward_backward",
        &torchscience::cuda::transform::discrete_wavelet_transform_backward_backward
    );
}
