#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include CPU implementation which uses device-agnostic ops
#include "../../cpu/transform/inverse_discrete_wavelet_transform.h"

namespace torchscience::cuda::transform {

/**
 * CUDA implementation of inverse discrete wavelet transform.
 *
 * This implementation reuses the CPU implementation which uses
 * device-agnostic PyTorch operations (conv_transpose1d, cat) that
 * automatically use CUDA kernels when given CUDA tensors.
 */
at::Tensor inverse_discrete_wavelet_transform(
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    TORCH_CHECK(coeffs.is_cuda(), "inverse_discrete_wavelet_transform: coeffs must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(coeffs.device());

    return cpu::transform::inverse_discrete_wavelet_transform(
        coeffs, filter_lo, filter_hi, levels, mode, output_length
    );
}

at::Tensor inverse_discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    TORCH_CHECK(grad_output.is_cuda(), "inverse_discrete_wavelet_transform_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_output.device());

    return cpu::transform::inverse_discrete_wavelet_transform_backward(
        grad_output, coeffs, filter_lo, filter_hi, levels, mode, output_length
    );
}

std::tuple<at::Tensor, at::Tensor> inverse_discrete_wavelet_transform_backward_backward(
    const at::Tensor& grad_grad_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t output_length
) {
    TORCH_CHECK(grad_grad_coeffs.is_cuda(), "inverse_discrete_wavelet_transform_backward_backward: grad must be a CUDA tensor");
    c10::cuda::CUDAGuard device_guard(grad_grad_coeffs.device());

    return cpu::transform::inverse_discrete_wavelet_transform_backward_backward(
        grad_grad_coeffs, grad_output, coeffs, filter_lo, filter_hi, levels, mode, output_length
    );
}

}  // namespace torchscience::cuda::transform

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "inverse_discrete_wavelet_transform",
        &torchscience::cuda::transform::inverse_discrete_wavelet_transform
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward",
        &torchscience::cuda::transform::inverse_discrete_wavelet_transform_backward
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward_backward",
        &torchscience::cuda::transform::inverse_discrete_wavelet_transform_backward_backward
    );
}
