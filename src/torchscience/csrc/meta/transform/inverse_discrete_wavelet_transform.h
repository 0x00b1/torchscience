#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::transform {

inline at::Tensor inverse_discrete_wavelet_transform(
    const at::Tensor& coeffs,
    [[maybe_unused]] const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    [[maybe_unused]] int64_t levels,
    [[maybe_unused]] int64_t mode,
    int64_t output_length
) {
    std::vector<int64_t> output_shape(coeffs.sizes().begin(), coeffs.sizes().end());
    output_shape.back() = output_length;

    return at::empty(output_shape, coeffs.options());
}

inline at::Tensor inverse_discrete_wavelet_transform_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    [[maybe_unused]] const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    [[maybe_unused]] int64_t levels,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t output_length
) {
    return at::empty_like(coeffs);
}

inline std::tuple<at::Tensor, at::Tensor> inverse_discrete_wavelet_transform_backward_backward(
    [[maybe_unused]] const at::Tensor& grad_grad_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    [[maybe_unused]] const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    [[maybe_unused]] int64_t levels,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t output_length
) {
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(coeffs));
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "inverse_discrete_wavelet_transform",
        &torchscience::meta::transform::inverse_discrete_wavelet_transform
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward",
        &torchscience::meta::transform::inverse_discrete_wavelet_transform_backward
    );

    module.impl(
        "inverse_discrete_wavelet_transform_backward_backward",
        &torchscience::meta::transform::inverse_discrete_wavelet_transform_backward_backward
    );
}
