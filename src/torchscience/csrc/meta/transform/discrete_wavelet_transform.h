#pragma once

#include <tuple>
#include <vector>

#include <torch/library.h>

namespace torchscience::meta::transform {

/**
 * Compute total output length for packed DWT coefficients.
 *
 * The DWT pads the signal, convolves, then downsamples by 2.
 * After padding and valid convolution, output length = input_len,
 * then downsampled to (input_len + 1) / 2.
 */
inline int64_t compute_dwt_output_length(
    int64_t input_length,
    [[maybe_unused]] int64_t filter_length,
    int64_t levels
) {
    int64_t total = 0;
    int64_t current_len = input_length;

    for (int64_t i = 0; i < levels; i++) {
        int64_t coeff_len = (current_len + 1) / 2;
        total += coeff_len;  // detail coefficients
        current_len = coeff_len;
    }
    total += current_len;  // final approximation

    return total;
}

inline at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    int64_t levels,
    [[maybe_unused]] int64_t mode
) {
    int64_t filter_len = filter_lo.size(0);
    int64_t input_len = input.size(-1);

    int64_t output_len = compute_dwt_output_length(input_len, filter_len, levels);

    std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
    output_shape.back() = output_len;

    return at::empty(output_shape, input.options());
}

inline at::Tensor discrete_wavelet_transform_backward(
    [[maybe_unused]] const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    [[maybe_unused]] int64_t levels,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t input_length
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor> discrete_wavelet_transform_backward_backward(
    [[maybe_unused]] const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    [[maybe_unused]] const at::Tensor& filter_lo,
    [[maybe_unused]] const at::Tensor& filter_hi,
    [[maybe_unused]] int64_t levels,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] int64_t input_length
) {
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(input));
}

}  // namespace torchscience::meta::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::meta::transform::discrete_wavelet_transform
    );

    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::meta::transform::discrete_wavelet_transform_backward
    );

    module.impl(
        "discrete_wavelet_transform_backward_backward",
        &torchscience::meta::transform::discrete_wavelet_transform_backward_backward
    );
}
