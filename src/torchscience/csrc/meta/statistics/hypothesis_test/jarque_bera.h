#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Jarque-Bera test for normality - Meta implementation.
 *
 * Returns tensors with correct shape but no data (for shape inference).
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (JB-statistic, p-value) meta tensors
 */
inline std::tuple<at::Tensor, at::Tensor> jarque_bera(const at::Tensor& input) {
    auto output_shape = input.sizes().vec();
    output_shape.pop_back();

    auto options = input.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    return std::make_tuple(statistic, pvalue);
}

/**
 * Backward pass for Jarque-Bera test - Meta implementation.
 */
inline at::Tensor jarque_bera_backward(
    const at::Tensor& grad_statistic,
    const at::Tensor& input
) {
    return at::empty_like(input);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("jarque_bera", &torchscience::meta::statistics::hypothesis_test::jarque_bera);
    m.impl("jarque_bera_backward", &torchscience::meta::statistics::hypothesis_test::jarque_bera_backward);
}
