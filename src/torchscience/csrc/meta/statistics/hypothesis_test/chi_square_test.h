#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Chi-square goodness-of-fit test - Meta implementation.
 *
 * Returns tensors with correct shape but no data (for shape inference).
 */
inline std::tuple<at::Tensor, at::Tensor> chi_square_test(
    const at::Tensor& observed,
    const c10::optional<at::Tensor>& expected,
    int64_t ddof
) {
    auto output_shape = observed.sizes().vec();
    output_shape.pop_back();

    auto options = observed.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    return std::make_tuple(statistic, pvalue);
}

/**
 * Backward pass for chi-square test - Meta implementation.
 */
inline at::Tensor chi_square_test_backward(
    const at::Tensor& grad_statistic,
    const at::Tensor& observed,
    const c10::optional<at::Tensor>& expected
) {
    return at::empty_like(observed);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("chi_square_test", &torchscience::meta::statistics::hypothesis_test::chi_square_test);
    m.impl("chi_square_test_backward", &torchscience::meta::statistics::hypothesis_test::chi_square_test_backward);
}
