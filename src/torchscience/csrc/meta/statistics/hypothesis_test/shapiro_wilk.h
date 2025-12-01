// src/torchscience/csrc/meta/statistics/hypothesis_test/shapiro_wilk.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Shapiro-Wilk test meta implementation.
 *
 * Computes output shape and returns empty tensors for shape inference.
 * Output shape = input shape with last dimension removed.
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (W-statistic, p-value) tensors
 */
inline std::tuple<at::Tensor, at::Tensor> shapiro_wilk(
    const at::Tensor& input
) {
    // Output shape is input shape without the last dimension
    auto output_shape = input.sizes().vec();
    output_shape.pop_back();

    // Create output tensors with the computed shape
    auto options = input.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "shapiro_wilk",
        &torchscience::meta::statistics::hypothesis_test::shapiro_wilk
    );
}
