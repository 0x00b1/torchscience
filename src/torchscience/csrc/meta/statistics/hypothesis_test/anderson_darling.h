// src/torchscience/csrc/meta/statistics/hypothesis_test/anderson_darling.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Anderson-Darling test meta implementation.
 *
 * Computes output shape and returns empty tensors for shape inference.
 * Output shape = input shape with last dimension removed.
 *
 * @param input Input tensor where the last dimension contains samples
 * @return Tuple of (A^2 statistic, critical_values, significance_levels) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> anderson_darling(
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

    // Critical values: batch_shape + 5
    auto cv_shape = output_shape;
    cv_shape.push_back(5);
    at::Tensor critical_values = at::empty(cv_shape, options);

    // Significance levels: fixed shape (5,)
    at::Tensor significance_levels = at::empty({5}, options);

    return std::make_tuple(statistic, critical_values, significance_levels);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "anderson_darling",
        &torchscience::meta::statistics::hypothesis_test::anderson_darling
    );
}
