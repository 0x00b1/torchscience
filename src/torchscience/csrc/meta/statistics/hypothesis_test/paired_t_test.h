// src/torchscience/csrc/meta/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Paired t-test meta implementation.
 *
 * Computes output shape and returns empty tensors for shape inference.
 * Output shape = input1 shape with last dimension removed.
 *
 * @param input1 First sample tensor (used for output shape)
 * @param input2 Second sample tensor (unused for shape inference)
 * @param alternative Alternative hypothesis (unused for shape inference)
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> paired_t_test(
    const at::Tensor& input1,
    [[maybe_unused]] const at::Tensor& input2,
    [[maybe_unused]] c10::string_view alternative
) {
    // Output shape is input1 shape without the last dimension
    auto output_shape = input1.sizes().vec();
    output_shape.pop_back();

    // Create output tensors with the computed shape
    auto options = input1.options();
    at::Tensor t_stat = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor p_value = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor df = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    return std::make_tuple(t_stat, p_value, df);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "paired_t_test",
        &torchscience::meta::statistics::hypothesis_test::paired_t_test
    );
}
