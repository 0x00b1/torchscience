// src/torchscience/csrc/meta/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * One-sample t-test meta implementation.
 *
 * Computes output shape and returns empty tensors for shape inference.
 * Output shape = input shape with last dimension removed.
 *
 * @param input Input tensor where the last dimension contains samples
 * @param popmean Population mean to test against (unused for shape inference)
 * @param alternative Alternative hypothesis (unused for shape inference)
 * @return Tuple of (t_statistic, p_value, degrees_of_freedom) tensors
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> one_sample_t_test(
    const at::Tensor& input,
    [[maybe_unused]] double popmean,
    [[maybe_unused]] c10::string_view alternative
) {
    // Output shape is input shape without the last dimension
    auto output_shape = input.sizes().vec();
    output_shape.pop_back();

    // Create output tensors with the computed shape
    auto options = input.options();
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
        "one_sample_t_test",
        &torchscience::meta::statistics::hypothesis_test::one_sample_t_test
    );
}
