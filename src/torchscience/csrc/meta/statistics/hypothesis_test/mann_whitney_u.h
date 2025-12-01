// src/torchscience/csrc/meta/statistics/hypothesis_test/mann_whitney_u.h
#pragma once

#include <tuple>
#include <string>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Mann-Whitney U test meta implementation.
 *
 * Returns empty scalar tensors for shape inference.
 *
 * @param x First sample (1D tensor)
 * @param y Second sample (1D tensor)
 * @param alternative Alternative hypothesis (not used for shape inference)
 * @return Tuple of (U-statistic, p-value) tensors (scalars)
 */
inline std::tuple<at::Tensor, at::Tensor> mann_whitney_u(
    const at::Tensor& x,
    const at::Tensor& y,
    c10::string_view alternative
) {
    // Output is always scalar
    auto options = x.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "mann_whitney_u",
        &torchscience::meta::statistics::hypothesis_test::mann_whitney_u
    );
}
