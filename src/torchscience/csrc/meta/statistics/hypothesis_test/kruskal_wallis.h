// src/torchscience/csrc/meta/statistics/hypothesis_test/kruskal_wallis.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Kruskal-Wallis H test - Meta implementation.
 *
 * Returns empty scalar tensors for shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor> kruskal_wallis(
    const at::Tensor& data,
    const at::Tensor& group_sizes
) {
    auto options = data.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);
    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "kruskal_wallis",
        &torchscience::meta::statistics::hypothesis_test::kruskal_wallis
    );
}
