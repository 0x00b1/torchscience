// src/torchscience/csrc/meta/statistics/hypothesis_test/wilcoxon_signed_rank.h
#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * Wilcoxon signed-rank test - Meta implementation.
 *
 * Returns empty scalar tensors for shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor> wilcoxon_signed_rank(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& y,
    c10::string_view alternative,
    c10::string_view zero_method
) {
    auto options = x.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);
    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl(
        "wilcoxon_signed_rank",
        &torchscience::meta::statistics::hypothesis_test::wilcoxon_signed_rank
    );
}
