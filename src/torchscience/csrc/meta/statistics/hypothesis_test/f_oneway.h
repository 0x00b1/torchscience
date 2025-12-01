#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

/**
 * F-oneway meta implementation for shape inference.
 *
 * @param data Concatenated data from all groups
 * @param group_sizes Tensor of group sizes
 * @return Tuple of (F-statistic, p-value) tensors (scalar shapes)
 */
inline std::tuple<at::Tensor, at::Tensor> f_oneway(
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
    m.impl("f_oneway", &torchscience::meta::statistics::hypothesis_test::f_oneway);
}
