#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor coinformation(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(ndim >= 2, "joint must have at least 2 dimensions");
    TORCH_CHECK(ndim <= 10, "joint must have at most 10 dimensions");

    // Coinformation outputs a scalar (no batch support for now)
    return at::empty({}, joint.options());
}

inline at::Tensor coinformation_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("coinformation", &torchscience::meta::information_theory::coinformation);
    m.impl("coinformation_backward", &torchscience::meta::information_theory::coinformation_backward);
}
