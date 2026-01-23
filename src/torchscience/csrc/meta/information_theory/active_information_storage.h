#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor active_information_storage(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(ndim >= 2, "joint must have at least 2 dimensions");

    // Compute output shape (all dims except the last 2)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 2; ++i) {
        output_shape.push_back(joint.size(i));
    }

    if (reduction == "none") {
        if (output_shape.empty()) {
            return at::empty({}, joint.options());
        }
        return at::empty(output_shape, joint.options());
    } else {
        return at::empty({}, joint.options());
    }
}

inline at::Tensor active_information_storage_backward(
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
    m.impl("active_information_storage", &torchscience::meta::information_theory::active_information_storage);
    m.impl("active_information_storage_backward", &torchscience::meta::information_theory::active_information_storage_backward);
}
