#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline std::vector<at::Tensor> partial_information_decomposition(
    const at::Tensor& joint,
    const std::string& method,
    const std::string& input_type,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(ndim >= 3, "joint must have at least 3 dimensions");

    // Compute output shape (all dims except the last 3)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) {
        output_shape.push_back(joint.size(i));
    }

    at::Tensor redundancy, unique_x, unique_y, synergy, mutual_info;

    if (output_shape.empty()) {
        redundancy = at::empty({}, joint.options());
        unique_x = at::empty({}, joint.options());
        unique_y = at::empty({}, joint.options());
        synergy = at::empty({}, joint.options());
        mutual_info = at::empty({}, joint.options());
    } else {
        redundancy = at::empty(output_shape, joint.options());
        unique_x = at::empty(output_shape, joint.options());
        unique_y = at::empty(output_shape, joint.options());
        synergy = at::empty(output_shape, joint.options());
        mutual_info = at::empty(output_shape, joint.options());
    }

    return {redundancy, unique_x, unique_y, synergy, mutual_info};
}

inline std::vector<at::Tensor> partial_information_decomposition_backward(
    const at::Tensor& grad_redundancy,
    const at::Tensor& grad_unique_x,
    const at::Tensor& grad_unique_y,
    const at::Tensor& grad_synergy,
    const at::Tensor& grad_mutual_info,
    const at::Tensor& joint,
    const std::string& method,
    const std::string& input_type,
    c10::optional<double> base
) {
    return {at::empty_like(joint)};
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("partial_information_decomposition", &torchscience::meta::information_theory::partial_information_decomposition);
    m.impl("partial_information_decomposition_backward", &torchscience::meta::information_theory::partial_information_decomposition_backward);
}
