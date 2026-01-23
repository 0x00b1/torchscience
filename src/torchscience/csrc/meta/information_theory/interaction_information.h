#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor interaction_information(
    const at::Tensor& joint,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    TORCH_CHECK(joint.dim() == 3, "joint must have exactly 3 dimensions for interaction information");

    // Interaction information outputs a scalar
    return at::empty({}, joint.options());
}

inline at::Tensor interaction_information_backward(
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
    m.impl("interaction_information", &torchscience::meta::information_theory::interaction_information);
    m.impl("interaction_information_backward", &torchscience::meta::information_theory::interaction_information_backward);
}
