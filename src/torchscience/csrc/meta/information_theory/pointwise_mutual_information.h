#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor pointwise_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    // PMI preserves the shape of the input
    return at::empty_like(joint);
}

inline at::Tensor pointwise_mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

inline std::tuple<at::Tensor, at::Tensor> pointwise_mutual_information_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    c10::optional<double> base
) {
    return std::make_tuple(at::empty_like(joint), at::empty_like(joint));
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("pointwise_mutual_information", &torchscience::meta::information_theory::pointwise_mutual_information);
    m.impl("pointwise_mutual_information_backward", &torchscience::meta::information_theory::pointwise_mutual_information_backward);
    m.impl("pointwise_mutual_information_backward_backward", &torchscience::meta::information_theory::pointwise_mutual_information_backward_backward);
}
