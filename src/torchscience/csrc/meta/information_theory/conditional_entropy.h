#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor conditional_entropy(
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(ndim >= 2, "conditional_entropy: joint must have at least 2 dimensions");

    if (condition_dim < 0) condition_dim = ndim + condition_dim;
    if (target_dim < 0) target_dim = ndim + target_dim;

    // Output shape excludes both condition and target dims
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != condition_dim && i != target_dim) {
            output_shape.push_back(joint.size(i));
        }
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

inline at::Tensor conditional_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

inline std::tuple<at::Tensor, at::Tensor> conditional_entropy_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    int64_t condition_dim,
    int64_t target_dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(joint));
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("conditional_entropy", &torchscience::meta::information_theory::conditional_entropy);
    m.impl("conditional_entropy_backward", &torchscience::meta::information_theory::conditional_entropy_backward);
    m.impl("conditional_entropy_backward_backward", &torchscience::meta::information_theory::conditional_entropy_backward_backward);
}
