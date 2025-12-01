#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor joint_entropy(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();

    // Normalize dims and compute output shape
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        bool is_reduce_dim = false;
        for (auto d : dims) {
            int64_t norm_d = d < 0 ? ndim + d : d;
            if (i == norm_d) {
                is_reduce_dim = true;
                break;
            }
        }
        if (!is_reduce_dim) {
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

inline at::Tensor joint_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

inline std::tuple<at::Tensor, at::Tensor> joint_entropy_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(joint));
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("joint_entropy", &torchscience::meta::information_theory::joint_entropy);
    m.impl("joint_entropy_backward", &torchscience::meta::information_theory::joint_entropy_backward);
    m.impl("joint_entropy_backward_backward", &torchscience::meta::information_theory::joint_entropy_backward_backward);
}
