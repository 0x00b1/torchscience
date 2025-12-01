#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(dims.size() == 2, "mutual_information: dims must have exactly 2 elements");

    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "mutual_information: dims[0] out of range");
    TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "mutual_information: dims[1] out of range");
    TORCH_CHECK(dim0 != dim1, "mutual_information: dims must be different");

    // Compute output shape (all dims except the two specified)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim0 && i != dim1) {
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

inline at::Tensor mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

inline std::tuple<at::Tensor, at::Tensor> mutual_information_backward_backward(
    const at::Tensor& gg_joint,
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    int64_t dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
    int64_t dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim0 && i != dim1) {
            output_shape.push_back(joint.size(i));
        }
    }

    at::Tensor grad_grad_output;
    if (reduction == "none") {
        if (output_shape.empty()) {
            grad_grad_output = at::empty({}, joint.options());
        } else {
            grad_grad_output = at::empty(output_shape, joint.options());
        }
    } else {
        grad_grad_output = at::empty({}, joint.options());
    }

    return std::make_tuple(grad_grad_output, at::empty_like(joint));
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("mutual_information", &torchscience::meta::information_theory::mutual_information);
    m.impl("mutual_information_backward", &torchscience::meta::information_theory::mutual_information_backward);
    m.impl("mutual_information_backward_backward", &torchscience::meta::information_theory::mutual_information_backward_backward);
}
