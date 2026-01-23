#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor conditional_mutual_information(
    const at::Tensor& joint,
    at::IntArrayRef dims_x,
    at::IntArrayRef dims_y,
    at::IntArrayRef dims_z,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = joint.dim();
    TORCH_CHECK(ndim >= 3, "joint must have at least 3 dimensions");
    TORCH_CHECK(
        dims_x.size() == 1 && dims_y.size() == 1 && dims_z.size() == 1,
        "Currently only single-dimension variables are supported"
    );

    int64_t dim_x = dims_x[0] < 0 ? ndim + dims_x[0] : dims_x[0];
    int64_t dim_y = dims_y[0] < 0 ? ndim + dims_y[0] : dims_y[0];
    int64_t dim_z = dims_z[0] < 0 ? ndim + dims_z[0] : dims_z[0];

    // Compute output shape (all dims except x, y, z)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim_x && i != dim_y && i != dim_z) {
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

inline at::Tensor conditional_mutual_information_backward(
    const at::Tensor& grad_output,
    const at::Tensor& joint,
    at::IntArrayRef dims_x,
    at::IntArrayRef dims_y,
    at::IntArrayRef dims_z,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(joint);
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("conditional_mutual_information", &torchscience::meta::information_theory::conditional_mutual_information);
    m.impl("conditional_mutual_information_backward", &torchscience::meta::information_theory::conditional_mutual_information_backward);
}
