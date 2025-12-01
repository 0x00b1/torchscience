#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor shannon_entropy(
    const at::Tensor& p,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "shannon_entropy: dim out of range");

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(p.size(i));
        }
    }

    if (reduction == "none") {
        if (output_shape.empty()) {
            return at::empty({}, p.options());
        }
        return at::empty(output_shape, p.options());
    } else {
        return at::empty({}, p.options());
    }
}

inline at::Tensor shannon_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return at::empty_like(p);
}

inline std::tuple<at::Tensor, at::Tensor> shannon_entropy_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    c10::optional<double> base
) {
    return std::make_tuple(at::empty_like(grad_output), at::empty_like(p));
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("shannon_entropy", &torchscience::meta::information_theory::shannon_entropy);
    m.impl("shannon_entropy_backward", &torchscience::meta::information_theory::shannon_entropy_backward);
    m.impl("shannon_entropy_backward_backward", &torchscience::meta::information_theory::shannon_entropy_backward_backward);
}
