#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor chi_squared_divergence(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "chi_squared_divergence: dim out of range");
    TORCH_CHECK(p.sizes() == q.sizes(), "chi_squared_divergence: p and q must have the same shape");

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

inline std::tuple<at::Tensor, at::Tensor> chi_squared_divergence_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    TORCH_CHECK(p.sizes() == q.sizes(), "chi_squared_divergence: p and q must have the same shape");
    return std::make_tuple(
        at::empty_like(p),
        at::empty_like(q)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> chi_squared_divergence_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& reduction
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(p.size(i));
        }
    }

    at::Tensor grad_grad_output;
    if (reduction == "none") {
        if (output_shape.empty()) {
            grad_grad_output = at::empty({}, p.options());
        } else {
            grad_grad_output = at::empty(output_shape, p.options());
        }
    } else {
        grad_grad_output = at::empty({}, p.options());
    }

    return std::make_tuple(
        grad_grad_output,
        at::empty_like(p),
        at::empty_like(q)
    );
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("chi_squared_divergence", &torchscience::meta::information_theory::chi_squared_divergence);
    m.impl("chi_squared_divergence_backward", &torchscience::meta::information_theory::chi_squared_divergence_backward);
    m.impl("chi_squared_divergence_backward_backward", &torchscience::meta::information_theory::chi_squared_divergence_backward_backward);
}
