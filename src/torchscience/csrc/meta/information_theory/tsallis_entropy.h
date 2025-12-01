#pragma once

#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::information_theory {

inline at::Tensor tsallis_entropy(
    const at::Tensor& p,
    double q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction
) {
    int64_t ndim = p.dim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "tsallis_entropy: dim out of range");

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

inline at::Tensor tsallis_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    double q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction
) {
    return at::empty_like(p);
}

}  // namespace torchscience::meta::information_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("tsallis_entropy", &torchscience::meta::information_theory::tsallis_entropy);
    m.impl("tsallis_entropy_backward", &torchscience::meta::information_theory::tsallis_entropy_backward);
}
