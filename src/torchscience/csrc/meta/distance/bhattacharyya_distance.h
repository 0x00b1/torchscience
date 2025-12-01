#pragma once

#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::distance {

/**
 * Meta implementation for Bhattacharyya distance shape inference.
 */
inline at::Tensor bhattacharyya_distance(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    if (pairwise) {
        TORCH_CHECK(
            p.dim() == 2 && q.dim() == 2,
            "bhattacharyya_distance: pairwise mode requires 2D tensors"
        );

        int64_t m = p.size(0);
        int64_t n = q.size(0);

        if (reduction == "none") {
            return at::empty({m, n}, p.options());
        } else {
            return at::empty({}, p.options());
        }
    } else {
        int64_t ndim = p.dim();
        int64_t normalized_dim = dim < 0 ? ndim + dim : dim;

        std::vector<int64_t> output_shape;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != normalized_dim) {
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
}

/**
 * Meta implementation for Bhattacharyya distance backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor> bhattacharyya_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    return std::make_tuple(
        at::empty_like(p),
        at::empty_like(q)
    );
}

}  // namespace torchscience::meta::distance

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("bhattacharyya_distance", &torchscience::meta::distance::bhattacharyya_distance);
    m.impl("bhattacharyya_distance_backward", &torchscience::meta::distance::bhattacharyya_distance_backward);
}
