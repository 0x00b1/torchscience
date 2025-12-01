// src/torchscience/csrc/meta/distance/hellinger_distance.h
#pragma once

#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::distance {

/**
 * Meta implementation for Hellinger distance shape inference.
 *
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which to compute distance
 * @param input_type Type of input: "probability", "log_probability", or "logits"
 * @param reduction Reduction to apply: "none", "mean", or "sum"
 * @param pairwise If true, compute all-pairs distance matrix
 * @return Empty tensor with correct output shape
 */
inline at::Tensor hellinger_distance(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    if (pairwise) {
        // Pairwise mode: output shape is {m, n} for reduction="none", else scalar
        TORCH_CHECK(
            p.dim() == 2 && q.dim() == 2,
            "hellinger_distance: pairwise mode requires 2D tensors"
        );

        int64_t m = p.size(0);
        int64_t n = q.size(0);

        if (reduction == "none") {
            return at::empty({m, n}, p.options());
        } else {
            return at::empty({}, p.options());
        }
    } else {
        // Standard mode: remove dim from p.sizes()
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
 * Meta implementation for Hellinger distance backward shape inference.
 *
 * @param grad_output Upstream gradient
 * @param p First probability distribution
 * @param q Second probability distribution
 * @param dim Dimension along which distance was computed
 * @param input_type Type of input
 * @param reduction Reduction that was applied
 * @param pairwise Whether pairwise mode was used
 * @return Tuple of empty tensors with shapes matching (grad_p, grad_q)
 */
inline std::tuple<at::Tensor, at::Tensor> hellinger_distance_backward(
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
    m.impl("hellinger_distance", &torchscience::meta::distance::hellinger_distance);
    m.impl("hellinger_distance_backward", &torchscience::meta::distance::hellinger_distance_backward);
}
