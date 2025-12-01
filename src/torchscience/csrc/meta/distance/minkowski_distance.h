// src/torchscience/csrc/meta/distance/minkowski_distance.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::distance {

/**
 * Meta implementation for shape inference.
 */
inline at::Tensor minkowski_distance(
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight
) {
    TORCH_CHECK(x.dim() == 2, "minkowski_distance: x must be 2D (m, d)");
    TORCH_CHECK(y.dim() == 2, "minkowski_distance: y must be 2D (n, d)");
    TORCH_CHECK(x.size(1) == y.size(1), "minkowski_distance: feature dimensions must match");

    int64_t m = x.size(0);
    int64_t n = y.size(0);

    return at::empty({m, n}, x.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> minkowski_distance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& y,
    double p,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& dist_output
) {
    at::Tensor grad_w;
    if (weight.has_value() && weight->defined()) {
        grad_w = at::empty_like(*weight);
    }
    return std::make_tuple(
        at::empty_like(x),
        at::empty_like(y),
        grad_w
    );
}

}  // namespace torchscience::meta::distance

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("minkowski_distance", &torchscience::meta::distance::minkowski_distance);
    m.impl("minkowski_distance_backward", &torchscience::meta::distance::minkowski_distance_backward);
}
