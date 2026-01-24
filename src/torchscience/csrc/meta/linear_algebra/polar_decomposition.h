// src/torchscience/csrc/meta/linear_algebra/polar_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polar_decomposition(
    const at::Tensor& a,
    c10::string_view side
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "polar_decomposition: a must be floating-point or complex");
    TORCH_CHECK(side == "right" || side == "left",
        "side must be 'right' or 'left', got '", std::string(side), "'");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a.size(-2);
    int64_t n = a.size(-1);

    auto dtype = a.scalar_type();

    bool is_right = (side == "right");

    // U shape: same as input (..., m, n)
    std::vector<int64_t> u_shape = batch_vec;
    u_shape.push_back(m);
    u_shape.push_back(n);

    // P shape: (..., n, n) for right polar, (..., m, m) for left polar
    std::vector<int64_t> p_shape = batch_vec;
    if (is_right) {
        p_shape.push_back(n);
        p_shape.push_back(n);
    } else {
        p_shape.push_back(m);
        p_shape.push_back(m);
    }

    return std::make_tuple(
        at::empty(u_shape, a.options().dtype(dtype)),
        at::empty(p_shape, a.options().dtype(dtype)),
        at::empty(batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("polar_decomposition", &torchscience::meta::linear_algebra::polar_decomposition);
}
