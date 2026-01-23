// src/torchscience/csrc/meta/linear_algebra/hessenberg.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> hessenberg(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a.size(-1);

    auto dtype = a.scalar_type();
    if (!at::isFloatingType(dtype) && !at::isComplexType(dtype)) {
        dtype = at::kDouble;
    }

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    return std::make_tuple(
        at::empty(mat_shape, a.options().dtype(dtype)),
        at::empty(mat_shape, a.options().dtype(dtype)),
        at::empty(batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("hessenberg", &torchscience::meta::linear_algebra::hessenberg);
}
