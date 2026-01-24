// src/torchscience/csrc/meta/linear_algebra/pivoted_lu.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pivoted_lu(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "pivoted_lu: a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "pivoted_lu: a must be floating-point or complex");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a.size(-2);
    int64_t n = a.size(-1);
    int64_t k = std::min(m, n);

    auto dtype = a.scalar_type();

    // L: (..., m, k) - unit lower triangular
    std::vector<int64_t> l_shape = batch_vec;
    l_shape.push_back(m);
    l_shape.push_back(k);

    // U: (..., k, n) - upper triangular
    std::vector<int64_t> u_shape = batch_vec;
    u_shape.push_back(k);
    u_shape.push_back(n);

    // pivots: (..., m) - row permutation for all m rows
    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(m);

    return std::make_tuple(
        at::empty(l_shape, a.options().dtype(dtype)),
        at::empty(u_shape, a.options().dtype(dtype)),
        at::empty(pivot_shape, a.options().dtype(at::kLong)),
        at::empty(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("pivoted_lu", &torchscience::meta::linear_algebra::pivoted_lu);
}
